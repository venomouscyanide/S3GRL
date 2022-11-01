import json
import sys
import math
import os
from pprint import pprint

import torch_geometric.utils
from tqdm import tqdm
import random
import numpy as np
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
import torch

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling, add_self_loops, train_test_split_edges, to_networkx, subgraph
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils import k_hop_subgraph as org_k_hop_subgraph

from tuned_SIGN import TunedSIGN, OptimizedSignOperations

import graphistry  # only really required for debug. code using graphity is commented by default.


# uncomment to use graphistry to debug data
# graphistry.register(api=3, protocol="https", server="hub.graphistry.com", username="i_see_nodes_everywhere",
#                     password=os.environ['graphistry_pass'])


def neighbors(fringe, A, outgoing=True):
    # Find all 1-hop neighbors of nodes in fringe from graph A, 
    # where A is a scipy csr adjacency matrix.
    # If outgoing=True, find neighbors with outgoing edges;
    # otherwise, find neighbors with incoming edges (you should
    # provide a csc matrix in this case).
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)

    return res


def k_hop_subgraph(src, dst, num_hops, A, sample_ratio=1.0,
                   max_nodes_per_hop=None, node_features=None,
                   y=1, directed=False, A_csc=None, rw_kwargs=None):
    debug = False  # set True manually to debug using matplotlib and gephi
    # Extract the k-hop enclosing subgraph around link (src, dst) from A.
    if not rw_kwargs:
        nodes = [src, dst]
        dists = [0, 0]
        visited = set([src, dst])
        fringe = set([src, dst])
        for dist in range(1, num_hops + 1):
            if not directed:
                fringe = neighbors(fringe, A)
            else:
                out_neighbors = neighbors(fringe, A)
                in_neighbors = neighbors(fringe, A_csc, False)
                fringe = out_neighbors.union(in_neighbors)
            fringe = fringe - visited
            visited = visited.union(fringe)
            if sample_ratio < 1.0:
                fringe = random.sample(fringe, int(sample_ratio * len(fringe)))
            if max_nodes_per_hop is not None:
                if max_nodes_per_hop < len(fringe):
                    fringe = random.sample(fringe, max_nodes_per_hop)
            if len(fringe) == 0:
                break
            nodes = nodes + list(fringe)
            dists = dists + [dist] * len(fringe)

        subgraph = A[nodes, :][:, nodes]

        # Remove target link between the subgraph.
        subgraph[0, 1] = 0
        subgraph[1, 0] = 0

        if node_features is not None:
            node_features = node_features[nodes]

        return nodes, subgraph, dists, node_features, y
    else:
        # Start of core-logic for ScaLed.
        rw_m = rw_kwargs['rw_m']
        rw_M = rw_kwargs['rw_M']
        sparse_adj = rw_kwargs['sparse_adj']
        edge_index = rw_kwargs['edge_index']
        device = rw_kwargs['device']
        data_org = rw_kwargs['data']

        if y == 1:
            cached_rw_sequences = rw_kwargs['cached_pos_rws']
        elif y == 0:
            cached_rw_sequences = rw_kwargs['cached_neg_rws']
        else:
            raise ValueError(f"Value of y is set to {y}, not 0/1")

        if cached_rw_sequences:
            rw_nodes_src = cached_rw_sequences[src]
            rw_nodes_dst = cached_rw_sequences[dst]
            nodes = torch.unique(torch.cat([rw_nodes_src, rw_nodes_dst])).tolist()
        else:
            if rw_kwargs.get('unique_nodes'):
                nodes = rw_kwargs.get('unique_nodes')[(src, dst)]
            else:
                row, col, _ = sparse_adj.csr()
                starting_nodes = torch.tensor([src, dst], dtype=torch.long, device=device)
                start = starting_nodes.repeat(rw_M)
                rw = torch.ops.torch_cluster.random_walk(row, col, start, rw_m, 1, 1)[0]
                if debug:
                    from networkx import write_gexf
                    draw_graph(to_networkx(data_org))
                    write_gexf(torch_geometric.utils.to_networkx(data_org), path='gephi.gexf')
                nodes = torch.unique(rw.flatten()).tolist()

        rw_set = nodes
        # import torch_geometric
        # edge_index_new, edge_attr_new = torch_geometric.utils.subgraph(subset=rw_set, edge_index=edge_index,
        #                                                                relabel_nodes=True)
        # subgraph api is same as org_k_hop_subgraph

        sub_nodes, sub_edge_index, mapping, _ = org_k_hop_subgraph(rw_set, 0, edge_index, relabel_nodes=True,
                                                                   num_nodes=data_org.num_nodes)
        y = torch.tensor([y], dtype=torch.int)
        x = data_org.x[sub_nodes] if hasattr(data_org.x, 'size') else None

        org_src = src
        org_dst = dst
        if rw_kwargs.get('sign'):
            # push src, dst to the front
            nodes.insert(0, nodes.pop(nodes.index(org_dst)))
            nodes.insert(0, nodes.pop(nodes.index(org_src)))

            subgraph = A[nodes, :][:, nodes]

            node_features = node_features[nodes]

            # mask src-target and target-src connections
            subgraph[0, 1] = 0
            subgraph[1, 0] = 0

            dists = torch.ones(size=[len(rw_set)])
            dists = dists.index_fill_(0, torch.tensor([0, 1]), 0).to(int).tolist()

            y = int(y)

            return nodes, subgraph, dists, node_features, y

        src_index = rw_set.index(src)
        dst_index = rw_set.index(dst)
        mapping_list = mapping.tolist()
        src, dst = mapping_list[src_index], mapping_list[dst_index]
        # Remove target link from the subgraph.
        mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
        mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
        sub_edge_index_revised = sub_edge_index[:, mask1 & mask2]

        # Calculate node labeling.
        if rw_kwargs['node_label'] == 'zo':
            z_revised = torch.zeros(size=[len(rw_set)])
            z_revised.index_fill_(0, torch.tensor([0, 1]), 1)
        elif rw_kwargs['node_label'] == 'drnl':
            z_revised = py_g_drnl_node_labeling(sub_edge_index_revised, src, dst,
                                                num_nodes=sub_nodes.size(0))
        else:
            raise NotImplementedError(f"ScaLed does not support {rw_kwargs['node_label']} labeling trick yet.")

        data_revised = Data(x=x, z=z_revised,
                            edge_index=sub_edge_index_revised, y=y, node_id=torch.LongTensor(rw_set),
                            num_nodes=len(rw_set), edge_weight=torch.ones(sub_edge_index_revised.shape[-1]))
        # end of core-logic for ScaLed
        return data_revised


def py_g_drnl_node_labeling(edge_index, src, dst, num_nodes=None):
    # adapted from: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/seal_link_pred.py
    # Double-radius node labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True,
                             indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True,
                             indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = torch.div(dist, 2, rounding_mode='trunc'), dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)


def drnl_node_labeling(adj, src, dst):
    # Double Radius Node Labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = torch.div(dist, 2, rounding_mode='trunc'), dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)


def de_node_labeling(adj, src, dst, max_dist=3):
    # Distance Encoding. See "Li et. al., Distance Encoding: Design Provably More 
    # Powerful Neural Networks for Graph Representation Learning."
    src, dst = (dst, src) if src > dst else (src, dst)

    dist = shortest_path(adj, directed=False, unweighted=True, indices=[src, dst])
    dist = torch.from_numpy(dist)

    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long).t()


def de_plus_node_labeling(adj, src, dst, max_dist=100):
    # Distance Encoding Plus. When computing distance to src, temporarily mask dst;
    # when computing distance to dst, temporarily mask src. Essentially the same as DRNL.
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = torch.cat([dist2src.view(-1, 1), dist2dst.view(-1, 1)], 1)
    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long)


def construct_pyg_graph(node_ids, adj, dists, node_features, y, node_label='drnl', sign_pyg_kwargs=None):
    # Construct a pytorch_geometric graph from a scipy csr adjacency matrix.
    u, v, r = ssp.find(adj)
    num_nodes = adj.shape[0]

    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    # r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    # edge_weight = r.to(torch.float)
    edge_weight = r
    y = torch.tensor([y])
    if node_label == 'drnl':  # DRNL
        z = drnl_node_labeling(adj, 0, 1)
    elif node_label == 'hop':  # mininum distance to src and dst
        z = torch.tensor(dists)
    elif node_label == 'zo':  # zero-one labeling trick
        z = (torch.tensor(dists) == 0).to(torch.long)
    elif node_label == 'de':  # distance encoding
        z = de_node_labeling(adj, 0, 1)
    elif node_label == 'de+':
        z = de_plus_node_labeling(adj, 0, 1)
    elif node_label == 'degree':  # this is technically not a valid labeling trick
        z = torch.tensor(adj.sum(axis=0)).squeeze(0)
        z[z > 100] = 100  # limit the maximum label to 100
    else:
        z = torch.zeros(len(dists), dtype=torch.long)
    if sign_pyg_kwargs:
        # SIGN PyG graph construction flow
        assert node_features is not None, "Node features cannot be None. Check logic."
        node_features = torch.cat([z.reshape(z.size()[0], 1), node_features.to(torch.float)], -1)
        data = Data(node_features, edge_index, edge_weight=edge_weight, y=y, node_id=node_ids, num_nodes=num_nodes)
    else:
        data = Data(node_features, edge_index, edge_weight=edge_weight, y=y, z=z,
                    node_id=node_ids, num_nodes=num_nodes)
    return data


def calc_node_edge_ratio(src, dst, num_hops, A, ratio_per_hop,
                         max_nodes_per_hop, x, y, directed, A_csc, node_label, rw_kwargs, verbose=False):
    # TODO: reuse *.num_nodes and .num_edges
    # calculate the % of nodes/edges in original k-hop vs rw induced graph
    tmp = k_hop_subgraph(src, dst, num_hops, A, ratio_per_hop,
                         max_nodes_per_hop, node_features=x, y=y,
                         directed=directed, A_csc=A_csc)

    data_k_hop = construct_pyg_graph(*tmp, node_label)

    data_rw = k_hop_subgraph(src, dst, num_hops, A, ratio_per_hop,
                             max_nodes_per_hop, node_features=x, y=y,
                             directed=directed, A_csc=A_csc, rw_kwargs=rw_kwargs)
    node_ratio = data_k_hop.num_nodes / data_rw.num_nodes
    try:
        edge_ratio = data_k_hop.num_edges / data_rw.num_edges
    except ZeroDivisionError:
        edge_ratio = 0

    if verbose:
        print(f"\n node ratio: {node_ratio} and edge ratio: {edge_ratio} \n")

    num_nodes_seal = data_k_hop.num_nodes
    num_nodes_sweal = data_rw.num_nodes

    num_edges_seal = data_k_hop.num_edges
    num_edges_sweal = data_rw.num_edges

    return node_ratio, edge_ratio, num_nodes_seal, num_nodes_sweal, num_edges_seal, num_edges_sweal


def calc_ratio_helper(link_index_pos, link_index_neg, A, x, y, num_hops, node_label='drnl',
                      ratio_per_hop=1.0, max_nodes_per_hop=None,
                      directed=False, A_csc=None, rw_kwargs=None, split='train', dataset_name='', seed=1):
    # TODO: this needs to be updated to account for addition of SIGN
    # calculate sparsity of subgraphs of seal vs ScaLed for the split
    stats_dict = {}

    overall_seal_node_storage = []
    overall_sweal_node_storage = []

    overall_seal_edge_storage = []
    overall_sweal_edge_storage = []

    if seed == 1:
        overall_average_seal_node_storage = np.array([], dtype=np.float)
        overall_average_sweal_node_storage = np.array([], dtype=np.float)

        overall_average_seal_edge_storage = np.array([], dtype=np.float)
        overall_average_sweal_edge_storage = np.array([], dtype=np.float)

    else:
        saved_npz = np.load(f'saved_calc_ratio{dataset_name}.npz')

        overall_average_seal_node_storage = saved_npz['overall_average_seal_node_storage']
        overall_average_sweal_node_storage = saved_npz['overall_average_sweal_node_storage']

        overall_average_seal_edge_storage = saved_npz['overall_average_seal_edge_storage']
        overall_average_sweal_edge_storage = saved_npz['overall_average_sweal_edge_storage']

    link_index = torch.cat((link_index_pos, link_index_neg), dim=-1)

    for src, dst in tqdm(link_index.t().tolist()):
        node_ratio, edge_ratio, num_nodes_seal, num_nodes_sweal, num_edges_seal, num_edges_sweal = calc_node_edge_ratio(
            src, dst, num_hops, A, ratio_per_hop, max_nodes_per_hop, x, y, directed, A_csc, node_label, rw_kwargs)

        overall_seal_node_storage = np.append(overall_seal_node_storage, num_nodes_seal)
        overall_sweal_node_storage = np.append(overall_sweal_node_storage, num_nodes_sweal)

        overall_seal_edge_storage = np.append(overall_seal_edge_storage, num_edges_seal)
        overall_sweal_edge_storage = np.append(overall_sweal_edge_storage, num_edges_sweal)

    overall_average_seal_node_storage = np.append(overall_average_seal_node_storage, overall_seal_node_storage.mean())
    overall_average_sweal_node_storage = np.append(overall_average_sweal_node_storage, overall_sweal_node_storage.mean()
                                                   )

    overall_average_seal_edge_storage = np.append(overall_average_seal_edge_storage, overall_seal_edge_storage.mean())
    overall_average_sweal_edge_storage = np.append(overall_average_sweal_edge_storage,
                                                   overall_sweal_edge_storage.mean())

    # sanity check
    assert seed == len(overall_average_seal_node_storage) == len(overall_average_sweal_node_storage) == len(
        overall_average_seal_edge_storage) == len(overall_average_sweal_edge_storage), "Error in saving to npz"

    np.savez(f'saved_calc_ratio{dataset_name}.npz', overall_average_seal_node_storage=overall_average_seal_node_storage,
             overall_average_sweal_node_storage=overall_average_sweal_node_storage,
             overall_average_seal_edge_storage=overall_average_seal_edge_storage,
             overall_average_sweal_edge_storage=overall_average_sweal_edge_storage)

    if seed == 5:
        stats_dict[split] = {

            'SEAL average no of nodes': f'{round(overall_average_seal_node_storage.mean())}',
            'SWEAL average no of nodes': f'{round(overall_average_sweal_node_storage.mean())}',

            'SEAL average no of edges': f'{round(overall_average_seal_edge_storage.mean())}',
            'SWEAL average no of edges': f'{round(overall_average_sweal_edge_storage.mean())}'

        }
        print("--------------------------------------------------------------")
        pprint(stats_dict, sort_dicts=False)
        print("--------------------------------------------------------------")

        os.makedirs('calc_ratio', exist_ok=True)
        with open(f'calc_ratio/preprocessing_stats_{dataset_name}_{split}.json', 'w', encoding='utf-8') as stats_file:
            json.dump(stats_dict, stats_file, ensure_ascii=False)

        os.remove(f'saved_calc_ratio{dataset_name}.npz')


def create_rw_cache(sparse_adj, edges, device, rw_m, rw_M):
    mapped_rw_cache = {}
    row, col, _ = sparse_adj.csr()

    starting_nodes = torch.unique(torch.tensor(edges.flatten(), dtype=torch.long, device=device))
    start = starting_nodes.repeat(rw_M)
    node_ids, _ = torch.ops.torch_cluster.random_walk(row, col, start, rw_m, 1, 1)
    node_ids = node_ids.to('cpu')
    
    for seq in node_ids:
        key = int(seq[0])
        if type(mapped_rw_cache.get(key)) == torch.Tensor:
            values_to_add = torch.cat([mapped_rw_cache[key], seq])
            mapped_rw_cache[key] = torch.unique(values_to_add)
        else:
            mapped_rw_cache[key] = torch.unique(seq)

    return mapped_rw_cache


def extract_enclosing_subgraphs(link_index, A, x, y, num_hops, node_label='drnl',
                                ratio_per_hop=1.0, max_nodes_per_hop=None,
                                directed=False, A_csc=None, rw_kwargs=None, sign_kwargs=None, powers_of_A=None,
                                data=None):
    # Extract enclosing subgraphs from A for all links in link_index.
    data_list = []

    if sign_kwargs:
        if powers_of_A and sign_kwargs['optimize_sign'] and sign_kwargs['sign_type'] == 'hybrid':
            # Keep in mind that in hybrid, you do not need to set sign_k to 5. Just set it at the individual level.
            # ie; the operators per model you want. If you need 5 in total, use sign_k as 3. This is confusing,
            # but, works.
            # optimized hybrid (PoS + SuP) flow. We do not support non-optimized hybrid flow.
            sign_k = sign_kwargs['sign_k']

            print("Prepping SuP data")
            sup_data_list = OptimizedSignOperations.get_SuP_prepped_ds(link_index, num_hops, A, ratio_per_hop,
                                                                       max_nodes_per_hop, directed, A_csc, x, y,
                                                                       sign_kwargs, rw_kwargs)
            if sign_k == 1:
                return sup_data_list

            print("Prepping PoS data")
            pos_data_list = OptimizedSignOperations.get_PoS_prepped_ds(powers_of_A, link_index, A, x, y)

            combined_data = []
            for sup_data, pos_data in zip(sup_data_list, pos_data_list):
                data = sup_data

                for k in range(sign_k + 1, sign_k * 2):
                    data[f'x{k}'] = pos_data[f'x{k - sign_k + 1}']

                combined_data.append(data)
            return combined_data
        elif powers_of_A and sign_kwargs['optimize_sign']:
            # optimized PoS flow
            pos_data_list = OptimizedSignOperations.get_PoS_prepped_ds(powers_of_A, link_index, A, x, y)
            return pos_data_list
        elif not powers_of_A and sign_kwargs['optimize_sign']:
            # optimized SuP flow
            sup_data_list = OptimizedSignOperations.get_SuP_prepped_ds(link_index, num_hops, A, ratio_per_hop,
                                                                       max_nodes_per_hop, directed, A_csc, x, y,
                                                                       sign_kwargs, rw_kwargs)
            return sup_data_list
        elif not sign_kwargs['optimize_sign']:
            # SIGN + SEAL flow; includes both SuP and PoS flows
            for src, dst in tqdm(link_index.t().tolist()):
                if not powers_of_A:
                    # SuP flow

                    # debug code with graphistry
                    # networkx_G = to_networkx(data)  # the full graph
                    # graphistry.bind(source='src', destination='dst', node='nodeid').plot(networkx_G)
                    # check against the nodes that is received in tmp before the relabeling occurs

                    tmp = k_hop_subgraph(src, dst, num_hops, A, ratio_per_hop,
                                         max_nodes_per_hop, node_features=x, y=y,
                                         directed=directed, A_csc=A_csc, rw_kwargs=rw_kwargs)

                    sign_pyg_kwargs = {
                        'use_feature': sign_kwargs['use_feature'],
                    }

                    data = construct_pyg_graph(*tmp, node_label, sign_pyg_kwargs)

                    sign_t = TunedSIGN(sign_kwargs['sign_k'])
                    data = sign_t(data, sign_kwargs['sign_k'])

                    data_list.append(data)
                else:
                    # PoS flow

                    # debug code with graphistry
                    # networkx_G = to_networkx(data)  # the full graph
                    # graphistry.bind(source='src', destination='dst', node='nodeid').plot(networkx_G)
                    # check against the nodes that is received in tmp before the relabeling occurs
                    pos_data_list = []
                    for index, power_of_a in enumerate(powers_of_A, start=1):
                        tmp = k_hop_subgraph(src, dst, num_hops, power_of_a, ratio_per_hop,
                                             max_nodes_per_hop, node_features=x, y=y,
                                             directed=directed, A_csc=A_csc, rw_kwargs=rw_kwargs)
                        sign_pyg_kwargs = {
                            'use_feature': sign_kwargs['use_feature'],
                        }

                        data = construct_pyg_graph(*tmp, node_label, sign_pyg_kwargs)
                        pos_data_list.append(data)

                    sign_t = TunedSIGN(sign_kwargs['sign_k'])
                    data = sign_t.PoS_data_creation(pos_data_list)
                    data_list.append(data)
        else:
            # No match found
            raise NotImplementedError("No matching configuration for model data prep found. Please check code.")
        return data_list

    for src, dst in tqdm(link_index.t().tolist()):
        if not rw_kwargs['rw_m']:
            # SEAL flow
            tmp = k_hop_subgraph(src, dst, num_hops, A, ratio_per_hop,
                                 max_nodes_per_hop, node_features=x, y=y,
                                 directed=directed, A_csc=A_csc)

            data = construct_pyg_graph(*tmp, node_label)
        else:
            # ScaLed flow
            data = k_hop_subgraph(src, dst, num_hops, A, ratio_per_hop,
                                  max_nodes_per_hop, node_features=x, y=y,
                                  directed=directed, A_csc=A_csc, rw_kwargs=rw_kwargs)
        draw = False
        if draw:
            draw_graph(to_networkx(data))
        data_list.append(data)

    return data_list


def do_seal_edge_split(data):
    # this is for datasets involving the WalkPooling paper
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos.t()
    split_edge['train']['edge_neg'] = data.train_neg.t()
    split_edge['valid']['edge'] = data.val_pos.t()
    split_edge['valid']['edge_neg'] = data.val_neg.t()
    split_edge['test']['edge'] = data.test_pos.t()
    split_edge['test']['edge_neg'] = data.test_neg.t()
    return split_edge


def do_edge_split(dataset, fast_split=False, val_ratio=0.05, test_ratio=0.1, neg_ratio=1, data_passed=False):
    if not data_passed:
        data = dataset[0]
    else:
        # for flow involving SEAL datasets, we pass data in dataset arg directly
        data = dataset

    if not fast_split:
        data = train_test_split_edges(data, val_ratio, test_ratio)
        edge_index, _ = add_self_loops(data.train_pos_edge_index)
        data.train_neg_edge_index = negative_sampling(
            edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1) * neg_ratio)
    else:
        raise NotImplementedError('Fast split is untested and unsupported.')
        num_nodes = data.num_nodes
        row, col = data.edge_index
        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]
        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))
        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]
        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
        neg_edge_index = negative_sampling(
            data.edge_index, num_nodes=num_nodes,
            num_neg_samples=row.size(0) * neg_ratio)
        data.val_neg_edge_index = neg_edge_index[:, :n_v]
        data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]
        data.train_neg_edge_index = neg_edge_index[:, n_v + n_t:]

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    return split_edge


def get_pos_neg_edges(split, split_edge, edge_index, num_nodes, percent=100, neg_ratio=1):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge'].t()

        if 'edge_neg' in split_edge['train']:
            # use pre-sampled  negative training edges for ogbl-vessel
            neg_edge = split_edge[split]['edge_neg'].t()
        else:
            new_edge_index, _ = add_self_loops(edge_index)
            neg_edge = negative_sampling(
                new_edge_index, num_nodes=num_nodes,
                num_neg_samples=pos_edge.size(1) * neg_ratio)

        # subsample for pos_edge
        num_pos = pos_edge.size(1)
        perm = np.random.permutation(num_pos)
        perm = perm[:int(percent / 100 * num_pos)]
        pos_edge = pos_edge[:, perm]
        # subsample for neg_edge
        num_neg = neg_edge.size(1)
        perm = np.random.permutation(num_neg)
        perm = perm[:int(percent / 100 * num_neg)]
        neg_edge = neg_edge[:, perm]

    elif 'source_node' in split_edge['train']:
        # TODO: find out what dataset split prompts this flow
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        if split == 'train':
            target_neg = torch.randint(0, num_nodes, [target.size(0), 1],
                                       dtype=torch.long)
        else:
            target_neg = split_edge[split]['target_node_neg']
        # subsample
        num_source = source.size(0)
        perm = np.random.permutation(num_source)
        perm = perm[:int(percent / 100 * num_source)]
        source, target, target_neg = source[perm], target[perm], target_neg[perm, :]
        pos_edge = torch.stack([source, target])
        neg_per_target = target_neg.size(1)
        neg_edge = torch.stack([source.repeat_interleave(neg_per_target),
                                target_neg.view(-1)])
    return pos_edge, neg_edge


def CN(A, edge_index, batch_size=100000):
    # The Common Neighbor heuristic score.
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
    return torch.FloatTensor(np.concatenate(scores, 0)), edge_index


def AA(A, edge_index, batch_size=100000):
    # The Adamic-Adar heuristic score.
    multiplier = 1 / np.log(A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index


def PPR(A, edge_index):
    # The Personalized PageRank heuristic score.
    # Need install fast_pagerank by "pip install fast-pagerank"
    # Too slow for large datasets now.
    from fast_pagerank import pagerank_power
    num_nodes = A.shape[0]
    src_index, sort_indices = torch.sort(edge_index[0])
    dst_index = edge_index[1, sort_indices]
    edge_index = torch.stack([src_index, dst_index])
    # edge_index = edge_index[:, :50]
    scores = []
    visited = set([])
    j = 0
    for i in tqdm(range(edge_index.shape[1])):
        if i < j:
            continue
        src = edge_index[0, i]
        personalize = np.zeros(num_nodes)
        personalize[src] = 1
        ppr = pagerank_power(A, p=0.85, personalize=personalize, tol=1e-7)
        j = i
        while edge_index[0, j] == src:
            j += 1
            if j == edge_index.shape[1]:
                break
        all_dst = edge_index[1, i:j]
        cur_scores = ppr[all_dst]
        if cur_scores.ndim == 0:
            cur_scores = np.expand_dims(cur_scores, 0)
        scores.append(np.array(cur_scores))

    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def add_info(self, epochs, runs):
        self.epochs = epochs
        self.runs = runs

    def print_statistics(self, run=None, f=sys.stdout):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 0].argmax().item()
            print(f'Run {run + 1:02d}:', file=f)
            print(f'Highest Valid: {result[:, 0].max():.2f}', file=f)
            print(f'Highest Eval Point: {argmax + 1}', file=f)
            print(f'Highest Test: {result[argmax, 1]:.2f}', file=f)
            print(f'Average Test: {result.T[1].mean():.2f} ± {result.T[1].std():.2f}', file=f)
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                valid = r[:, 0].max().item()
                test = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:', file=f)
            r = best_result[:, 0]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}', file=f)
            r = best_result[:, 1]
            print(f'Highest Test: {r.mean():.2f} ± {r.std():.2f}', file=f)

            best_test = r[0]

            print(f'\n(Precision of 5)Highest Test: {r.mean():.5f} ± {r.std():.5f}\n', file=f)
            if hasattr(self, 'epochs'):
                # logger won't have epochs while running heuristic models
                r_revised = torch.reshape(result, (self.epochs * self.runs, 2))[:, 1]
                print(f'Average Test: {r_revised.mean():.2f} ± {r_revised.std():.2f}', file=f)
            return best_test


def draw_graph(graph):
    # helps draw a graph object and save it as a png file
    f = plt.figure(1, figsize=(48, 48))
    nx.draw(graph, with_labels=True, pos=nx.spring_layout(graph))
    plt.show()  # check if same as in the doc visually
    f.savefig("input_graph.pdf", bbox_inches='tight')


# https://stackoverflow.com/a/45846841/12918863
def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
