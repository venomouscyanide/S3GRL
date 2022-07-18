# Standalone code that helps examine Golden Operator Matrices as graphs
import copy
import random

import torch
from networkx import Graph
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.utils import coalesce, add_self_loops, negative_sampling, train_test_split_edges, to_dense_adj
from torch_sparse import SparseTensor
import scipy.sparse as ssp
from tqdm import tqdm
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import numpy as np

from torch_geometric.transforms import SIGN


def do_edge_split(dataset, fast_split=False, val_ratio=0.05, test_ratio=0.1, neg_ratio=1, data_passed=False):
    if not data_passed:
        data = dataset[0]
    else:
        # for flow involving SEAL datasets, we pass data in dataset arg directly
        data = dataset

    data = train_test_split_edges(data, val_ratio, test_ratio)
    edge_index, _ = add_self_loops(data.train_pos_edge_index)
    data.train_neg_edge_index = negative_sampling(
        edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1) * neg_ratio)

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    return split_edge


def get_pos_neg_edges(split, split_edge, edge_index, num_nodes, percent=100, neg_ratio=1):
    pos_edge = split_edge[split]['edge'].t()
    if split == 'train':
        new_edge_index, _ = add_self_loops(edge_index)
        neg_edge = negative_sampling(
            new_edge_index, num_nodes=num_nodes,
            num_neg_samples=pos_edge.size(1) * neg_ratio)
    else:
        neg_edge = split_edge[split]['edge_neg'].t()
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

    return pos_edge, neg_edge


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


class TunedSIGN(SIGN):
    """
    Helps pop the keys that are not consumed during training
    """

    def __call__(self, data, sign_k):
        data = super().__call__(data)
        if sign_k == -1:
            for idx in range(1, self.K):
                data.pop(f'x{idx}')
        return data


def k_hop_subgraph(src, dst, num_hops, A, sample_ratio=1.0,
                   max_nodes_per_hop=None, node_features=None,
                   y=1, directed=False, A_csc=None, rw_kwargs=None):
    # Extract the k-hop enclosing subgraph around link (src, dst) from A.
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


def construct_pyg_graph(node_ids, adj, dists, node_features, y, node_label='drnl', sign_pyg_kwargs=None):
    # Construct a pytorch_geometric graph from a scipy csr adjacency matrix.
    u, v, r = ssp.find(adj)
    num_nodes = adj.shape[0]

    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    edge_weight = r.to(torch.float)
    y = torch.tensor([y])

    z = (torch.tensor(dists) == 0).to(torch.long)
    if sign_pyg_kwargs:
        # SIGN PyG graph construction flow
        if sign_pyg_kwargs['use_feature'] and node_features is not None:
            node_features = torch.cat([z.reshape(z.size()[0], 1), node_features.to(torch.float)], -1)
        else:
            # flow never really enters here due to check in main()
            node_features = z
        data = Data(node_features, edge_index, edge_weight=edge_weight, y=y, node_id=node_ids, num_nodes=num_nodes)
    else:
        data = Data(node_features, edge_index, edge_weight=edge_weight, y=y, z=z,
                    node_id=node_ids, num_nodes=num_nodes)
    return data


def extract_enclosing_subgraphs(link_index, A, x, y, num_hops, node_label='zo',
                                ratio_per_hop=1.0, max_nodes_per_hop=None,
                                directed=False, A_csc=None, rw_kwargs=None, sign_kwargs=None):
    # Extract enclosing subgraphs from A for all links in link_index.
    data_list = []
    for src, dst in tqdm(link_index.t().tolist()):
        num_hops = 1  # restrict to 1, then taken powers of A
        tmp = k_hop_subgraph(src, dst, num_hops, A, ratio_per_hop,
                             max_nodes_per_hop, node_features=x, y=y,
                             directed=directed, A_csc=A_csc)

        sign_pyg_kwargs = {
            'use_feature': sign_kwargs['use_feature'],
        }
        data = construct_pyg_graph(*tmp, node_label, sign_pyg_kwargs)

        sign_t = TunedSIGN(sign_kwargs['num_layers'])
        data = sign_t(data, sign_kwargs['sign_k'])

        data_list.append(data)

    return data_list


class SEALDataset(InMemoryDataset):
    def __init__(self, root, data, split_edge, num_hops, sign_k, percent=100, split='train',
                 use_coalesce=False, node_label='drnl', ratio_per_hop=1.0,
                 max_nodes_per_hop=None, directed=False, rw_kwargs=None, device='cpu', pairwise=False,
                 pos_pairwise=False, neg_ratio=1, use_feature=False, args=None):
        # TODO: avoid args, use the exact arguments instead
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = int(percent) if percent >= 1.0 else percent
        self.split = split
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = directed
        self.device = device
        self.N = self.data.num_nodes
        self.E = self.data.edge_index.size()[-1]
        self.sparse_adj = SparseTensor(
            row=self.data.edge_index[0].to(self.device), col=self.data.edge_index[1].to(self.device),
            value=torch.arange(self.E, device=self.device),
            sparse_sizes=(self.N, self.N))
        self.rw_kwargs = rw_kwargs
        self.pairwise = pairwise
        self.pos_pairwise = pos_pairwise
        self.neg_ratio = neg_ratio
        self.use_feature = use_feature
        self.args = args
        self.sign_k = sign_k
        super(SEALDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.percent == 100:
            name = 'SEAL_{}_data'.format(self.split)
        else:
            name = 'SEAL_{}_data_{}'.format(self.split, self.percent)
        name += '.pt'
        return [name]

    def process(self):
        pos_edge, neg_edge = get_pos_neg_edges(self.split, self.split_edge,
                                               self.data.edge_index,
                                               self.data.num_nodes,
                                               self.percent, neg_ratio=self.neg_ratio)

        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight,
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        A = ssp.csr_matrix(
            (edge_weight, (self.data.edge_index[0], self.data.edge_index[1])),
            shape=(self.data.num_nodes, self.data.num_nodes)
        )

        if self.directed:
            A_csc = A.tocsc()
        else:
            A_csc = None

        # Extract enclosing subgraphs for pos and neg edges

        rw_kwargs = {}
        sign_kwargs = {}

        num_layers = self.num_hops
        sign_kwargs.update({
            "sign_k": self.sign_k,
            "use_feature": self.use_feature,
            "num_layers": num_layers,
        })

        print("Setting up Positive Subgraphs")
        pos_list = extract_enclosing_subgraphs(
            pos_edge, A, self.data.x, 1, self.num_hops, self.node_label,
            self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc, rw_kwargs, sign_kwargs)
        print("Setting up Negative Subgraphs")
        neg_list = extract_enclosing_subgraphs(
            neg_edge, A, self.data.x, 0, self.num_hops, self.node_label,
            self.ratio_per_hop, self.max_nodes_per_hop, self.directed, A_csc, rw_kwargs, sign_kwargs)
        torch.save(self.collate(pos_list + neg_list), self.processed_paths[0])
        del pos_list, neg_list


if __name__ == '__main__':
    # fix seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    # examine some A matrices with and without self-loops
    # the code is extracted from the extended ScaLed code base keeping only the skeletal elements
    num_hops = 3
    sign_k = 3
    path = 'datasets'
    dataset_name = 'Cora'

    dataset = Planetoid(path, dataset_name)
    split_edge = do_edge_split(dataset, False, val_ratio=0.05, test_ratio=0.1, neg_ratio=1)
    data = dataset[0]
    data.edge_index = split_edge['train']['edge'].t()

    train_dataset = SEALDataset(root=path, data=data, split_edge=split_edge, num_hops=num_hops, sign_k=sign_k,
                                use_feature=True, node_label='zo')

    # matplotlib.use("Agg")

    loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    for g in loader:
        if g.edge_index.nelement() != 0:
            dense_adj = to_dense_adj(g.edge_index).reshape([g.num_nodes, g.num_nodes])
        else:
            dense_adj = torch.zeros(size=(g.num_nodes, g.num_nodes))
        all_powers = [dense_adj]
        for power in range(2, sign_k + 1):
            all_powers.append(torch.linalg.matrix_power(dense_adj, power))

        for index, ajc_power in enumerate(all_powers, start=1):
            f = plt.figure(figsize=(5, 5))
            limits = plt.axis('off')
            node_size = 100  # cut-off to viz better

            with_labels = True
            G = Graph(ajc_power.detach().numpy(), )
            labels = {i: str(i + 1) for i in range(len(G))}

            print(f"Drawing graph {index} of {sign_k}")
            nx.draw(G, node_size=node_size, arrows=False, with_labels=with_labels, labels=labels)
            f.show()
            plt.show()
