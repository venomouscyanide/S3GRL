import torch
from scipy.sparse import dok_matrix
from torch_geometric.data import Data
from torch_geometric.transforms import SIGN
from torch_sparse import SparseTensor, from_scipy, spspmm
import torch.nn.functional as F
from tqdm import tqdm
from torch_scatter import scatter_max
import scipy.sparse as ssp
import numpy as np


class TunedSIGN(SIGN):
    """
    Custom SIGN class for SuP and PoS
    """

    def __call__(self, data, sign_k):
        data = super().__call__(data)
        if sign_k == -1:
            for idx in range(1, self.K):
                data.pop(f'x{idx}')
        return data

    def PoS_data_creation(self, pos_data_list):
        original_data = pos_data_list[0]

        for index, data in enumerate(pos_data_list, start=1):
            assert data.edge_index is not None
            row, col = data.edge_index
            adj_t = SparseTensor(row=col, col=row, value=torch.tensor(data.edge_weight),
                                 sparse_sizes=(data.num_nodes, data.num_nodes))

            assert data.x is not None

            original_data[f'x{index}'] = (adj_t @ data.x)
        max_dim = (original_data[f'x'].size()[0])

        for created_op in range(1, self.K + 1):
            max_dim = max(max_dim, original_data[f'x{created_op}'].size()[0])

        for operator in range(1, self.K + 1):
            if original_data[f'x{operator}'].size()[0] == max_dim:
                continue
            original_data[f'x{operator}'] = F.pad(original_data[f'x{operator}'],
                                                  (0, 0, 0, max_dim - original_data[f'x{operator}'].size()[0]))

        if original_data['x'].size()[0] < max_dim:
            original_data['x'] = F.pad(original_data[f'x'],
                                       (0, 0, 0, max_dim - original_data[f'x'].size()[0]))

        # the following keys are useless in SIGN-esque training
        del original_data['node_id']
        del original_data['num_nodes']
        del original_data['edge_index']
        del original_data['edge_weight']

        return original_data


class OptimizedSignOperations:
    @staticmethod
    def get_PoS_prepped_ds(powers_of_A, link_index, A, x, y, sign_kwargs):
        print("PoS Optimized Flow.")
        # optimized PoS flow, everything is created on the CPU, then in train() sent to GPU on a batch basis

        pos_data_list = []

        a_global_list = []
        g_global_list = []
        normalized_powers_of_A = powers_of_A
        g_h_global_list = []

        list_of_training_edges = link_index.t().tolist()

        K = sign_kwargs['sign_k']
        pos_num_A = 1000  # sign_kwargs['pos_num_graphs']

        list_of_powers_A = []
        list_of_training_edges_split = np.array_split(list_of_training_edges, pos_num_A)

        A = normalized_powers_of_A[-1].to_scipy().tolil()

        for index, list_of_edges in enumerate(list_of_training_edges_split):
            A_copy = A.copy()

            for edge in list_of_edges:
                A_copy[edge[0], edge[1]] = 0
                A_copy[edge[1], edge[0]] = 0

            list_of_powers_inner = [A_copy]
            for sign_k in range(2, K + 1):
                list_of_powers_inner.append((A_copy @ list_of_powers_inner[-1]).tolil())
            list_of_powers_A.extend(list_of_powers_inner)


        print("Setting up A Global List")
        split_helper = []
        [split_helper.extend([idx] * K) for idx in range(pos_num_A)]

        for index, power_of_a in enumerate(list_of_powers_A, start=0):
            print(f"Constructing A[{index}]")
            num_training_egs = len(list_of_training_edges_split[split_helper[index]])
            a_global_list.append(
                dok_matrix((num_training_egs * 2, A.shape[0]), dtype=np.float32)
            )
            list_of_lilmtrx = []
            power_of_a_scipy_lil = power_of_a
            for link in tqdm(list_of_training_edges_split[split_helper[index]], ncols=70):
                src, dst = link[0], link[1]
                interim_src = power_of_a_scipy_lil.getrow(src)
                interim_dst = power_of_a_scipy_lil.getrow(dst)
                list_of_lilmtrx.append(interim_src)
                list_of_lilmtrx.append(interim_dst)

            to_update = a_global_list[index]
            print("Converting to DOK")
            for overall_row, item in tqdm(enumerate(list_of_lilmtrx), ncols=70):
                data = item.data
                rows = item.rows

                to_update[overall_row, rows[0]] = data[0]

            idx, values = from_scipy(a_global_list[index])
            a_global_list[index] = torch.sparse_coo_tensor(idx, values, size=[num_training_egs * 2, A.shape[0]],
                                                           dtype=torch.float32)
        print("Setting up G Global List")
        original_x = x.detach()
        x = x.to_sparse()
        for operator_id in tqdm(range(len(a_global_list)), ncols=70):
            mult_index, mult_value = spspmm(a_global_list[operator_id].coalesce().indices(),
                                            a_global_list[operator_id].coalesce().values(), x.indices(),
                                            x.values(), a_global_list[operator_id].size()[0],
                                            a_global_list[operator_id].size()[1],
                                            x.size()[1])
            g_global_list.append(
                torch.sparse_coo_tensor(mult_index, mult_value, size=[a_global_list[operator_id].size()[0],
                                                                      x.size()[-1]]).to_dense())

        print("Setting up G H Global List")
        for index, src_dst_x in enumerate(g_global_list, start=0):
            num_training_egs = len(list_of_training_edges_split[split_helper[index]])
            g_h_global_list.append(
                torch.empty(size=[num_training_egs * 2, g_global_list[split_helper[index]].shape[-1] + 1]))
            print(f"Setting up G H Global [{index}]")
            for inner_index, link in tqdm(enumerate(list_of_training_edges_split[split_helper[index]]), ncols=70):
                src, dst = link[0], link[1]
                h_src = torch.tensor(list_of_powers_A[index][src, src] + list_of_powers_A[index][src, dst])
                h_dst = torch.tensor(list_of_powers_A[index][dst, dst] + list_of_powers_A[index][dst, src])
                g_h_global_list[index][inner_index] = torch.hstack(
                    [h_src, g_global_list[index][inner_index]])
                g_h_global_list[index][inner_index + 1] = torch.hstack(
                    [h_dst, g_global_list[index][inner_index + 1]])

        print("Finishing Prep with creation of data")
        x = original_x

        split_gh = np.array_split(g_h_global_list, pos_num_A)

        for link_list, gh_data in tqdm(zip(list_of_training_edges_split, split_gh), ncols=70):
            src_counter = 0
            dst_counter = 1

            for link in link_list:
                src, dst = link[0], link[1]
                data = Data(
                    x=torch.hstack(
                        [torch.tensor([[1], [1]]),
                         torch.vstack([x[src], x[dst]]),
                         ]),
                    y=y,
                )

                for global_index, all_i_operators in enumerate(gh_data):
                    src_features = all_i_operators[src_counter]
                    dst_features = all_i_operators[dst_counter]
                    subgraph_features = torch.vstack([src_features, dst_features])

                    data[f'x{global_index + 1}'] = subgraph_features
                    data.edge_index = (src, dst)
                src_counter += 2
                dst_counter += 2

                pos_data_list.append(data)
        return pos_data_list

    @staticmethod
    def get_SuP_prepped_ds(link_index, num_hops, A, ratio_per_hop, max_nodes_per_hop, directed, A_csc, x, y,
                           sign_kwargs, rw_kwargs):
        # optimized SuP flow
        print("SuP Optimized Flow.")
        from utils import k_hop_subgraph
        sup_data_list = []
        print("Start with SuP data prep")
        for src, dst in tqdm(link_index.t().tolist()):
            tmp = k_hop_subgraph(src, dst, num_hops, A, ratio_per_hop,
                                 max_nodes_per_hop, node_features=x, y=y,
                                 directed=directed, A_csc=A_csc, rw_kwargs=rw_kwargs)

            u, v, r = ssp.find(tmp[1])
            u, v = torch.LongTensor(u), torch.LongTensor(v)
            adj_t = SparseTensor(row=u, col=v,
                                 sparse_sizes=(tmp[1].shape[0], tmp[1].shape[0]))

            deg = adj_t.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

            subgraph_features = tmp[3]
            subgraph = adj_t

            if subgraph.size(1) == 0 and subgraph.size(0) == 0:
                # empty enclosing subgraph
                continue

            assert subgraph_features is not None

            powers_of_a = [subgraph]
            K = sign_kwargs['sign_k']

            for _ in range(K - 1):
                powers_of_a.append(subgraph @ powers_of_a[-1])

            all_a_values = torch.empty(size=[K * 2, subgraph.size(0)])

            # construct A ( (K * 2) X num_nodes)
            for operator_index in range(0, K * 2, 2):
                all_a_values[[operator_index, operator_index + 1], :] = torch.tensor(
                    powers_of_a[operator_index // 2][[0, 1], :].to_dense()
                )
            # calculate AX ( (K * 2) X num_input_feat)
            all_ax_values = all_a_values @ subgraph_features

            # inject label info into AX' ( (K * 2) X (num_input_feat + 1))
            updated_features = torch.empty(size=[K * 2, all_ax_values[0].size()[-1] + 1])
            for operator_index in range(0, K * 2, 2):
                label_src = all_a_values[operator_index][0] + all_a_values[operator_index][1]
                label_dst = all_a_values[operator_index + 1][0] + all_a_values[operator_index + 1][1]

                updated_features[operator_index, :] = torch.hstack([label_src, all_ax_values[operator_index]])
                updated_features[operator_index + 1, :] = torch.hstack(
                    [label_dst, all_ax_values[operator_index + 1]])

            # convert AX' into PyG Data object
            data = Data(
                x=torch.hstack(
                    [torch.tensor([[1], [1]]),
                     torch.vstack([subgraph_features[0], subgraph_features[1]]),
                     ]),
                y=y,
            )

            for operator_index in range(0, K * 2, 2):
                data[f'x{operator_index // 2 + 1}'] = torch.vstack(
                    [updated_features[operator_index], updated_features[operator_index + 1]]
                )

            sup_data_list.append(data)

        return sup_data_list

    @staticmethod
    def get_KSuP_prepped_ds(link_index, num_hops, A, ratio_per_hop, max_nodes_per_hop, directed, A_csc, x, y,
                            sign_kwargs, rw_kwargs):
        # optimized k-heuristic SuP flow
        print("K Heuristic SuP Optimized Flow.")
        from utils import k_hop_subgraph, neighbors
        sup_data_list = []
        print("Start with SuP data prep")

        K = sign_kwargs['sign_k']

        for src, dst in tqdm(link_index.t().tolist()):
            tmp = k_hop_subgraph(src, dst, num_hops, A, ratio_per_hop,
                                 max_nodes_per_hop, node_features=x, y=y,
                                 directed=directed, A_csc=A_csc, rw_kwargs=rw_kwargs)
            csr_subgraph = tmp[1]
            u, v, r = ssp.find(csr_subgraph)
            u, v = torch.LongTensor(u), torch.LongTensor(v)
            adj_t = SparseTensor(row=u, col=v,
                                 sparse_sizes=(csr_subgraph.shape[0], csr_subgraph.shape[0]))

            deg = adj_t.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

            un_norm_powers = [csr_subgraph]
            for _ in range(K - 1):
                un_norm_powers.append(csr_subgraph @ un_norm_powers[-1])

            subgraph_features = tmp[3]
            subgraph = adj_t

            assert subgraph_features is not None

            powers_of_a = [subgraph]

            for _ in range(K - 1):
                powers_of_a.append(subgraph @ powers_of_a[-1])

            # source, target is always 0, 1
            strat = sign_kwargs['k_node_set_strategy']
            strat_hop_nodes = {}
            for sign_k in range(K):
                subgraph = un_norm_powers[sign_k]
                if strat == 'union':
                    one_hop_nodes = neighbors({0}, subgraph).union(neighbors({1}, subgraph))
                elif strat == 'intersection':
                    one_hop_nodes = neighbors({0}, subgraph).intersection(neighbors({1}, subgraph))
                else:
                    raise NotImplementedError(f"check strat {strat}")
                strat_hop_nodes[sign_k] = one_hop_nodes

            all_a_values = []
            # construct all a rows
            starting_indices = []
            slice_helper = []
            len_so_far = 0
            for sign_k_val in range(0, K, 1):
                selected_rows = strat_hop_nodes[sign_k_val]
                selected_rows.discard(0)
                selected_rows.discard(1)
                selected_rows = [0, 1] + list(selected_rows)

                rows_of_op = powers_of_a[sign_k_val][list(selected_rows)].to_dense()
                all_a_values.extend(rows_of_op)

                starting_indices += [len_so_far, len_so_far + 1]
                slice_helper.append(len_so_far)

                len_so_far += len(selected_rows)

            all_a_values = torch.stack((all_a_values))
            # calculate AX
            all_ax_values = all_a_values @ subgraph_features

            # inject label info
            row = 0
            final_a_values = torch.hstack([torch.zeros(size=(all_ax_values.size()[0], 1)), all_ax_values])
            source_target_indices = set(starting_indices)
            while row < len(all_ax_values):
                if row in source_target_indices:
                    label = all_a_values[row][0] + all_a_values[row][1]
                    final_a_values[row][0] = label

                row += 1

            if strat == 'union':
                x_a = torch.tensor([[1]] + [[1]] + [[0] for _ in range(subgraph_features.size(0) - 2)])
                x_b = subgraph_features
            elif strat == 'intersection':
                x_a = torch.tensor([[1]] + [[1]])
                x_b = subgraph_features[[0, 1]]
            else:
                raise NotImplementedError(f"check strat {strat}")

            data = Data(
                x=torch.hstack([x_a, x_b]),
                y=y,
            )

            index = 0
            for index, (start, end) in enumerate(zip(slice_helper, slice_helper[1:]), start=1):
                x_operator = final_a_values[start:end]
                data[f'x{index}'] = x_operator
            data[f'x{index + 1}'] = final_a_values[slice_helper[-1]:]

            sup_data_list.append(data)

        return sup_data_list
