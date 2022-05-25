import torch
import os

from torch_geometric.utils import to_undirected, from_scipy_sparse_matrix, is_undirected
from torch_geometric.data import Data

import numpy as np


# floor(), load_splitted_data() and  load_unsplitted_data()
# are adapted from WalkPooling: https://github.com/DaDaCheng/WalkPooling
def floor(x):
    return torch.div(x, 1, rounding_mode='trunc')


def load_splitted_data(data_name, data_split_num, test_ratio, val_ratio):
    data_name = f'{data_name}_split_{data_split_num}'
    if test_ratio == 0.5:
        data_dir = os.path.join('data/splitted_0_5/{}.mat'.format(data_name))
    else:
        data_dir = os.path.join('data/splitted/{}.mat'.format(data_name))
    import scipy.io as sio
    print('Load data from: ' + data_dir)
    net = sio.loadmat(data_dir)
    data = Data()

    data.train_pos = torch.from_numpy(np.int64(net['train_pos']))
    data.train_neg = torch.from_numpy(np.int64(net['train_neg']))
    data.test_pos = torch.from_numpy(np.int64(net['test_pos']))
    data.test_neg = torch.from_numpy(np.int64(net['test_neg']))

    n_pos = floor(val_ratio * len(data.train_pos)).int()
    nlist = np.arange(len(data.train_pos))
    np.random.shuffle(nlist)
    val_pos_list = nlist[:n_pos]
    train_pos_list = nlist[n_pos:]
    data.val_pos = data.train_pos[val_pos_list]
    data.train_pos = data.train_pos[train_pos_list]

    n_neg = floor(val_ratio * len(data.train_neg)).int()
    nlist = np.arange(len(data.train_neg))
    np.random.shuffle(nlist)
    val_neg_list = nlist[:n_neg]
    train_neg_list = nlist[n_neg:]
    data.val_neg = data.train_neg[val_neg_list]
    data.train_neg = data.train_neg[train_neg_list]

    data.val_pos = torch.transpose(data.val_pos, 0, 1)
    data.val_neg = torch.transpose(data.val_neg, 0, 1)
    data.train_pos = torch.transpose(data.train_pos, 0, 1)
    data.train_neg = torch.transpose(data.train_neg, 0, 1)
    data.test_pos = torch.transpose(data.test_pos, 0, 1)
    data.test_neg = torch.transpose(data.test_neg, 0, 1)
    num_nodes = max(torch.max(data.train_pos), torch.max(data.test_pos)) + 1
    num_nodes = max(num_nodes, torch.max(data.val_pos) + 1)
    data.num_nodes = num_nodes

    return data


def load_unsplitted_data(args):
    # read .mat format files
    data_dir = os.path.join('data/{}.mat'.format(args.data_name))
    print('Load data from: ' + data_dir)
    import scipy.io as sio
    net = sio.loadmat(data_dir)
    edge_index, _ = from_scipy_sparse_matrix(net['net'])
    data = Data(edge_index=edge_index)
    if is_undirected(data.edge_index) == False:  # in case the dataset is directed
        data.edge_index = to_undirected(data.edge_index)
    data.num_nodes = torch.max(data.edge_index) + 1
    return data


# read_edges() and read_label() are adapted from DE: https://github.com/snap-stanford/distance-encoding/
def read_edges(seal_ds_path, node_id_mapping):
    edges = []
    fin_edges = open(os.path.join(seal_ds_path, 'edges.txt'))
    for line in fin_edges.readlines():
        node1, node2 = line.strip().split()[:2]
        edges.append([node_id_mapping[node1], node_id_mapping[node2]])
    fin_edges.close()
    return edges


def read_label(seal_ds_path):
    nodes = []
    with open(os.path.join(seal_ds_path, 'edges.txt')) as ef:
        for line in ef.readlines():
            nodes.extend(line.strip().split()[:2])
    nodes = sorted(list(set(nodes)))
    node_id_mapping = {old_id: new_id for new_id, old_id in enumerate(nodes)}
    return node_id_mapping
