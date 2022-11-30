import os
from os import path as osp
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB
from torch_geometric.utils import to_undirected

from data_utils import read_label, read_edges
from utils import do_edge_split


def get_data_helper(args):
    if args.dataset in ['Cora', 'Pubmed', 'CiteSeer']:
        path = osp.join('dataset', args.dataset)
        dataset = Planetoid(path, args.dataset)
        split_edge = do_edge_split(dataset, args.fast_split, val_ratio=args.split_val_ratio,
                                   test_ratio=args.split_test_ratio, neg_ratio=args.neg_ratio)
        data = dataset[0]
        data.edge_index = split_edge['train']['edge'].t()
        import networkx as nx
        G = nx.Graph()
        G.add_edges_from(data.edge_index.T.detach().numpy())
    elif args.dataset in ['USAir', 'NS', 'Power', 'Celegans', 'Router', 'PB', 'Ecoli', 'Yeast']:
        # We consume the dataset split index as well
        if os.path.exists('data'):
            file_name = os.path.join('data', 'link_prediction', args.dataset.lower())
        else:
            # we consume user path
            file_name = os.path.join(str(Path.home()), 'ExtendoScaLed', 'data', 'link_prediction', args.dataset.lower())
            if not os.path.exists(file_name):
                raise FileNotFoundError("Clone repo in your user path")
        node_id_mapping = read_label(file_name)
        edges = read_edges(file_name, node_id_mapping)

        import networkx as nx
        G = nx.Graph(edges)
        edges_coo = torch.tensor(edges, dtype=torch.long).t().contiguous()
        data = Data(edge_index=edges_coo.view(2, -1))
        data.edge_index = to_undirected(data.edge_index)
        data.num_nodes = torch.max(data.edge_index) + 1

        split_edge = do_edge_split(data, args.fast_split, val_ratio=args.split_val_ratio,
                                   test_ratio=args.split_test_ratio, neg_ratio=args.neg_ratio, data_passed=True)
        data.edge_index = split_edge['train']['edge'].t()
        print("Finish reading from file")
    elif args.dataset in ['chameleon', 'crocodile', 'squirrel']:
        path = osp.join('dataset', args.dataset)
        dataset = WikipediaNetwork(path, args.dataset)
        split_edge = do_edge_split(dataset, args.fast_split, val_ratio=args.split_val_ratio,
                                   test_ratio=args.split_test_ratio, neg_ratio=args.neg_ratio)
        data = dataset[0]
        data.edge_index = split_edge['train']['edge'].t()
        import networkx as nx
        G = nx.Graph()
        G.add_edges_from(data.edge_index.T.detach().numpy())
    elif args.dataset in ['Cornell', 'Texas', 'Wisconsin']:
        path = osp.join('dataset', args.dataset)
        dataset = WebKB(path, args.dataset)
        split_edge = do_edge_split(dataset, args.fast_split, val_ratio=args.split_val_ratio,
                                   test_ratio=args.split_test_ratio, neg_ratio=args.neg_ratio)
        data = dataset[0]
        data.edge_index = split_edge['train']['edge'].t()
        import networkx as nx
        G = nx.Graph()
        G.add_edges_from(data.edge_index.T.detach().numpy())
    else:
        raise NotImplementedError(f'dataset {args.dataset} is not yet supported.')

    return data, split_edge
