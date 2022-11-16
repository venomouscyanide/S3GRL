# Adapted from WalkPool codebase: https://github.com/DaDaCheng/WalkPooling/blob/main/software/node2vec.py
import argparse

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, AttributedGraphDataset
from torch_geometric.nn import Node2Vec
from tqdm import tqdm
from data_utils import read_label, read_edges
from torch_geometric.utils import to_undirected
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric import seed_everything

import os.path as osp
import os
from utils import do_edge_split

parser = argparse.ArgumentParser(description='')
# Dataset settings
parser.add_argument('--dataset', type=str, default='ogbl-collab')
parser.add_argument('--fast_split', action='store_true',
                    help="for large custom datasets (not OGB), do a fast data split")
parser.add_argument('--neg_ratio', type=int, default=1,
                    help="Compile neg_ratio times the positive samples for compiling neg_samples"
                         "(only for Training data)")
parser.add_argument('--split_val_ratio', type=float, default=0.05)
parser.add_argument('--split_test_ratio', type=float, default=0.1)
# Node feature settings.
# deg means use node degree. one means use homogeneous embeddings.
# nodeid means use pretrained node embeddings in ./Emb
parser.add_argument('--use_deg', action='store_true')
parser.add_argument('--use_one', action='store_true')
parser.add_argument('--use_nodeid', action='store_true')
parser.add_argument('--hidden_channels', type=int, default=256)
# Train settings
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--path', type=str, default="Emb/")
parser.add_argument('--name', type=str, default="opt")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--use_seed', action='store_true')
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

seed_everything(args.seed)

def node_2_vec_pretrain(edge_index, num_nodes, emb_dim, device):
    n2v = Node2Vec(edge_index, num_nodes=num_nodes, embedding_dim=emb_dim, walk_length=20,
                   context_size=10, walks_per_node=10,
                   num_negative_samples=1, p=1, q=1, sparse=True).to(device)
    loader = n2v.loader(batch_size=32, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(n2v.parameters()), lr=0.01)
    n2v.train()

    print(f'Prepping n2v embeddings with hidden_dim: {emb_dim}')
    for i in tqdm(range(201), ncols=70):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = n2v.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if i % 20 == 0:
            print(f'Step: {i} /200, Loss : {total_loss:.4f}')

    output = (n2v.forward()).cpu().clone().detach()

    del n2v
    del loader
    torch.cuda.empty_cache()

    print('Finish prepping n2v embeddings')
    torch.save(output, f"{args.path}{args.dataset}_{emb_dim}_seed{args.seed}.pt")


if args.dataset.startswith('ogbl'):
    dataset = PygLinkPropPredDataset(name=args.dataset)
    split_edge = dataset.get_edge_split()
    data = dataset[0]

    if args.dataset.startswith('ogbl-vessel'):
        # normalize node features
        data.x[:, 0] = torch.nn.functional.normalize(data.x[:, 0], dim=0)
        data.x[:, 1] = torch.nn.functional.normalize(data.x[:, 1], dim=0)
        data.x[:, 2] = torch.nn.functional.normalize(data.x[:, 2], dim=0)

elif args.dataset.startswith('attributed'):
    dataset_name = args.dataset.split('-')[-1]
    path = osp.join('dataset', dataset_name)
    dataset = AttributedGraphDataset(path, dataset_name)
    split_edge = do_edge_split(dataset, args.fast_split, val_ratio=args.split_val_ratio,
                               test_ratio=args.split_test_ratio, neg_ratio=args.neg_ratio)
    data = dataset[0]
    data.edge_index = split_edge['train']['edge'].t()

elif args.dataset in ['Cora', 'Pubmed', 'CiteSeer']:
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
    file_name = os.path.join('data', 'link_prediction', args.dataset.lower())
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


    # backward compatibility
    class DummyDataset:
        def __init__(self, root):
            self.root = root
            self.num_features = 0

        def __repr__(self):
            return args.dataset

        def __len__(self):
            return 1

    dataset = DummyDataset(root=f'dataset/{args.dataset}/SEALDataset_{args.dataset}')
    print("Finish reading from file")
else:
    raise NotImplementedError(f'dataset {args.dataset} is not yet supported.')

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

node_2_vec_pretrain(data.edge_index, data.num_nodes, args.hidden_channels, device)
