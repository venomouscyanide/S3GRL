import os

import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.nn import Linear, ReLU, BatchNorm1d, Sequential
from torch_geometric import seed_everything
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from torch_geometric.utils import negative_sampling, to_undirected

from torch_geometric.transforms import OneHotDegree

from data_utils import read_label, read_edges
from utils import Logger, do_edge_split


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, layer='GCN'):
        super().__init__()

        if layer == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif layer == 'SAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        elif layer == "GIN":
            self.conv1 = GINConv(Sequential(
                Linear(in_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                BatchNorm1d(hidden_channels),

            ), train_eps=None)
            self.conv2 = GINConv(Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                BatchNorm1d(hidden_channels),

            ), train_eps=None)
        else:
            raise NotImplementedError(f"Layer {layer} not supported")

    def encode(self, x, edge_index, dropout):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=dropout, training=self.training)
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def reset_parameters(self):
        for conv in [self.conv1, self.conv2]:
            conv.reset_parameters()


def train(model, optimizer, train_data, criterion, split_edge, dropout):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index, dropout)

    # We perform a new round of negative sampling for every training epoch:
    pos_edge = split_edge['train']['edge'].t()
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=pos_edge.size(1), method='sparse')

    edge_label_index = torch.cat(
        [pos_edge, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        torch.ones(pos_edge.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(eval_edge_index, eval_neg_edge_index, model, data, dropout):
    model.eval()
    z = model.encode(data.x, data.edge_index, dropout)

    eval_concat_edge_index = torch.cat([eval_edge_index, eval_neg_edge_index], dim=-1)
    eval_labels = torch.cat([
        torch.ones(eval_edge_index.size(1)),
        torch.zeros(eval_neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, eval_concat_edge_index).view(-1).sigmoid()

    precision_score = average_precision_score(eval_labels.cpu().numpy(), out.cpu().numpy())
    auc_score = roc_auc_score(eval_labels.cpu().numpy(), out.cpu().numpy())

    return precision_score, auc_score


def _dataset_creator(args, one_hot_encode):
    file_name = os.path.join('data', 'link_prediction', args.dataset.lower())
    node_id_mapping = read_label(file_name)
    edges = read_edges(file_name, node_id_mapping)

    edges_coo = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data = Data(edge_index=edges_coo.view(2, -1))
    data.edge_index = to_undirected(data.edge_index)
    data.num_nodes = torch.max(data.edge_index) + 1

    split_edge = do_edge_split(data, args.fast_split, val_ratio=args.split_val_ratio,
                               test_ratio=args.split_test_ratio, neg_ratio=args.neg_ratio, data_passed=True)
    data.edge_index = split_edge['train']['edge'].t()

    if one_hot_encode:
        one_hot = OneHotDegree(max_degree=1024)
        data = one_hot(data)
    else:
        data.x = torch.ones(size=(data.num_nodes, 128))

    val_data = split_edge['valid']['edge'].t()
    val_neg = split_edge['valid']['edge_neg'].t()

    test_data = split_edge['test']['edge'].t()
    test_neg = split_edge['test']['edge_neg'].t()

    return data, split_edge, val_data, val_neg, test_data, test_neg


def train_gnn(device, args, one_hot_encode=True):
    log_file = os.path.join(args.res_dir, 'log.txt')

    loggers = {
        'AUC': Logger(args.runs, args),
        'AP': Logger(args.runs, args)
    }
    criterion = torch.nn.BCEWithLogitsLoss()

    for run in range(args.runs):
        seed_everything(run * args.seed)
        data, split_edge, val_data, val_neg, test_data, test_neg = _dataset_creator(args, one_hot_encode)

        model = Net(data.x.size(-1), args.hidden_channels, args.hidden_channels, layer=args.model).to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, optimizer, data, criterion, split_edge, args.dropout)

            if epoch % args.eval_steps == 0:
                val_ap, val_auc = test(val_data, val_neg, model, data, args.dropout)
                test_ap, test_auc = test(test_data, test_neg, model, data, args.dropout)

                results = {}

                results['AUC'] = (val_auc, test_auc)
                results['AP'] = (val_ap, test_ap)

                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        valid_res, test_res = result
                        to_print = (f'Run: {run + 1:02d}, Epoch: {epoch:02d}, ' +
                                    f'Loss: {loss:.4f}, Valid: {100 * valid_res:.2f}%, ' +
                                    f'Test: {100 * test_res:.2f}%')
                        print(key)
                        print(to_print)
                        with open(log_file, 'a') as f:
                            print(key, file=f)
                            print(to_print, file=f)

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    total_params = sum(p.numel() for param in list(model.parameters()) for p in param)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()

    print(f"Total Parameters are: {total_params}")
