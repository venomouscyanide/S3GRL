import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.nn import Linear, ReLU, Sequential
from torch_geometric import seed_everything

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from torch_geometric.utils import negative_sampling

from baselines.baseline_utils import get_data_helper
from utils import Logger

import os

import torch


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, layer='GCN'):
        super().__init__()

        if layer == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
        elif layer == 'SAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        elif layer == "GIN":
            self.conv1 = GINConv(Sequential(
                Linear(in_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
            ), train_eps=False)
            self.conv2 = GINConv(Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
            ), train_eps=False)
            self.conv3 = GINConv(Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
            ), train_eps=False)
        else:
            raise NotImplementedError(f"Layer {layer} not supported")

    def encode(self, x, edge_index, dropout):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=dropout, training=self.training)
        return self.conv3(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def reset_parameters(self):
        for conv in [self.conv1, self.conv2, self.conv3]:
            conv.reset_parameters()


def train(model, optimizer, train_data, criterion, split_edge, dropout, device):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index, dropout)

    # We perform a new round of negative sampling for every training epoch:
    pos_edge = split_edge['train']['edge'].t()
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=pos_edge.size(1), method='sparse')

    edge_label_index = torch.cat(
        [pos_edge.to(device), neg_edge_index.to(device)],
        dim=-1,
    ).to(device)
    edge_label = torch.cat([
        torch.ones(pos_edge.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ], dim=0).to(device)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(eval_edge_index, eval_neg_edge_index, model, data, dropout, device):
    model.eval()
    z = model.encode(data.x, data.edge_index, dropout)

    eval_concat_edge_index = torch.cat([eval_edge_index.to(device), eval_neg_edge_index.to(device)], dim=-1).to(device)
    eval_labels = torch.cat([
        torch.ones(eval_edge_index.size(1)),
        torch.zeros(eval_neg_edge_index.size(1))
    ], dim=0).to(device)

    out = model.decode(z, eval_concat_edge_index).view(-1).sigmoid()

    precision_score = average_precision_score(eval_labels.cpu().numpy(), out.cpu().numpy())
    auc_score = roc_auc_score(eval_labels.cpu().numpy(), out.cpu().numpy())

    return precision_score, auc_score


def train_gnn(device, args):
    log_file = os.path.join(args.res_dir, 'log.txt')

    criterion = torch.nn.BCEWithLogitsLoss()

    all_runs_auc = []
    for run in range(1, args.runs + 1, 1):
        seed_everything(run)
        run = 0  # ensure consistency for the Logger obj
        loggers = {
            'AUC': Logger(1, args),
            'AP': Logger(1, args)
        }
        data, split_edge = get_data_helper(args)

        if not data.x:
            # if no features, we simply set x to be identity matrix as seen in GAE paper
            data.x = torch.eye(data.num_nodes)

        val_data = split_edge['valid']['edge'].t()
        val_neg = split_edge['valid']['edge_neg'].t()
        test_data = split_edge['test']['edge'].t()
        test_neg = split_edge['test']['edge_neg'].t()

        data.to(device)

        model = Net(data.x.size(-1), args.hidden_channels, layer=args.model).to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, optimizer, data, criterion, split_edge, args.dropout, device)

            if epoch % args.eval_steps == 0:
                val_ap, val_auc = test(val_data, val_neg, model, data, args.dropout, device)
                test_ap, test_auc = test(test_data, test_neg, model, data, args.dropout, device)

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

        best_test_scores = []
        for key in loggers.keys():
            print(key)
            loggers[key].add_info(args.epochs, 1)
            best_test_scores += [loggers[key].print_statistics()]
            with open(log_file, 'a') as f:
                print(key, file=f)
                loggers[key].print_statistics(f=f)
        all_runs_auc.append(best_test_scores[0])

    total_params = sum(p.numel() for param in list(model.parameters()) for p in param)
    print(f"Total Parameters are: {total_params}")

    array = np.array(all_runs_auc)
    print(f'Final Average Test of {args.runs} runs is: {array.mean():.2f} ± {array.std():.2f}')
