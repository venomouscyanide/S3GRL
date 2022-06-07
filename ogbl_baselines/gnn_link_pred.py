# adapted from: https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/collab/gnn.py
import torch
from ogb.linkproppred import Evaluator

from torch.nn import Linear, ReLU, Sequential
from torch_geometric import seed_everything

import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GINConv

from torch_geometric.transforms import OneHotDegree
from utils import Logger


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


def train(model, optimizer, data, split_edge, dropout, batch_size):
    model.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()

        edge = pos_train_edge[perm].t()
        h = model.encode(data.x, edge, dropout)

        pos_out = model.decode(h, edge)
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long,
                             device=h.device)
        neg_out = model.decode(h, edge)
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, data, split_edge, evaluator, batch_size, dropout, device):
    model.eval()

    pos_valid_edge = split_edge['valid']['edge'].to(device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(device)
    pos_test_edge = split_edge['test']['edge'].to(device)
    neg_test_edge = split_edge['test']['edge_neg'].to(device)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        h = model.encode(data.x, edge, dropout)
        out = model.decode(h, edge).sigmoid()
        pos_valid_preds += [out.squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        h = model.encode(data.x, edge, dropout)
        out = model.decode(h, edge).sigmoid()
        neg_valid_preds += [out.squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        h = model.encode(data.x, edge, dropout)
        out = model.decode(h, edge).sigmoid()
        pos_test_preds += [out.squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        h = model.encode(data.x, edge, dropout)
        out = model.decode(h, edge).sigmoid()
        neg_test_preds += [out.squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results


def train_gae_ogbl(device, data, split_edge, args, one_hot_encode=False):
    evaluator = Evaluator(name='ogbl-collab')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }

    if not args.use_feature:
        if one_hot_encode:
            one_hot = OneHotDegree(max_degree=1024)
            data = one_hot(data)
        else:
            # if no features, we simply set x to be identity matrix as seen in GAE paper
            data.x = torch.eye(data.num_nodes)

    for run in range(args.runs):
        seed_everything(args.seed)
        data.to(device)

        model = Net(data.x.size(-1), args.hidden_channels, layer=args.model).to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, optimizer, data, split_edge, args.dropout, args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(model, data, split_edge, evaluator, args.batch_size, args.dropout, device)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    total_params = sum(p.numel() for param in list(model.parameters()) for p in param)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()

    print(f"Total Parameters are: {total_params}")
