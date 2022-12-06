# adapted from: https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/collab/mf.py
import os

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from torch_geometric import seed_everything

from utils import Logger


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(predictor, x, split_edge, optimizer, batch_size):
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        edge = pos_train_edge[perm].t()

        pos_out = predictor(x[edge[0]], x[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, x.size(0), edge.size(), dtype=torch.long,
                             device=x.device)
        neg_out = predictor(x[edge[0]], x[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(predictor, x, split_edge, device, batch_size):
    predictor.eval()

    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0).to(device)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0).to(device)

    val_labels = torch.cat([
        torch.ones(split_edge['valid']['edge'].size(0)),
        torch.zeros(split_edge['valid']['edge'].size(0))
    ], dim=0).to(device)
    all_valid_preds = torch.cat([pos_valid_pred, neg_valid_pred], dim=0).to(device)

    val_auc = roc_auc_score(val_labels.cpu().numpy(), all_valid_preds.cpu().numpy())
    val_ap = average_precision_score(val_labels.cpu().numpy(), all_valid_preds.cpu().numpy())

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        perm_preds = predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()
        if not perm_preds.shape:
            perm_preds = perm_preds.reshape(1)
        pos_test_preds += [perm_preds]

    pos_test_pred = torch.cat(pos_test_preds, dim=0).to(device)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        perm_preds = predictor(x[edge[0]], x[edge[1]]).squeeze().cpu()
        if not perm_preds.shape:
            perm_preds = perm_preds.reshape(1)
        neg_test_preds += [perm_preds]
    neg_test_pred = torch.cat(neg_test_preds, dim=0).to(device)

    test_labels = torch.cat([
        torch.ones(split_edge['test']['edge'].size(0)),
        torch.zeros(split_edge['test']['edge'].size(0))
    ], dim=0).to(device)
    all_test_preds = torch.cat([pos_test_pred, neg_test_pred], dim=0).to(device)

    test_auc = roc_auc_score(test_labels.cpu().numpy(), all_test_preds.cpu().numpy())
    test_ap = average_precision_score(test_labels.cpu().numpy(), all_test_preds.cpu().numpy())

    return val_ap, val_auc, test_ap, test_auc


def train_mf(data, split_edge, device, log_steps, num_layers, hidden_channels, dropout, batch_size, lr, epochs,
             eval_steps, runs, seed, args):
    seed_everything(seed)
    emb = torch.nn.Embedding(data.num_nodes, hidden_channels).to(device)
    predictor = LinkPredictor(hidden_channels, hidden_channels, 1,
                              num_layers, dropout).to(device)

    loggers = {
        'AUC': Logger(runs, args),
        'AP': Logger(runs, args)
    }
    log_file = os.path.join(args.res_dir, 'log.txt')

    for run in range(runs):
        seed_everything(seed)
        emb.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(emb.parameters()) + list(predictor.parameters()), lr=lr)

        for epoch in range(1, 1 + epochs):
            loss = train(predictor, emb.weight, split_edge, optimizer,
                         batch_size)
            if epoch % eval_steps == 0:
                val_ap, val_auc, test_ap, test_auc = test(predictor, emb.weight, split_edge, device, batch_size)

                results = {}

                results['AUC'] = (val_auc, test_auc)
                results['AP'] = (val_ap, test_ap)

                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % log_steps == 0:
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

    total_params = sum(p.numel() for param in list(predictor.parameters()) for p in param)

    print(f"Total Parameters are: {total_params}")
    best_test_scores = []
    for key in loggers.keys():
        print(key)
        loggers[key].add_info(args.epochs, 1)
        best_test_scores += [loggers[key].print_statistics()]
        with open(log_file, 'a') as f:
            print(key, file=f)
            loggers[key].print_statistics(f=f)
    return best_test_scores[0]
