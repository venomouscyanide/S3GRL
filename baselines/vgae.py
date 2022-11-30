# Adapted from the authors of WalkPool: https://github.com/DaDaCheng/WalkPooling/blob/main/software/vgae.py
# and https://github.com/pyg-team/pytorch_geometric/blob/master/examples/argva_node_clustering.py
# and https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py
import os

import torch
from torch_geometric.nn import VGAE, GCNConv, GAE, ARGVA
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
from utils import Logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ARGVAEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv_mu = GCNConv(hidden_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, out_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels1)
        self.lin2 = torch.nn.Linear(hidden_channels1, hidden_channels2)
        self.lin3 = torch.nn.Linear(hidden_channels2, out_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return x


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv_mu = GCNConv(hidden_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


def compute_scores(z, test_pos, test_neg):
    test = torch.cat((test_pos, test_neg), dim=1)
    labels = torch.zeros(test.size(1), 1)
    labels[0:test_pos.size(1)] = 1
    row, col = test
    src = z[row]
    tgt = z[col]
    scores = torch.sigmoid(torch.sum(src * tgt, dim=1))
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return auc, ap


def run_vgae(edge_index, x, test_and_val, model, args):
    print('___Calculating VGAE embbeding___')

    log_file = os.path.join(args.res_dir, 'log.txt')
    run = 0
    loggers = {
        'AUC': Logger(1, args),
        'AP': Logger(1, args)
    }

    test_pos, test_neg, val_pos, val_neg = test_and_val
    out_channels = int(args.embedding_dim)
    num_features = x.size(1)

    if model == 'GAE':
        model = GAE(GCNEncoder(num_features, out_channels, args.hidden_channels)).to(device)
    elif model == 'VGAE':
        model = VGAE(VariationalGCNEncoder(num_features, out_channels, args.hidden_channels)).to(device)
    elif model == 'ARGVA':
        encoder = ARGVAEncoder(x.size(1), hidden_channels=args.hidden_channels, out_channels=args.embedding_dim)
        discriminator = Discriminator(in_channels=args.embedding_dim, hidden_channels1=args.hidden_channels // 4,
                                      hidden_channels2=args.hidden_channels,
                                      out_channels=out_channels)
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),
                                                   lr=args.lr)
        model = ARGVA(encoder, discriminator).to(device)
    else:
        raise NotImplementedError(f"Model f{model} is not supported.")

    edge_index = edge_index.to(device)
    x = x.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_nodes = torch.max(edge_index)

    for epoch in range(1, args.epochs + 1, 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, edge_index)

        if model == 'ARGVA':
            for i in range(5):
                discriminator_optimizer.zero_grad()
                discriminator_loss = model.discriminator_loss(z)
                discriminator_loss.backward()
                discriminator_optimizer.step()

        loss = model.recon_loss(z, edge_index)
        if model in ['ARGVA', 'VGAE']:
            loss = loss + (1 / num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        if epoch % args.eval_steps == 0:
            model.eval()
            z = model.encode(x, edge_index)
            z = z.cpu().clone().detach()

            results = {}

            val_ap, val_auc = compute_scores(z, val_pos, val_neg)
            test_ap, test_auc = compute_scores(z, test_pos, test_neg)

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

    total_params = sum(p.numel() for param in list(model.parameters()) for p in param)
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
