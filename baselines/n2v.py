# motivated & adapted from https://colab.research.google.com/github/stellargraph/stellargraph/blob/master/demos/link-prediction/node2vec-link-prediction.ipynb#scrollTo=LlVABgrwIzBd
# and https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py
import os

import torch

from torch_geometric import seed_everything
from torch_geometric.nn import Node2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from utils import Logger


def run_n2v(device, data, split_edge, epochs, lr, hidden_channels, neg_ratio, batch_size, num_threads, args):
    loggers = {
        'AUC': Logger(args.runs, args),
        'AP': Logger(args.runs, args)
    }
    log_file = os.path.join(args.res_dir, 'log.txt')

    test_edges = torch.cat([split_edge['test']['edge'], split_edge['test']['edge_neg']], dim=0).to(device)
    test_labels = torch.cat([
        torch.ones(split_edge['test']['edge'].size(0)),
        torch.zeros(split_edge['test']['edge'].size(0))
    ], dim=0).to(device)

    val_edges = torch.cat([split_edge['valid']['edge'], split_edge['valid']['edge_neg']], dim=0).to(device)
    val_labels = torch.cat([
        torch.ones(split_edge['valid']['edge'].size(0)),
        torch.zeros(split_edge['valid']['edge'].size(0))
    ], dim=0).to(device)

    for run in range(args.runs):
        model = Node2Vec(data.edge_index, embedding_dim=hidden_channels, walk_length=20,
                         context_size=10, walks_per_node=10,
                         num_negative_samples=neg_ratio, p=1, q=1, sparse=True).to(device)

        loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=num_threads)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=lr)

        seed_everything(args.seed)

        for epoch in range(epochs):
            loss = train(model, optimizer, loader, device)

            if epoch % args.eval_steps == 0:
                clf = train_link_classifier(device, hidden_channels, model, split_edge)

                val_ap, val_auc = get_auc(val_edges, model, clf, val_labels, hidden_channels)
                test_ap, test_auc = get_auc(test_edges, model, clf, test_labels, hidden_channels)

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


@torch.no_grad()
def train_link_classifier(device, hidden_channels, model, split_edge):
    clf = link_prediction_classifier()
    train_edges = torch.cat([split_edge['train']['edge'], split_edge['train']['edge_neg']]).cpu().numpy()
    train_labels = torch.cat([
        torch.ones(split_edge['train']['edge'].size(0)),
        torch.zeros(split_edge['train']['edge'].size(0))
    ], dim=0).to(device)
    link_features = link_examples_to_features(train_edges, model, hidden_channels)
    clf.fit(link_features, train_labels)
    return clf


@torch.no_grad()
def get_auc(edges, model, clf, labels, hidden_channels):
    link_features = link_examples_to_features(edges, model, hidden_channels)
    auc = roc_auc_score(labels, clf.predict(link_features))
    ap = average_precision_score(labels, clf.predict(link_features))
    return ap, auc


def link_examples_to_features(link_examples, model, hidden_channels):
    return [get_link_emb(src, dst, model, hidden_channels) for src, dst in link_examples]


def get_link_emb(src, dst, model, hidden_channels):
    had = model(src) * model(dst)
    return had.reshape(hidden_channels).cpu().numpy()


def link_prediction_classifier():
    lr = LogisticRegression()
    return lr


def train(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
