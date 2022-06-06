# Adapted from https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/collab/node2vec.py
import os

import torch
from ogb.linkproppred import Evaluator
from sklearn.linear_model import LogisticRegression
from torch_geometric import seed_everything
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import negative_sampling

from baselines.n2v import train, link_examples_to_features
from utils import Logger


@torch.no_grad()
def test(model, split_edge, device, hidden_channels, evaluator):
    model.eval()

    pos_train_edge = split_edge['train']['edge'].to(device)
    neg_train_edge = negative_sampling(torch.tensor(split_edge['train']['edge']).t()).to(device)
    pos_valid_edge = split_edge['valid']['edge'].to(device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(device)
    pos_test_edge = split_edge['test']['edge'].to(device)
    neg_test_edge = split_edge['test']['edge_neg'].to(device)

    clf = LogisticRegression()
    link_features = link_examples_to_features(torch.cat([pos_train_edge, neg_train_edge.t()], dim=0).to(device), model,
                                              hidden_channels)
    labels = torch.cat([
        torch.ones(split_edge['train']['edge'].size(0)),
        torch.zeros(neg_train_edge.t().size(0))
    ], dim=0).to(device)

    clf.fit(link_features, labels.cpu().numpy())

    val_link_features = link_examples_to_features(torch.cat([pos_valid_edge, neg_valid_edge], dim=0).to(device), model,
                                                  hidden_channels)
    predictions = clf.predict(val_link_features)
    val_all_preds = torch.tensor(predictions).to(device)
    pos_valid_pred = val_all_preds[:split_edge['valid']['edge'].size(0)]
    neg_valid_pred = val_all_preds[split_edge['valid']['edge'].size(0):]

    test_link_features = link_examples_to_features(torch.cat([pos_test_edge, neg_test_edge], dim=0).to(device), model,
                                                   hidden_channels)
    predictions = clf.predict(test_link_features)
    test_all_preds = torch.tensor(predictions).to(device)
    pos_test_pred = test_all_preds[:split_edge['test']['edge'].size(0)]
    neg_test_pred = test_all_preds[split_edge['test']['edge'].size(0):]

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


def run_n2v_ogbl(device, data, split_edge, epochs, lr, hidden_channels, neg_ratio, batch_size, num_threads, args):
    evaluator = Evaluator(name='ogbl-collab')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }

    log_file = os.path.join(args.res_dir, 'log.txt')

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
                results = test(model, split_edge, device, hidden_channels, evaluator)

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
