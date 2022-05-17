# adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/link_pred.py
import torch
from ogb.linkproppred import Evaluator
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling

from gae_link_pred import Net


def train(model, optimizer, train_data, train_edges, criterion):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_edges['edge'].t(),
        num_neg_samples=train_edges['edge'].t().size(1), method='dense')

    edge_label_index = torch.cat(
        [train_edges['edge'].t(), neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_edges['edge'].t().new_ones(neg_edge_index.size(1)),
        train_edges['edge'].t().new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label.float())
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(train_data, edges, model, evaluator):
    model.eval()
    edge_label_index = torch.cat(
        [edges['edge'].t(), edges['edge_neg'].t()],
        dim=-1,
    )
    z = model.encode(train_data.x, train_data.edge_index)
    edge_labels = torch.cat([
        edges['edge'].t().new_ones(edges['edge'].t().size(1)),
        edges['edge_neg'].t().new_zeros(edges['edge_neg'].t().size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1).sigmoid()

    # if you want to see the AUC score uncomment below
    # auc = roc_auc_score(edge_labels, out.cpu().numpy())

    test_hits = evaluator.eval({
        'y_pred_pos': out[:len(edges['edge'])],
        'y_pred_neg': out[len(edges['edge_neg']):],
    })
    test_hits = test_hits['hits@50']

    return test_hits


def gae_train_helper_ogbl(dataset, device, train_data, train_edges, val_edges, test_edges, lr, epochs, layer):
    model = Net(dataset.num_features, 256, 256, 256, layer).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_auc = final_test_acc = 0
    evaluator = Evaluator(name=dataset.name)
    for epoch in range(1, epochs):
        loss = train(model, optimizer, train_data, train_edges, criterion)
        val_acc = test(train_data, val_edges, model, evaluator)
        test_acc = test(train_data, test_edges, model, evaluator)
        if val_acc > best_val_auc:
            best_val = val_acc
        final_test_acc = test_acc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc * 100:.4f}, '
              f'Test: {test_acc * 100:.4f}')

    print(f'Final Test: {final_test_acc * 100:.4f}')
    return final_test_acc
