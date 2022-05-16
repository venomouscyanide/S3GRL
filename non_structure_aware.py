import torch
from ogb.linkproppred import Evaluator
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import MLP
from torch_geometric.utils import negative_sampling


def train_mlp(train, test_data, device, lr, dropout, epochs, dim=256):
    """
    Train for link prediction downstream task
    """
    mlp = MLP(channel_list=[train.x.size(1), dim, dim, dim], dropout=dropout, batch_norm=False).to(device)
    mlp.reset_parameters()
    optimizer = torch.optim.Adam(params=mlp.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    mlp.train()
    for epoch in range(epochs):
        # perform new round of neg sampling per epoch
        neg_edge_index = negative_sampling(
            edge_index=train.edge_index, num_nodes=train.num_nodes,
            num_neg_samples=train.edge_label_index.size(1), method='dense')
        neg_edge_index.to(device)

        edge_label_index = torch.cat(
            [train.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label_index.to(device)

        edge_label = torch.cat([
            train.edge_label,
            train.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)
        edge_label.to(device)

        train.new_edge_label_index = edge_label_index

        out = mlp(train.x)
        link_embedding = out[train.new_edge_label_index[0]] * out[train.new_edge_label_index[1]]  # hadamard product

        loss = criterion(link_embedding.sum(dim=-1), edge_label)
        if epoch % 5 == 0:
            print(f"Loss for epoch {epoch}: {loss.item()}")
        loss.backward()
        optimizer.step()

    # forward pass on test data
    mlp.eval()
    out = mlp(test_data.x)

    test_neg_edge_index = negative_sampling(
        edge_index=test_data.edge_index, num_nodes=test_data.num_nodes,
        num_neg_samples=test_data.edge_label_index.size(1), method='dense')
    test_neg_edge_index.to(device)

    test_edge_label_index = torch.cat(
        [test_data.edge_label_index, test_neg_edge_index],
        dim=-1,
    )
    test_edge_label_index.to(device)

    test_edge_label = torch.cat([
        test_data.edge_label,
        test_data.edge_label.new_zeros(test_neg_edge_index.size(1))
    ], dim=0)
    test_edge_label.to(device)

    test_data.new_edge_label_index = test_edge_label_index

    link_embedding = out[test_data.new_edge_label_index[0]] * out[test_data.new_edge_label_index[1]]
    test_roc = roc_auc_score(test_edge_label.detach().cpu().numpy(),
                             link_embedding.sum(dim=-1).detach().cpu().numpy())
    print(f"Test AUC: {test_roc}")

    return test_roc


def train_mlp_ogbl(train, train_edges, test_edges, device, lr, dropout, epochs, dataset, dim=256):
    """
    Train for link prediction downstream task on the OGBL datasets
    """
    mlp = MLP(channel_list=[train.x.size(1), dim, dim, dim], dropout=dropout, batch_norm=False).to(device)
    mlp.reset_parameters()
    optimizer = torch.optim.Adam(params=mlp.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    mlp.train()
    for epoch in range(epochs):
        # perform new round of neg sampling per epoch
        neg_edge_index = negative_sampling(
            edge_index=train.edge_index,
            num_neg_samples=train_edges['edge'].size(0), method='dense')
        neg_edge_index.to(device)

        edge_label_index = torch.cat(
            [train_edges['edge'].t(), neg_edge_index],
            dim=-1,
        )
        edge_label_index.to(device)

        edge_label = torch.cat([
            train_edges['edge'].t().new_ones(neg_edge_index.size(1)),
            train_edges['edge'].t().new_zeros(neg_edge_index.size(1))
        ], dim=0)
        edge_label.to(device)

        train.new_edge_label_index = edge_label_index

        out = mlp(train.x)
        link_embedding = out[train.new_edge_label_index[0]] * out[train.new_edge_label_index[1]]  # hadamard product

        loss = criterion(link_embedding.sum(dim=-1), edge_label.float())
        if epoch % 5 == 0:
            print(f"Loss for epoch {epoch}: {loss.item()}")
        loss.backward()
        optimizer.step()

    # forward pass on test data
    mlp.eval()
    out = mlp(train.x)

    evaluator = Evaluator(name=dataset)

    edge_label_index = torch.cat(
        [test_edges['edge'].t(), test_edges['edge_neg'].t()],
        dim=-1,
    )
    edge_label_index.to(device)

    edge_label = torch.cat([
        test_edges['edge'].t().new_ones(test_edges['edge'].size(0)),
        test_edges['edge_neg'].t().new_zeros(test_edges['edge_neg'].size(0))
    ], dim=0)
    edge_label.to(device)

    link_embedding = out[edge_label_index[0]] * out[edge_label_index[1]]
    link_embedding = link_embedding.sum(dim=-1)

    test_pred, test_true = link_embedding, edge_label
    pos_test_pred = test_pred[test_true == 1]
    neg_test_pred = test_pred[test_true == 0]

    test_hits = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })
    test_hits = test_hits['hits@50']
    print(f"Test HITS@50: {test_hits}")

    return test_hits
