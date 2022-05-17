# adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/link_pred.py
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, out_channels, layer):
        super().__init__()
        self.conv1 = layer(in_channels, hidden_channels1)
        self.conv2 = layer(hidden_channels1, hidden_channels2)
        self.conv3 = layer(hidden_channels2, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


def train(model, optimizer, train_data, criterion, device):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse').to(device)

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    ).to(device)
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0).to(device)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(data, model):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


def gae_train_helper(dataset, device, train_data, val_data, test_data, lr, epochs, layer):
    model = Net(dataset.num_features, 256, 256, 256, layer).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    train_data.to(device)

    best_val_auc = final_test_auc = 0
    for epoch in range(1, epochs):
        loss = train(model, optimizer, train_data, criterion, device)
        val_auc = test(val_data, model)
        test_auc = test(test_data, model)
        if val_auc > best_val_auc:
            best_val = val_auc
            final_test_auc = test_auc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
              f'Test: {test_auc:.4f}')

    print(f'Final Test: {final_test_auc:.4f}')
    return final_test_auc
