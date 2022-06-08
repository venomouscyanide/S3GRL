# Adapted from https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/collab/node2vec.py

import torch
from torch_geometric.nn import Node2Vec


def save_embedding(model):
    torch.save(model.embedding.weight.data.cpu(), 'embedding.pt')


def run_and_save_n2v(args, device, data):
    model = Node2Vec(data.edge_index, embedding_dim=args.hidden_channels, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=args.neg_ratio, p=1, q=1, sparse=True).to(device)

    loader = model.loader(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=args.lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

            if (i + 1) % args.log_steps == 0:
                print(f'Epoch: {epoch:02d}, Step: {i + 1:03d}/{len(loader)}, '
                      f'Loss: {loss:.4f}')

            if (i + 1) % 100 == 0:  # Save model every 100 steps.
                save_embedding(model)
        save_embedding(model)
