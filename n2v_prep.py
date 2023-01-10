# Adapted from WalkPool codebase: https://github.com/DaDaCheng/WalkPooling/blob/main/software/node2vec.py
from pathlib import Path

import torch
from torch_geometric.nn import Node2Vec
from tqdm import tqdm
import os


def node_2_vec_pretrain(dataset, edge_index, num_nodes, emb_dim, seed, device, epochs, hypertuning=False,
                        extra_identifier=''):
    if hypertuning:
        emb_folder = f'{Path.home()}/Emb'
    else:
        emb_folder = "Emb"
    if not os.path.exists(emb_folder):
        os.makedirs(emb_folder)

    unique_identifier = f"{emb_folder}/{dataset}_{emb_dim}_seed{seed}_{extra_identifier}.pt"
    if os.path.exists(unique_identifier):
        print("Using cached n2v embeddings. Skipping n2v pretraining.")
        return torch.load(unique_identifier, map_location=torch.device('cpu')).detach()

    n2v = Node2Vec(edge_index, num_nodes=num_nodes, embedding_dim=emb_dim, walk_length=20,
                   context_size=10, walks_per_node=10,
                   num_negative_samples=1, p=1, q=1, sparse=True).to(device)
    loader = n2v.loader(batch_size=32, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(n2v.parameters()), lr=0.01)
    n2v.train()

    print(f'Prepping n2v embeddings with hidden_dim: {emb_dim}')
    for i in tqdm(range(epochs), ncols=70):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = n2v.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if i % 20 == 0:
            print(f'Step: {i} /200, Loss : {total_loss:.4f}')

    output = (n2v.forward()).cpu().clone().detach()

    print('Finish prepping n2v embeddings')
    torch.save(output, unique_identifier)
    return output
