# Adapted from WalkPool codebase: https://github.com/DaDaCheng/WalkPooling/blob/main/software/node2vec.py

import torch
from torch_geometric.nn import Node2Vec
from tqdm import tqdm


def node_2_vec_pretrain(edge_index, emd_dim, device):
    n2v = Node2Vec(edge_index, embedding_dim=emd_dim, walk_length=20,
                   context_size=10, walks_per_node=10,
                   num_negative_samples=1, p=1, q=1, sparse=True).to(device)
    loader = n2v.loader(batch_size=32, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(n2v.parameters()), lr=0.01)
    n2v.train()

    print('Prepping n2v embeddings')
    for i in tqdm(range(201), ncols=70):
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

    del n2v
    del loader
    torch.cuda.empty_cache()

    print('Finish prepping n2v embeddings')
    return output
