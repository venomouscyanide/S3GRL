#!/bin/sh
cd SWEAL_OGB/collab_exp
# ogbl-collab MLP
python mlp_collab.py --lr 0.0001 --dropout 0.5 --epochs 500 --runs 5
# ogbl-collab GCN
python gnn_collab.py --use_valedges_as_input --lr 0.0001 --dropout 0.5 --epochs 50 --runs 5
# ogbl-collab SAGE
python gnn_collab.py --use_valedges_as_input --use_sage --lr 0.0001 --dropout 0.5 --epochs 50 --runs 5