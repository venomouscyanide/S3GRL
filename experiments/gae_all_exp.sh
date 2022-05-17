#!/bin/sh
# Cora
python seal_link_pred.py --dataset Cora --train_gae --base_gae GCN --epochs 50 --split_val_ratio 0.05 --split_test_ratio 0.10 --runs 5
python seal_link_pred.py --dataset Cora --train_gae --base_gae SAGE --epochs 50 --split_val_ratio 0.05 --split_test_ratio 0.10 --runs 5
# CiteSeer
python seal_link_pred.py --dataset CiteSeer --train_gae --base_gae GCN --epochs 50 --split_val_ratio 0.05 --split_test_ratio 0.10 --runs 5
python seal_link_pred.py --dataset CiteSeer --train_gae --base_gae SAGE --epochs 50 --split_val_ratio 0.05 --split_test_ratio 0.10 --runs 5
# Pubmed
python seal_link_pred.py --dataset Pubmed --train_gae --base_gae GCN --epochs 50 --split_val_ratio 0.05 --split_test_ratio 0.10 --runs 5
python seal_link_pred.py --dataset Pubmed --train_gae --base_gae SAGE --epochs 50 --split_val_ratio 0.05 --split_test_ratio 0.10 --runs 5
# attributed-Facebook
python seal_link_pred.py --dataset attributed-Facebook --train_gae --base_gae GCN --epochs 50 --split_val_ratio 0.05 --split_test_ratio 0.50 --runs 5
python seal_link_pred.py --dataset attributed-Facebook --train_gae --base_gae SAGE --epochs 50 --split_val_ratio 0.05 --split_test_ratio 0.50 --runs 5
