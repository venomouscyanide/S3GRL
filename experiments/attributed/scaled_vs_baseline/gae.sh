#!/bin/bash
# this will help reproduce scaled vs baselines results for GAE models(with base GCN, SAGE and GIN) for Cora, CiteSeer
# all runs are on 5 different seeds(1 run each of every seed). Different seed ensure different set of initial weight and dataset splits

# GCN on Cora
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --train_gae --model GCN --epochs 50 --seed $SEED --use_feature
done

# SAGE on Cora
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --train_gae --model SAGE --epochs 50 --seed $SEED --use_feature
done

# GIN on Cora
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --train_gae --model GIN --epochs 50 --seed $SEED --use_feature
done

# GCN on CiteSeer
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --train_gae --model GCN --epochs 50 --seed $SEED --use_feature
done

# SAGE on CiteSeer
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --train_gae --model SAGE --epochs 50 --seed $SEED --use_feature
done

# GIN on CiteSeer
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --train_gae --model GIN --epochs 50 --seed $SEED --use_feature
done
