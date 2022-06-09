#!/bin/bash
# this will help reproduce scaled vs baselines results for GAE models(with base GCN, SAGE and GIN) for ogbl-collab
# all runs are on 5 different seeds(1 run each of every seed). Different seed ensure different set of initial weight and dataset splits

# GCN on ogbl-collab
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset ogbl-collab --train_gae --model GCN --epochs 400 --seed $SEED --use_feature --lr 0.001 --batch_size 65536 --hidden_channels 256 --use_valedges_as_input --dropout 0.0
done

# SAGE on ogbl-collab
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset ogbl-collab --train_gae --model SAGE --epochs 400 --seed $SEED --use_feature --lr 0.001 --batch_size 65536 --hidden_channels 256 --use_valedges_as_input --dropout 0.0
done

# GIN on ogbl-collab
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset ogbl-collab --train_gae --model GIN --epochs 400 --seed $SEED --use_feature --lr 0.001 --batch_size 65536 --hidden_channels 256 --use_valedges_as_input --dropout 0.0
done
