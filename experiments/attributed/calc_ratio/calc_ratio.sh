#!/bin/bash
# calculate the node/edge numbers and ratio for 5 seeds across all attributed datasets

# Cora
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --num_hops 3 --m 3 --M 20 --use_feature --calc_ratio --seed $SEED
done

# CiteSeer
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --num_hops 3 --m 3 --M 20 --use_feature --calc_ratio --seed $SEED
done
