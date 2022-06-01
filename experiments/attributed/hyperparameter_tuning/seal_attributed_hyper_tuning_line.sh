#!/bin/bash
# constant line for SEAL for hypertuning results

# Cora
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --epochs 5 --num_hops 3 --use_feature --seed $SEED
done

# CiteSeer
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --epochs 5 --num_hops 3 --use_feature --seed $SEED
done
