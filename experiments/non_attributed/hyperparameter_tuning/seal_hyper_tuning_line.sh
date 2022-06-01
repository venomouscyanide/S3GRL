#!/bin/bash
# constant line for SEAL for hypertuning results

# PB
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset PB --epochs 5 --num_hops 2 --seed $SEED
done

# Ecoli
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Ecoli --epochs 5 --num_hops 2 --seed $SEED
done
