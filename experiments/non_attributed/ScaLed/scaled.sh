#!/bin/bash
# use the results from hypertuning to set the m and M for all ScaLed runs keeping in mind the time taken.
# We also verify the edge and node ratios before setting the m and M values to ensure sparsity vs SEAL

# USAir
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset USAir --epochs 50 --m 2 --M 20 --seed $SEED --profile
done

# NS
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset NS --epochs 50 --m 2 --M 20 --seed $SEED --profile
done

# Power
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Power --epochs 50 --m 2 --M 20 --seed $SEED --profile
done

# Celegans
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Celegans --epochs 50 --m 2 --M 20 --seed $SEED --profile
done

# Router
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Router --epochs 50 --m 2 --M 20 --seed $SEED --profile
done

# PB
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset PB --epochs 50 --m 2 --M 20 --seed $SEED --profile
done

# Ecoli
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Ecoli --epochs 50 --m 2 --M 20 --seed $SEED --profile
done

# Yeast
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Yeast --epochs 50 --m 2 --M 20 --seed $SEED --profile
done
