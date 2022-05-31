#!/bin/bash
# calculate the node/edge numbers and ratio for 5 seeds across all non-attributed datasets

# USAir
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset USAir --num_hops 2 --m 2 --M 20 --seed $SEED --calc_ratio
done

# NS
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset NS --num_hops 2 --m 2 --M 20 --seed $SEED --calc_ratio
done

# Power
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Power --num_hops 2 --m 2 --M 20 --seed $SEED --calc_ratio
done

# Celegans
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Celegans --num_hops 2 --m 2 --M 20 --seed $SEED --calc_ratio
done

# Router
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Router --num_hops 2 --m 2 --M 20 --seed $SEED --calc_ratio
done

# PB
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset PB --num_hops 2 --m 2 --M 20 --seed $SEED --calc_ratio
done

# Ecoli
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Ecoli --num_hops 2 --m 2 --M 20 --seed $SEED --calc_ratio
done

# Yeast
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Yeast --num_hops 2 --m 2 --M 20 --seed $SEED --calc_ratio
done
