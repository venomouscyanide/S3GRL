#!/bin/bash
# this will help reproduce scaled vs baselines results for heuristics; AA, CN and PPR on all 8 SEAL datasets
# all runs are on 5 different seeds(1 run each of every seed). Different seed ensure different set of initial weight and dataset splits

# USAir
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset USAir --use_heuristic CN --seed $SEED
done

# USAir
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset USAir --use_heuristic AA --seed $SEED
done

# USAir
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset USAir --use_heuristic PPR --seed $SEED
done

# NS
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset NS --use_heuristic CN --seed $SEED
done

# NS
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset NS --use_heuristic AA --seed $SEED
done

# NS
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset NS --use_heuristic PPR --seed $SEED
done

# Power
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Power --use_heuristic CN --seed $SEED
done

# Power
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Power --use_heuristic AA --seed $SEED
done

# Power
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Power --use_heuristic PPR --seed $SEED
done

# Celegans
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Celegans --use_heuristic CN --seed $SEED
done

# Celegans
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Celegans --use_heuristic AA --seed $SEED
done

# Celegans
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Celegans --use_heuristic PPR --seed $SEED
done

# Router
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Router --use_heuristic CN --seed $SEED
done

# Router
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Router --use_heuristic AA --seed $SEED
done

# Router
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Router --use_heuristic PPR --seed $SEED
done

# PB
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset PB --use_heuristic CN --seed $SEED
done

# PB
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset PB --use_heuristic AA --seed $SEED
done

# PB
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset PB --use_heuristic PPR --seed $SEED
done

# Ecoli
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Ecoli --use_heuristic CN --seed $SEED
done

# Ecoli
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Ecoli --use_heuristic AA --seed $SEED
done

# Ecoli
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Ecoli --use_heuristic PPR --seed $SEED
done

# Yeast
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Yeast --use_heuristic CN --seed $SEED
done

# Yeast
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Yeast --use_heuristic AA --seed $SEED
done

# Yeast
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Yeast --use_heuristic PPR --seed $SEED
done
