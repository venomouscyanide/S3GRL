#!/bin/bash
# this will help reproduce scaled vs baselines results for GAE models(with base GCN, SAGE and GIN) on all 8 SEAL datasets
# all runs are on 5 different seeds(1 run each of every seed). Different seed ensure different set of initial weight and dataset splits

# GCN on USAir
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset USAir --train_gae --model GCN --epochs 50 --seed $SEED
done

# SAGE on USAir
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset USAir --train_gae --model SAGE --epochs 50 --seed $SEED
done

# GIN on USAir
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset USAir --train_gae --model GIN --epochs 50 --seed $SEED
done

# GCN on NS
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset NS --train_gae --model GCN --epochs 50 --seed $SEED
done

# SAGE on NS
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset NS --train_gae --model SAGE --epochs 50 --seed $SEED
done

# GIN on NS
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset NS --train_gae --model GIN --epochs 50 --seed $SEED
done

# GCN on Power
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Power --train_gae --model GCN --epochs 50 --seed $SEED
done

# SAGE on Power
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Power --train_gae --model SAGE --epochs 50 --seed $SEED
done

# GIN on Power
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Power --train_gae --model GIN --epochs 50 --seed $SEED
done

# GCN on Celegans
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Celegans --train_gae --model GCN --epochs 50 --seed $SEED
done

# SAGE on Celegans
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Celegans --train_gae --model SAGE --epochs 50 --seed $SEED
done

# GIN on Celegans
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Celegans --train_gae --model GIN --epochs 50 --seed $SEED
done

# GCN on Router
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Router --train_gae --model GCN --epochs 50 --seed $SEED
done

# SAGE on Router
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Router --train_gae --model SAGE --epochs 50 --seed $SEED
done

# GIN on Router
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Router --train_gae --model GIN --epochs 50 --seed $SEED
done

# GCN on PB
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset PB --train_gae --model GCN --epochs 50 --seed $SEED
done

# SAGE on PB
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset PB --train_gae --model SAGE --epochs 50 --seed $SEED
done

# GIN on PB
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset PB --train_gae --model GIN --epochs 50 --seed $SEED
done

# GCN on Ecoli
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Ecoli --train_gae --model GCN --epochs 50 --seed $SEED
done

# SAGE on Ecoli
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Ecoli --train_gae --model SAGE --epochs 50 --seed $SEED
done

# GIN on Ecoli
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Ecoli --train_gae --model GIN --epochs 50 --seed $SEED
done

# GCN on Yeast
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Yeast --train_gae --model GCN --epochs 50 --seed $SEED
done

# SAGE on Yeast
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Yeast --train_gae --model SAGE --epochs 50 --seed $SEED
done

# GIN on Yeast
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Yeast --train_gae --model GIN --epochs 50 --seed $SEED
done
