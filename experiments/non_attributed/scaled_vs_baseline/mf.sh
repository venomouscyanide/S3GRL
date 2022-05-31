#!/bin/bash
# run matrix factorization to get results vs ScaLed

# USAir
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset USAir --epochs 50 --train_mf --hidden_channels 32 --lr 0.01 --seed $SEED
done

# NS
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset NS --epochs 50 --train_mf --hidden_channels 32 --lr 0.01 --seed $SEED
done

# Power
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Power --epochs 50 --train_mf --hidden_channels 32 --lr 0.01 --seed $SEED
done

# Celegans
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Celegans --epochs 50 --train_mf --hidden_channels 32 --lr 0.01 --seed $SEED
done

# Router
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Router --epochs 50 --train_mf --hidden_channels 32 --lr 0.01 --seed $SEED
done

# PB
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset PB --epochs 50 --train_mf --hidden_channels 32 --lr 0.01 --seed $SEED
done

# Ecoli
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Ecoli --epochs 50 --train_mf --hidden_channels 32 --lr 0.01 --seed $SEED
done

# Yeast
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Yeast --epochs 50 --train_mf --hidden_channels 32 --lr 0.01 --seed $SEED
done
