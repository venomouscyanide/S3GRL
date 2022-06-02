#!/bin/bash
# run matrix factorization to get results vs ScaLed

# Cora
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --epochs 50 --train_mf --hidden_channels 32 --lr 0.01 --seed $SEED
done

# CiteSeer
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --epochs 50 --train_mf --hidden_channels 32 --lr 0.01 --seed $SEED
done
