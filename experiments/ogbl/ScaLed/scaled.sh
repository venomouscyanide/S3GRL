#!/bin/bash


# ogbl-collab
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset ogbl-collab --use_feature --epochs 50 --m 1 --M 20 --train_percent 15 --hidden_channels 256 --use_valedges_as_input --seed $SEED --profile
done