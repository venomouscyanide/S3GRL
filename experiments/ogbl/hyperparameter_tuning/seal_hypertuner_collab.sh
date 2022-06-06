#!/bin/bash

for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset ogbl-collab --num_hops 1 --train_percent 15 --hidden_channels 256 --use_valedges_as_input --epochs 5 --seed $SEED
done