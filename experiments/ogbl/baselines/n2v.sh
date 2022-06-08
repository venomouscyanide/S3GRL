#!/bin/bash
# run n2v to get results on ogbl-collab

# ogbl-collab
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset ogbl-collab --epochs 50 --train_n2v --hidden_channels 256 --lr 0.01 --seed $SEED --batch_size 32
done
