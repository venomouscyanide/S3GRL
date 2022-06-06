#!/bin/bash
# reproduce ogbl results for SEAL. We fix hop_num to 1

# ogbl-collab
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset ogbl-collab --epochs 50 --num_hops 1 --train_percent 15 --hidden_channels 256 --use_valedges_as_input --seed $SEED --profile
done