#!/bin/bash
# CORA
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --epochs 50 --seed $SEED --num_hops 3 --use_feature --hidden_channels 256 --data_appendix cora_seal_$SEED
done
rm -rf dataset/

# CiteSeer
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --epochs 50 --seed $SEED --num_hops 3 --use_feature --hidden_channels 256 --data_appendix citeseer_seal_$SEED
done
rm -rf dataset/
