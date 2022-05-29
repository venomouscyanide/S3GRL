#!/bin/bash
# run ScaLed on Cora and CiteSeer to show performance on attributed datasets
# CORA
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --m 5 --M 50 --use_feature --profile --seed $SEED
done

# CITESEER
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --m 5 --M 50 --use_feature --profile --seed $SEED
done
