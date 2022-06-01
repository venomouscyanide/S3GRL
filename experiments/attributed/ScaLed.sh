#!/bin/bash
# run ScaLed on Cora and CiteSeer to show performance on attributed datasets
# CORA
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --m 3 --M 20 --use_feature --profile --seed $SEED
done

# CITESEER
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --m 3 --M 20 --use_feature --profile --seed $SEED
done
