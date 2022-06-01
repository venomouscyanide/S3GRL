#!/bin/bash
# run SEAL on Cora and CiteSeer to show baseline SEAL's  performance on attributed datasets
# CORA
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --num_hops 3 --use_feature --profile --seed $SEED
done

# CITESEER
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --num_hops 3 --use_feature --profile --seed $SEED
done
