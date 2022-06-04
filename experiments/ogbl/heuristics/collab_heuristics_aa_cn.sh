#!/bin/bash
# this will help reproduce scaled vs baselines results for heuristics; AA, CN and PPR on Cora and CiteSeer
# all runs are on 5 different seeds(1 run each of every seed).

# CN
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset ogbl-collab --use_heuristic CN --seed $SEED
done

# AA
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset ogbl-collab --use_heuristic AA --seed $SEED
done
