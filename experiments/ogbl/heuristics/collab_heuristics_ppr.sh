#!/bin/bash
# this will help reproduce scaled vs baselines results for heuristics; PPR on ogbl-collab
# all runs are on 5 different seeds(1 run each of every seed).
# this takes a long time to finish

# PPR
for SEED in 1 2; do
  python seal_link_pred.py --dataset ogbl-collab --use_heuristic PPR --seed $SEED
done
