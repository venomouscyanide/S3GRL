#!/bin/bash
# this will help reproduce scaled vs baselines results for heuristics; AA, CN, PPR on ogbl-collab
# all runs are on 5 different seeds(1 run each of every seed).

# CN
for SEED in 1; do
  python seal_link_pred.py --dataset ogbl-collab --use_heuristic CN --seed $SEED --use_valedges_as_input
done

# AA
for SEED in 1; do
  python seal_link_pred.py --dataset ogbl-collab --use_heuristic AA --seed $SEED --use_valedges_as_input
done

# PPR
for SEED in 1; do
  python seal_link_pred.py --dataset ogbl-collab --use_heuristic PPR --seed $SEED --use_valedges_as_input
done
