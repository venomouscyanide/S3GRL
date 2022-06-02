#!/bin/bash
# this will help reproduce scaled vs baselines results for heuristics; AA, CN and PPR on Cora and CiteSeer
# all runs are on 5 different seeds(1 run each of every seed).

# Cora
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --use_heuristic CN --seed $SEED
done

# Cora
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --use_heuristic AA --seed $SEED
done

# Cora
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --use_heuristic PPR --seed $SEED
done

# CiteSeer
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --use_heuristic CN --seed $SEED
done

# CiteSeer
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --use_heuristic AA --seed $SEED
done

# CiteSeer
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --use_heuristic PPR --seed $SEED
done
