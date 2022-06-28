#!/bin/bash
# CORA and CiteSeer with DRNL instead of ZO

# CORA
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --epochs 50 --seed $SEED --node_label drnl --use_feature --hidden_channels 256 --num_layers 3 --sign_k 3 --data_appendix cora_k3_$SEED --model SIGN
done
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --epochs 50 --seed $SEED --node_label drnl --use_feature --hidden_channels 256 --num_layers 3 --sign_k -1 --data_appendix cora_k-1_$SEED --model SIGN
done

# CITESEER
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --epochs 50 --seed $SEED --node_label drnl --use_feature --hidden_channels 256 --num_layers 3 --sign_k 3 --data_appendix citeseer_k3_$SEED --model SIGN
done
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --epochs 50 --seed $SEED --node_label drnl --use_feature --hidden_channels 256 --num_layers 3 --sign_k -1 --data_appendix citeseer_k-1_$SEED --model SIGN
done
