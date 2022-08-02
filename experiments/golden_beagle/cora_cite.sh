#!/bin/bash
# CORA
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --epochs 50 --seed $SEED --node_label zo --use_feature --hidden_channels 256 --sign_k 3 --sign_type golden --num_hops 3 --data_appendix cora_k3_$SEED --model SIGN
done
rm -rf dataset/
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --epochs 50 --seed $SEED --node_label zo --use_feature --hidden_channels 256 --sign_k 3 --num_hops 1 --sign_type golden --data_appendix cora_k-1_$SEED --model SIGN
done
rm -rf dataset/
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --epochs 50 --seed $SEED --num_hops 3 --use_feature --hidden_channels 256 --data_appendix cora_seal_$SEED
done
rm -rf dataset/

# CITESEER
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --epochs 50 --seed $SEED --node_label zo --use_feature --hidden_channels 256 --sign_k 3 --sign_type golden --num_hops 3 --data_appendix citeseer_k3_$SEED --model SIGN
done
rm -rf dataset/
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --epochs 50 --seed $SEED --node_label zo --use_feature --hidden_channels 256 --sign_k 3 --num_hops 1 --sign_type golden --data_appendix citeseer_k3_$SEED --model SIGN
done
rm -rf dataset/
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --epochs 50 --seed $SEED --num_hops 3 --use_feature --hidden_channels 256 --data_appendix citeseer_seal_$SEED
done
rm -rf dataset/