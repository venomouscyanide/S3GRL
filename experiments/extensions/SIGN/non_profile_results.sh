#!/bin/bash

# PUBMED
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Pubmed --epochs 50 --seed $SEED --node_label zo --use_feature --num_layers 3 --sign_k 3 --data_appendix pubmed_k3_$SEED --model SIGN --dynamic_train
done
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Pubmed --epochs 50 --seed $SEED --node_label zo --use_feature --num_layers 3 --sign_k -1 --data_appendix pubmed_k-1_$SEED --model SIGN --dynamic_train
done

# attributed-Facebook
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset attributed-Facebook --epochs 50  --seed $SEED --use_feature --hidden_channels 256 --split_test_ratio 0.50 --node_label zo --data_appendix facebook_k3_$SEED --num_layers 3 --sign_k 3 --model SIGN
done
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset attributed-Facebook --epochs 50 --seed $SEED --use_feature --hidden_channels 256 --split_test_ratio 0.50 --node_label zo --data_appendix facebook_k-1_$SEED --num_layers 3 --sign_k -1 --model SIGN
done
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset attributed-Facebook --epochs 50 --seed $SEED --num_hops 1 --use_feature --hidden_channels 256 --split_test_ratio 0.50 --model DGCNN --data_appendix facebook_seal_$SEED
done

# PUBMED
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Pubmed --epochs 50 --seed $SEED --num_hops 3 --use_feature --data_appendix pubmed_seal_$SEED --dynamic_train --model DGCNN
done

# CORA
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --epochs 50 --seed $SEED --node_label zo --use_feature --hidden_channels 256 --num_layers 3 --sign_k 3 --data_appendix cora_k3_$SEED --model SIGN
done
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --epochs 50 --seed $SEED --node_label zo --use_feature --hidden_channels 256 --num_layers 3 --sign_k -1 --data_appendix cora_k-1_$SEED --model SIGN
done
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --epochs 50 --seed $SEED --num_hops 3 --use_feature --hidden_channels 256 --data_appendix cora_seal_$SEED --model DGCNN
done

# CITESEER
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --epochs 50 --seed $SEED --node_label zo --use_feature --hidden_channels 256 --num_layers 3 --sign_k 3 --data_appendix citeseer_k3_$SEED --model SIGN
done
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --epochs 50 --seed $SEED --node_label zo --use_feature --hidden_channels 256 --num_layers 3 --sign_k -1 --data_appendix citeseer_k-1_$SEED --model SIGN
done
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --epochs 50 --seed $SEED --num_hops 3 --use_feature --hidden_channels 256 --data_appendix citeseer_seal_$SEED --model DGCNN
done
