#!/bin/bash
# All ScaLed + SIGN runs without profiling

# PUBMED
# h,k = 1, 10; SIGN 3
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Pubmed --epochs 50 --seed $SEED --node_label zo --use_feature --num_layers 3 --sign_k 3 --data_appendix pubmed_k3_1_10_$SEED --model SIGN --m 1 --M 10
done
# h,k = 1, 20; SIGN 3
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Pubmed --epochs 50 --seed $SEED --node_label zo --use_feature --num_layers 3 --sign_k 3 --data_appendix pubmed_k3_1_20_$SEED --model SIGN --m 1 --M 20
done
rm -rf dataset

# h,k = 1, 10; SIGN -1
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Pubmed --epochs 50 --seed $SEED --node_label zo --use_feature --num_layers 3 --sign_k -1 --data_appendix pubmed_k-1_1_10_$SEED --model SIGN --m 1 --M 10
done
# h,k = 1, 20; SIGN -1
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Pubmed --epochs 50 --seed $SEED --node_label zo --use_feature --num_layers 3 --sign_k -1 --data_appendix pubmed_k-1_1_20_$SEED --model SIGN --m 1 --M 20
done
rm -rf dataset

# attributed-Facebook
# h,k = 1, 10; SIGN 1
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset attributed-Facebook --epochs 50 --seed $SEED --use_feature --hidden_channels 256 --split_test_ratio 0.50 --node_label zo --data_appendix facebook_k3_1_10_$SEED --num_layers 1 --sign_k 1 --model SIGN --m 1 --M 10
done
# h,k = 1, 20; SIGN 1
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset attributed-Facebook --epochs 50 --seed $SEED --use_feature --hidden_channels 256 --split_test_ratio 0.50 --node_label zo --data_appendix facebook_k3_1_20_$SEED --num_layers 1 --sign_k 1 --model SIGN --m 1 --M 20
done
rm -rf dataset

# h,k = 1, 10; SIGN -1
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset attributed-Facebook --epochs 50 --seed $SEED --use_feature --hidden_channels 256 --split_test_ratio 0.50 --node_label zo --data_appendix facebook_k-1_1_10_$SEED --num_layers 3 --sign_k -1 --model SIGN --m 1 --M 10
done
# h,k = 1, 20; SIGN -1
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset attributed-Facebook --epochs 50 --seed $SEED --use_feature --hidden_channels 256 --split_test_ratio 0.50 --node_label zo --data_appendix facebook_k-1_1_20_$SEED --num_layers 3 --sign_k -1 --model SIGN --m 1 --M 20
done
rm -rf dataset

# CORA
# h,k = 1, 10; SIGN 3
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --epochs 50 --seed $SEED --node_label zo --use_feature --hidden_channels 256 --num_layers 3 --sign_k 3 --data_appendix cora_k3_1_10_$SEED --model SIGN --m 1 --M 10
done
# h,k = 1, 20; SIGN 3
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --epochs 50 --seed $SEED --node_label zo --use_feature --hidden_channels 256 --num_layers 3 --sign_k 3 --data_appendix cora_k3_1_20_$SEED --model SIGN --m 1 --M 20
done
rm -rf dataset

# h,k = 1, 10; SIGN -1
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --epochs 50 --seed $SEED --node_label zo --use_feature --hidden_channels 256 --num_layers 3 --sign_k -1 --data_appendix cora_k-1_1_10_$SEED --model SIGN --m 1 --M 10
done
# h,k = 1, 20; SIGN -1
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --epochs 50 --seed $SEED --node_label zo --use_feature --hidden_channels 256 --num_layers 3 --sign_k -1 --data_appendix cora_k-1_1_20_$SEED --model SIGN --m 1 --M 20
done
rm -rf dataset

# CITESEER
# h,k = 1, 10; SIGN 3
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --epochs 50 --seed $SEED --node_label zo --use_feature --hidden_channels 256 --num_layers 3 --sign_k 3 --data_appendix citeseer_k3_1_10_$SEED --model SIGN --m 1 --M 10
done
# h,k = 1, 20; SIGN 3
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --epochs 50 --seed $SEED --node_label zo --use_feature --hidden_channels 256 --num_layers 3 --sign_k 3 --data_appendix citeseer_k3_1_20_$SEED --model SIGN --m 1 --M 20
done
rm -rf dataset

# h,k = 1, 10; SIGN -1
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --epochs 50 --seed $SEED --node_label zo --use_feature --hidden_channels 256 --num_layers 3 --sign_k -1 --data_appendix citeseer_k-1_1_10_$SEED --model SIGN --m 1 --M 10
done
# h,k = 1, 20; SIGN -1
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --epochs 50 --seed $SEED --node_label zo --use_feature --hidden_channels 256 --num_layers 3 --sign_k -1 --data_appendix citeseer_k-1_1_20_$SEED --model SIGN --m 1 --M 20
done
rm -rf dataset
