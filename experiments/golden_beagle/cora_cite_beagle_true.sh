#!/bin/bash
# TODO; uncomment the 3, 1 exps. num_hops is 2 as OOM
# CORA
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset Cora --epochs 50 --seed $SEED --node_label zo --use_feature --hidden_channels 256 --sign_k 3 --num_hops 2 --sign_type beagle --data_appendix cora_k1_$SEED --model SIGN --pool_operatorwise
done
rm -rf dataset/
#for SEED in 1 2 3 4 5; do
#  python sgrl_link_pred.py --dataset Cora --epochs 50 --seed $SEED --node_label zo --use_feature --hidden_channels 256 --sign_k 3 --num_hops 1 --sign_type beagle --data_appendix cora_k1_$SEED --model SIGN --pool_operatorwise
#done
#rm -rf dataset/

# CITESEER
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset CiteSeer --epochs 50 --seed $SEED --node_label zo --use_feature --hidden_channels 256 --sign_k 3 --num_hops 2 --sign_type beagle --data_appendix citeseer_k1_$SEED --model SIGN --pool_operatorwise
done
rm -rf dataset/

#for SEED in 1 2 3 4 5; do
#  python sgrl_link_pred.py --dataset CiteSeer --epochs 50 --seed $SEED --node_label zo --use_feature --hidden_channels 256 --sign_k 3 --num_hops 1 --sign_type beagle --data_appendix citeseer_k1_$SEED --model SIGN --pool_operatorwise
#done
#rm -rf dataset/
