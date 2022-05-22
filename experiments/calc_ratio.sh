#!/bin/sh
# Cora
python seal_link_pred.py --dataset Cora --use_feature --hidden_channels 256 --model DGCNN --m 5 --M 50 --calc_ratio --num_hops 3
# CiteSeer
python seal_link_pred.py --dataset CiteSeer --hidden_channels 256 --model DGCNN --m 5 --M 50 --calc_ratio --num_hops 3
# Pubmed
python seal_link_pred.py --dataset PubMed --model DGCNN --m 5 --M 50 --use_feature --calc_ratio --num_hops 3
# attributed-Facebook
python seal_link_pred.py --dataset attributed-Facebook --use_feature --hidden_channels 256 --model DGCNN --m 2 --M 5 --split_test_ratio 0.50 --calc_ratio --num_hops 1
# ogbl-collab
python seal_link_pred.py --dataset ogbl-collab --train_percent 15 --hidden_channels 256 --use_valedges_as_input --model DGCNN --m 2 --M 10 --calc_ratio --val_percent 10 --num_hops 1

