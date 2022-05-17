#!/bin/sh
# Cora
cd SEAL_OGB/
python seal_link_pred.py --dataset Cora --num_hops 3 --use_feature --hidden_channels 256 --runs 5 --model GCN --profile
python seal_link_pred.py --dataset Cora --num_hops 3 --use_feature --hidden_channels 256 --runs 5 --model SAGE --profile
cd ../SWEAL_OGB/
python seal_link_pred.py --dataset Cora --use_feature --hidden_channels 256 --runs 5 --model GCN --m 5 --M 50 --profile
python seal_link_pred.py --dataset Cora --use_feature --hidden_channels 256 --runs 5 --model SAGE --m 5 --M 50 --profile
# CiteSeer
cd ../SEAL_OGB/
python seal_link_pred.py --dataset CiteSeer --num_hops 3 --hidden_channels 256 --runs 5 --model GCN --profile
python seal_link_pred.py --dataset CiteSeer --num_hops 3 --hidden_channels 256 --runs 5 --model SAGE --profile
cd ../SWEAL_OGB/
python seal_link_pred.py --dataset CiteSeer --hidden_channels 256 --runs 5 --model GCN --m 5 --M 50 --profile
python seal_link_pred.py --dataset CiteSeer --hidden_channels 256 --runs 5 --model SAGE --m 5 --M 50 --profile
# Pubmed
cd ../SEAL_OGB/
python seal_link_pred.py --dataset PubMed --num_hops 3 --use_feature --dynamic_train --runs 5 --model GCN --profile
python seal_link_pred.py --dataset PubMed --num_hops 3 --use_feature --dynamic_train --runs 5 --model SAGE --profile
cd ../SWEAL_OGB/
python seal_link_pred.py --dataset PubMed --runs 5 --model GCN --m 5 --M 50 --use_feature --profile --dynamic_train
python seal_link_pred.py --dataset PubMed --runs 5 --model SAGE --m 5 --M 50 --use_feature --profile --dynamic_train
# Attributed-Facebook
cd ../SEAL_OGB/
python seal_link_pred.py --dataset attributed-Facebook --num_hops 1 --use_feature --hidden_channels 256 --split_test_ratio 0.50 --model GCN --runs 5 --profile
python seal_link_pred.py --dataset attributed-Facebook --num_hops 1 --use_feature --hidden_channels 256 --split_test_ratio 0.50 --model SAGE --runs 5 --profile
cd ../SWEAL_OGB/
python seal_link_pred.py --dataset attributed-Facebook --use_feature --hidden_channels 256 --runs 5 --model GCN --m 2 --M 50 --split_test_ratio 0.50 --profile
python seal_link_pred.py --dataset attributed-Facebook --use_feature --hidden_channels 256 --runs 5 --model SAGE --m 2 --M 50 --split_test_ratio 0.50 --profile
# ogbl-collab
cd ../SEAL_OGB/
python seal_link_pred.py --dataset ogbl-collab --num_hops 1 --train_percent 15 --hidden_channels 256 --use_valedges_as_input --runs 5 --model GCN --profile --val_percent 10
python seal_link_pred.py --dataset ogbl-collab --num_hops 1 --train_percent 15 --hidden_channels 256 --use_valedges_as_input --runs 5 --model SAGE --profile --val_percent 10
cd ../SWEAL_OGB/
python seal_link_pred.py --dataset ogbl-collab --train_percent 15 --hidden_channels 256 --use_valedges_as_input --runs 5 --model GCN --m 5 --M 20 --profile --val_percent 10
python seal_link_pred.py --dataset ogbl-collab --train_percent 15 --hidden_channels 256 --use_valedges_as_input --runs 5 --model SAGE --m 5 --M 20 --profile --val_percent 10