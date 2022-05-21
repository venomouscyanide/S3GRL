#!/bin/sh
# ogbl-collab
cd SEAL_OGB/
python seal_link_pred.py --dataset ogbl-collab --num_hops 1 --train_percent 15 --hidden_channels 256 --use_valedges_as_input --runs 5 --profile --val_percent 10
cd ../SWEAL_OGB/
python seal_link_pred.py --dataset ogbl-collab --train_percent 15 --hidden_channels 256 --use_valedges_as_input --runs 5 --model DGCNN --m 2 --M 10 --profile --val_percent 10
