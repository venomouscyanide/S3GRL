#!/bin/bash

python hypertuner.py --model DGCNN --dataset ogbl-collab --train_percent 15 --hidden_channels 256 --use_valedges_as_input --lr 0.0001 --epochs 5 --runs 1 --hyper_runs 5 --save_appendix "" --data_appendix "" --train_percent 100 --seed 1
