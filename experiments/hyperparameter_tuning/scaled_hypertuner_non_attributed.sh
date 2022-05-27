#!/bin/bash
# seed is fixed to a seed not used for training later on
python hypertuner.py --model DGCNN --dataset USAir --hidden_channels 32 --lr 0.0001 --epochs 5 --runs 1 --save_appendix "" --data_appendix "" --train_percent 100 --seed 500 --profile
python hypertuner.py --model DGCNN --dataset NS --hidden_channels 32 --lr 0.0001 --epochs 5 --runs 1 --save_appendix "" --data_appendix "" --train_percent 100 --seed 500 --profile
python hypertuner.py --model DGCNN --dataset Power --hidden_channels 32 --lr 0.0001 --epochs 5 --runs 1 --save_appendix "" --data_appendix "" --train_percent 100 --seed 500 --profile
python hypertuner.py --model DGCNN --dataset Celegans --hidden_channels 32 --lr 0.0001 --epochs 5 --runs 1 --save_appendix "" --data_appendix "" --train_percent 100 --seed 500 --profile
python hypertuner.py --model DGCNN --dataset Router --hidden_channels 32 --lr 0.0001 --epochs 5 --runs 1 --save_appendix "" --data_appendix "" --train_percent 100 --seed 500 --profile
python hypertuner.py --model DGCNN --dataset PB --hidden_channels 32 --lr 0.0001 --epochs 5 --runs 1 --save_appendix "" --data_appendix "" --train_percent 100 --seed 500 --profile
python hypertuner.py --model DGCNN --dataset Ecoli --hidden_channels 32 --lr 0.0001 --epochs 5 --runs 1 --save_appendix "" --data_appendix "" --train_percent 100 --seed 500 --profile
python hypertuner.py --model DGCNN --dataset Yeast --hidden_channels 32 --lr 0.0001 --epochs 5 --runs 1 --save_appendix "" --data_appendix "" --train_percent 100 --seed 500 --profile