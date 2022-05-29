#!/bin/bash
# seed is fixed to a figure(500) not used for training later on. This is to make sure we are not biased/influenced by a seed later on used for training
# for now, to reproduce the results, just run this bash script
# however, you will have to consume the data from the terminal output manually
# the data extracted manually is placed into plot.py for creating the plots
python hypertuner.py --model DGCNN --dataset USAir --hidden_channels 32 --lr 0.0001 --epochs 5 --runs 1 --save_appendix "" --data_appendix "" --train_percent 100 --seed 500 --profile
python hypertuner.py --model DGCNN --dataset NS --hidden_channels 32 --lr 0.0001 --epochs 5 --runs 1 --save_appendix "" --data_appendix "" --train_percent 100 --seed 500 --profile
python hypertuner.py --model DGCNN --dataset Power --hidden_channels 32 --lr 0.0001 --epochs 5 --runs 1 --save_appendix "" --data_appendix "" --train_percent 100 --seed 500 --profile
python hypertuner.py --model DGCNN --dataset Celegans --hidden_channels 32 --lr 0.0001 --epochs 5 --runs 1 --save_appendix "" --data_appendix "" --train_percent 100 --seed 500 --profile
python hypertuner.py --model DGCNN --dataset Router --hidden_channels 32 --lr 0.0001 --epochs 5 --runs 1 --save_appendix "" --data_appendix "" --train_percent 100 --seed 500 --profile
python hypertuner.py --model DGCNN --dataset PB --hidden_channels 32 --lr 0.0001 --epochs 5 --runs 1 --save_appendix "" --data_appendix "" --train_percent 100 --seed 500 --profile
python hypertuner.py --model DGCNN --dataset Ecoli --hidden_channels 32 --lr 0.0001 --epochs 5 --runs 1 --save_appendix "" --data_appendix "" --train_percent 100 --seed 500 --profile
python hypertuner.py --model DGCNN --dataset Yeast --hidden_channels 32 --lr 0.0001 --epochs 5 --runs 1 --save_appendix "" --data_appendix "" --train_percent 100 --seed 500 --profile
