#!/bin/bash
# seed is fixed to [1,2,3,4,5] which is same as training seed
# for now, to reproduce the results, just run this bash script, copy the terminal output log and parse it using parser
# the parsed data is placed inside plot.py to create the graphs
python hypertuner.py --model DGCNN --dataset PB --hidden_channels 32 --lr 0.0001 --epochs 5 --runs 1 --hyper_runs 5 --save_appendix "" --data_appendix "" --train_percent 100 --seed 1
python hypertuner.py --model DGCNN --dataset Ecoli --hidden_channels 32 --lr 0.0001 --epochs 5 --runs 1 --hyper_runs 5 --save_appendix "" --data_appendix "" --train_percent 100 --seed 1
