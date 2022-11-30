import argparse

import numpy as np
import torch
from torch_geometric import seed_everything

from baselines.baseline_utils import get_data_helper
from baselines.mf import train_mf


class DummyArgs:
    def __init__(self, dataset):
        self.dataset = dataset
        self.use_valedges_as_input = False
        self.fast_split = False
        self.res_dir = ""

        # important hyperparameters
        self.split_val_ratio = 0.05
        self.split_test_ratio = 0.1
        self.neg_ratio = 1
        self.runs = 1
        self.epochs = 50


def run_MF(dataset, runs):
    # MLP with one hidden layer, 50 epochs, 32 hidden channels, 50 epochs, batch size 32, 0.01 lr
    acc_list = []
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

    for run in range(1, runs + 1, 1):
        seed_everything(run)
        args = DummyArgs(dataset)
        data, split_edge = get_data_helper(args)
        acc_list += [
            train_mf(data=data, split_edge=split_edge, device=device, log_steps=1, num_layers=3, hidden_channels=32,
                     dropout=0.5, batch_size=32, lr=0.01, epochs=50,
                     eval_steps=1, runs=1, seed=run, args=args)
        ]

    array = np.array(acc_list)
    print(f'Final Average Test of {runs} runs is: {array.mean():.2f} ± {array.std():.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--runs', type=int, required=True)

    args = parser.parse_args()

    run_MF(args.dataset, args.runs)
