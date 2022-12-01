import argparse
import os
import sys

import numpy as np
import torch
from torch_geometric import seed_everything

from baselines.baseline_utils import get_data_helper


class DummyArgs:
    def __init__(self, dataset):
        self.data_name = dataset
        self.dataset = dataset
        self.use_valedges_as_input = False
        self.fast_split = False
        self.res_dir = ""
        self.eval_steps = 1
        self.log_steps = 1

        # important hyperparameters
        self.split_val_ratio = 0.05
        self.split_test_ratio = 0.1
        self.neg_ratio = 1
        self.runs = 1
        self.epochs = 50
        self.lr = 0.01
        self.embedding_dim = 32


def run_gic(dataset, runs):
    acc_list = []

    for run in range(1, runs + 1, 1):
        seed_everything(run)
        args = DummyArgs(dataset)
        data, split_edge = get_data_helper(args)
        test_and_val = [split_edge['test']['edge'].T, split_edge['test']['edge_neg'].T, split_edge['valid']['edge'].T,
                        split_edge['valid']['edge_neg'].T]
        edge_index = split_edge['train']['edge'].T

        if type(data.x) != torch.Tensor:
            # if no features, we simply set x to be identity matrix as seen in GAE paper
            data.x = torch.eye(data.num_nodes)
        args.seed = run

        x = data.x
        args.par_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        sys.path.append('%s/Software/GIC/' % args.par_dir)
        from GICEmbs import CalGIC

        acc_list += [
            CalGIC(edge_index, x, args.data_name, test_and_val, args)
        ]

    array = np.array(acc_list)
    print(f'Final Average Test of {runs} runs is: {array.mean():.2f} ± {array.std():.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--runs', type=int, required=True)

    args = parser.parse_args()

    run_gic(args.dataset, args.runs)
