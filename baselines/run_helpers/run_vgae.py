import argparse

import numpy as np
from torch_geometric import seed_everything

from baselines.baseline_utils import get_data_helper
from baselines.vgae import run_vgae
import torch


class DummyArgs:
    def __init__(self, dataset):
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
        self.embedding_dim = 32
        self.lr = 0.01
        self.hidden_channels = 64


def run_gae_helper(dataset, runs, model):
    # 64 -> 32 , 0.01 lr
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

        x = data.x
        auc, _ = run_vgae(edge_index=edge_index, x=x, test_and_val=test_and_val, model=model, args=args)
        acc_list += [auc]

    array = np.array(acc_list)
    print(f'Final Average Test of {runs} runs is: {array.mean():.2f} ± {array.std():.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--model', type=str, required=True, choices=['GAE', 'VGAE', 'ARGVA'])

    args = parser.parse_args()

    run_gae_helper(args.dataset, args.runs, args.model)
