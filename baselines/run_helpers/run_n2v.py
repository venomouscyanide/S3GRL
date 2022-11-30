import argparse

import numpy as np
from torch_geometric import seed_everything

from baselines.baseline_utils import get_data_helper
from baselines.n2v import run_n2v


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


def run_n2v_helper(dataset, runs):
    acc_list = []

    for run in range(1, runs + 1, 1):
        seed_everything(run)
        args = DummyArgs(dataset)
        data, split_edge = get_data_helper(args)
        acc_list += [
            run_n2v(device='cpu', data=data, split_edge=split_edge, epochs=50, lr=0.01, hidden_channels=32, neg_ratio=1,
                    batch_size=32, num_threads=8, args=args, seed=run)
        ]

    array = np.array(acc_list)
    print(f'Final Average Test of {runs} runs is: {array.mean():.2f} ± {array.std():.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--runs', type=int, required=True)

    args = parser.parse_args()

    run_n2v_helper(args.dataset, args.runs)
