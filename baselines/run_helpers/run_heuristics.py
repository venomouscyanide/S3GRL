import argparse

import numpy as np
from torch_geometric import seed_everything

from sgrl_link_pred import run_sgrl_learning


class DummyArgs:
    def __init__(self, dataset, heuristic, seed):
        self.use_heuristic = heuristic
        self.save_appendix = ""
        self.data_appendix = ""
        self.dataset = dataset
        self.res_dir = ""
        self.keep_old = True
        self.dataset_stats = False
        self.init_features = False
        self.eval_metric = "auc"
        self.seed = seed
        self.m = -1
        self.M = -1
        self.dropedge = False
        self.use_valedges_as_input = False
        self.fast_split = False
        self.init_representation = None

        # important hyperparameters
        self.split_val_ratio = 0.05
        self.split_test_ratio = 0.1
        self.neg_ratio = 1
        self.runs = 1


def run_heuristics_helper(dataset, heuristic, runs):
    list_of_auc = []

    for run in range(1, runs + 1, 1):
        print(f"Running {heuristic} with seed {run}")
        seed_everything(run)
        args = DummyArgs(dataset, heuristic, run)
        list_of_auc += [run_sgrl_learning(args, device='cpu') * 100]

    array = np.array(list_of_auc)
    print(f'Final Average Test of {runs} runs is: {array.mean():.2f} ± {array.std():.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--heuristic', type=str, required=True, choices=["CN", "AA", "PPR"])
    parser.add_argument('--runs', type=int, required=True)

    args = parser.parse_args()

    run_heuristics_helper(args.dataset, args.heuristic, args.runs)
