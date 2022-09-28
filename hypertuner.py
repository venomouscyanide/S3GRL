import itertools
import warnings
from timeit import default_timer

import torch
import argparse

from tqdm import tqdm

from seal_link_pred import run_sgrl_learning
from sgrl_run_manager import SGRLArgumentParser

warnings.filterwarnings(action="ignore")

from torch_geometric import seed_everything


class HyperTuningSearchSpace:
    m = [1, 2, 3, 5, 7]  # the length of rw sequences
    M = [2, 5, 10, 20, 40]  # the number of rw sequences
    dropedge = [0.00]


class ManualTuner:
    @staticmethod
    def tune(dataset, model, hidden_channels, use_feature, lr,
             runs, use_heuristic, m, M, cuda_device, dropedge, save_appendix, data_appendix, device, train_percent,
             delete_dataset,
             epochs, split_val_ratio, split_test_ratio, profile, seed, use_valedges_as_input):
        sweal_parser = SGRLArgumentParser(dataset=dataset, fast_split=False, model=model, sortpool_k=0.6, num_layers=3,
                                          hidden_channels=hidden_channels, batch_size=32, num_hops=1,
                                          ratio_per_hop=1.0, max_nodes_per_hop=None, node_label='drnl',
                                          use_feature=use_feature, use_edge_weight=False, lr=lr, epochs=epochs,
                                          runs=runs, train_percent=train_percent, val_percent=100, test_percent=100,
                                          dynamic_train=False,
                                          dynamic_val=False, dynamic_test=False, num_workers=16,
                                          train_node_embedding=False, pretrained_node_embedding=False,
                                          use_valedges_as_input=use_valedges_as_input, eval_steps=1, log_steps=1,
                                          data_appendix=data_appendix, save_appendix=save_appendix, keep_old=False,
                                          continue_from=None,
                                          only_test=False, test_multiple_models=False, use_heuristic=use_heuristic,
                                          m=m, M=M, cuda_device=cuda_device, dropedge=dropedge, calc_ratio=False,
                                          checkpoint_training=False,
                                          delete_dataset=delete_dataset, pairwise=False, loss_fn='', neg_ratio=1,
                                          profile=profile, split_val_ratio=split_val_ratio,
                                          split_test_ratio=split_test_ratio, train_mlp=False, dropout=0.50,
                                          train_gae=False, base_gae="", dataset_stats=False, seed=seed,
                                          dataset_split_num=1, train_n2v=False, train_mf=False, sign_k=3, sign_type='',
                                          pool_operatorwise=False, optimize_sign=False, init_features='')

        run_sgrl_learning(sweal_parser, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A quick way to manually tune the hyperparameters")

    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--hidden_channels', type=int)
    parser.add_argument('--use_feature', action='store_true')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--runs', type=int)
    parser.add_argument('--hyper_runs', type=int)
    parser.add_argument('--use_heuristic', action='store_true')
    parser.add_argument('--delete_dataset', action='store_true',
                        help="delete existing datasets folder before running new command")
    parser.add_argument('--save_appendix', type=str)
    parser.add_argument('--data_appendix', type=str)
    parser.add_argument('--train_percent', type=float)
    parser.add_argument('--cuda_device', type=int, default=0, help="Only set available the passed GPU")
    parser.add_argument('--split_val_ratio', type=float, default=0.05)
    parser.add_argument('--split_test_ratio', type=float, default=0.1)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()
    device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"

    permutations = list(itertools.product(
        HyperTuningSearchSpace.m, HyperTuningSearchSpace.M, HyperTuningSearchSpace.dropedge
    ))

    total_combinations = len(permutations)
    perms = tqdm(permutations, ncols=140)

    for perm in perms:
        for hyper_run in range(args.hyper_runs):
            seed_set = hyper_run + args.seed
            seed_everything(seed_set)  # goes like 1, 2, .. 5
            print(f"Running for hyper_run {hyper_run} m:{perm[0]}, M:{perm[1]}, dropedge:{perm[2]} seed: {seed_set}")

            start = default_timer()
            ManualTuner.tune(dataset=args.dataset, model=args.model, hidden_channels=args.hidden_channels,
                             lr=args.lr, runs=args.runs, use_feature=args.use_feature,
                             use_heuristic=args.use_heuristic, m=perm[0], M=perm[1], cuda_device=0, dropedge=perm[2],
                             save_appendix=args.save_appendix, data_appendix=args.data_appendix, device=device,
                             train_percent=args.train_percent, delete_dataset=args.delete_dataset, epochs=args.epochs,
                             split_val_ratio=args.split_val_ratio, split_test_ratio=args.split_test_ratio,
                             profile=args.profile, seed=seed_set, use_valedges_as_input=args.use_valedges_as_input)
            end = default_timer()

            print(f'Time taken for hyper_run {hyper_run} with m:{perm[0]}, M:{perm[1]}: {end - start:.2f} seconds')
