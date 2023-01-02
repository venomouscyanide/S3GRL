import argparse
import json
from timeit import default_timer

import torch
import numpy as np
from torch_geometric import seed_everything

from sgrl_link_pred import run_sgrl_learning, run_sgrl_with_run_profiling


class SGRLArgumentParser:
    def __init__(self, dataset, fast_split, model, sortpool_k, num_layers, hidden_channels, batch_size, num_hops,
                 ratio_per_hop, max_nodes_per_hop, node_label, use_feature, use_edge_weight, lr, epochs, runs,
                 train_percent, val_percent, test_percent, dynamic_train, dynamic_val, dynamic_test, num_workers,
                 train_node_embedding, pretrained_node_embedding, use_valedges_as_input, eval_steps, log_steps,
                 data_appendix, save_appendix, keep_old, continue_from, only_test, test_multiple_models, use_heuristic,
                 m, M, cuda_device, dropedge, calc_ratio, checkpoint_training, delete_dataset, pairwise, loss_fn,
                 neg_ratio,
                 profile, split_val_ratio, split_test_ratio, train_mlp, dropout, train_gae, base_gae, dataset_stats,
                 seed, dataset_split_num, train_n2v, train_mf, sign_k, sign_type, pool_operatorwise, optimize_sign,
                 init_features, n2v_dim=256, k_heuristic=0, k_node_set_strategy="", k_pool_strategy="",
                 init_representation=""):
        # Data Settings
        self.dataset = dataset
        self.fast_split = fast_split
        self.delete_dataset = delete_dataset

        # GNN Settings
        self.model = model
        self.sortpool_k = sortpool_k
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.batch_size = batch_size

        # Subgraph extraction settings
        self.num_hops = num_hops
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.node_label = node_label
        self.use_feature = use_feature
        self.use_edge_weight = use_edge_weight

        # Training settings
        self.lr = lr
        self.epochs = epochs
        self.runs = runs
        self.train_percent = train_percent
        self.val_percent = val_percent
        self.test_percent = test_percent
        self.dynamic_train = dynamic_train
        self.dynamic_val = dynamic_val
        self.dynamic_test = dynamic_test
        self.num_workers = num_workers
        self.train_node_embedding = train_node_embedding
        self.pretrained_node_embedding = pretrained_node_embedding

        # Testing settings
        self.use_valedges_as_input = use_valedges_as_input
        self.eval_steps = eval_steps
        self.log_steps = log_steps
        self.checkpoint_training = checkpoint_training
        self.data_appendix = data_appendix
        self.save_appendix = save_appendix
        self.keep_old = keep_old
        self.continue_from = continue_from
        self.only_test = only_test
        self.test_multiple_models = test_multiple_models
        self.use_heuristic = use_heuristic

        # ScaLed settings
        self.m = m
        self.M = M
        self.cuda_device = cuda_device
        self.dropedge = dropedge
        self.calc_ratio = calc_ratio
        self.pairwise = pairwise
        self.loss_fn = loss_fn
        self.neg_ratio = neg_ratio
        self.profile = profile
        self.split_val_ratio = split_val_ratio
        self.split_test_ratio = split_test_ratio
        self.train_mlp = train_mlp
        self.dropout = dropout
        self.train_gae = train_gae
        self.base_gae = base_gae
        self.dataset_stats = dataset_stats
        self.seed = seed
        self.dataset_split_num = dataset_split_num
        self.train_n2v = train_n2v
        self.train_mf = train_mf

        # SIGN related
        self.sign_k = sign_k
        self.sign_type = sign_type
        self.pool_operatorwise = pool_operatorwise
        self.optimize_sign = optimize_sign
        self.init_features = init_features
        self.n2v_dim = n2v_dim
        self.k_heuristic = k_heuristic
        self.k_node_set_strategy = k_node_set_strategy
        self.k_pool_strategy = k_pool_strategy
        self.init_representation = init_representation


def sgrl_master_controller(config, results_json):
    """
    Wrapper to run SGRL methods to capture the results in a cleaner fashion
    """

    exp_results = {}

    with open(config) as config_file:
        config = json.load(config_file)

    for identifier, ds_config in config['datasets'].items():
        ds_params = ds_config['ds_params']
        runs = ds_params['runs']
        seeds = ds_params['seeds']
        dataset = ds_params['dataset']

        kwargs = ds_config['hyperparams_per_run']

        best_test_scores = []
        prep_times = []
        total_run_times = []

        for run, seed in zip(range(1, runs + 1), seeds):
            kwargs.update(
                {
                    "dataset": dataset,
                    "seed": seed,
                }
            )

            device = torch.device(f'cuda:{kwargs["cuda_device"]}' if torch.cuda.is_available() else 'cpu')
            print(f"Run {run} of {dataset} with id {identifier} using device {device}")

            args = SGRLArgumentParser(**kwargs)
            seed_everything(args.seed)

            start = default_timer()
            if args.profile:
                total_prep_time, best_test_score, total_run_time = run_sgrl_with_run_profiling(args, device)
            else:
                total_prep_time, best_test_score = run_sgrl_learning(args, device)
                end = default_timer()
                total_run_time = end - start

            prep_times.append(total_prep_time)
            total_run_times.append(total_run_time)
            best_test_scores.append(best_test_score)

        prep_times = np.array(prep_times)
        total_run_times = np.array(total_run_times)
        best_test_scores = np.array(best_test_scores)

        exp_results[identifier] = {
            "results": {
                "Average Dataset Prep Time": f"{prep_times.mean():.2f} ± {prep_times.std():.2f}",
                "Average Runtime": f"{total_run_times.mean():.2f} ± {total_run_times.std():.2f}",
                "Average Test AUC": f"{best_test_scores.mean():.2f} ± {best_test_scores.std():.2f}",
            },
            "config_dump": ds_config
        }
        with open(results_json, 'w') as output_file:
            json.dump(exp_results, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cora_cite_comparison_5_runs.json')
    parser.add_argument('--results_json', type=str, default='result.json')

    args = parser.parse_args()

    config = args.config
    results_json = args.results_json
    sgrl_master_controller(config, results_json)
