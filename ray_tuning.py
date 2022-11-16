# currently only supports non-attr datasets
import argparse
import json
import os

import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch_geometric import seed_everything

from seal_link_pred import run_sgrl_learning_with_ray
from sgrl_run_manager import SGRLArgumentParser
from tuning_utils import TimeStopper


class HyperParameterTuning:
    MAX_EPOCHS = 50
    CPUS_AVAIL = 1
    GPUS_AVAIL = 0
    NUM_SAMPLES = 1000

    seed = 42

    CONFIG = {
        "hidden_channels": tune.choice([32, 64, 128, 256, 512, 1024]),
        "batch_size": tune.choice([32, 64, 128, 256, 512, 1024, 2048]),
        "num_hops": tune.choice([1, 2, 3]),
        "lr": tune.uniform(lower=0.0001, upper=0.0025),
        "dropout": tune.uniform(lower=0.2, upper=0.8),
        "sign_k": tune.choice([2, 3, 5, 7]),
        "n2v_dim": tune.choice([32, 64, 128, 256, 512, 1024]),
        "k_heuristic": 0
    }

    base_config = {
        "hyperparams_per_run": {
            "dataset": "",  # configurable
            "seed": seed,
            "fast_split": False,
            "delete_dataset": True,
            "model": "SIGN",
            "sortpool_k": -1,
            "num_layers": -1,
            "hidden_channels": 0,  # configurable
            "batch_size": 0,  # configurable
            "num_hops": 0,  # configurable
            "ratio_per_hop": 1.0,
            "max_nodes_per_hop": None,
            "node_label": "zo",
            "use_feature": False,
            "use_edge_weight": False,
            "lr": 0,  # configurable
            "epochs": MAX_EPOCHS,
            "runs": 1,
            "train_percent": 100,
            "val_percent": 100,
            "test_percent": 100,
            "dynamic_train": False,
            "dynamic_val": False,
            "dynamic_test": False,
            "num_workers": 16,
            "train_node_embedding": False,
            "pretrained_node_embedding": None,
            "use_valedges_as_input": False,
            "eval_steps": 1,
            "log_steps": 1,
            "checkpoint_training": False,
            "data_appendix": "",
            "save_appendix": "",
            "keep_old": True,
            "continue_from": None,
            "only_test": False,
            "test_multiple_models": False,
            "use_heuristic": None,
            "m": 0,
            "M": 0,
            "cuda_device": -1,  # configurable
            "dropedge": 0.0,
            "calc_ratio": False,
            "pairwise": False,
            "loss_fn": "",
            "neg_ratio": 1,
            "profile": False,
            "split_val_ratio": 0.05,
            "split_test_ratio": 0.10,
            "train_mlp": False,
            "dropout": 0.50,  # configurable
            "train_gae": False,
            "dataset_split_num": 1,
            "base_gae": "",
            "dataset_stats": False,
            "train_n2v": False,
            "train_mf": False,
            "sign_k": 3,  # configurable
            "sign_type": "SuP",
            "pool_operatorwise": True,
            "optimize_sign": True,
            "init_features": "n2v",
            "n2v_dim": 128,  # configurable
            "k_heuristic": 0,  # configurable
            "k_node_set_strategy": "intersection",
            "k_pool_strategy": "mean",
        }}


def ray_tune_helper(identifier, output_path, dataset, sign_type):
    hyper_class = HyperParameterTuning
    hyper_class.base_config['hyperparams_per_run']['dataset'] = dataset

    if sign_type == "KSuP":
        k_heuristic = tune.choice([2, 4, 6, 8, 10])
    else:
        k_heuristic = 0
    hyper_class.CONFIG['k_heuristic'] = k_heuristic

    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=hyper_class.MAX_EPOCHS,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(metric_columns=["val_loss", "val_accuracy", "training_iteration"])

    base_arg = SGRLArgumentParser(**HyperParameterTuning.base_config['hyperparams_per_run'])

    device = torch.device('cpu')  # fixed to cpu
    print(f"Using device: {device} for running ray tune")

    seed_everything(42)

    result = tune.run(
        tune.with_parameters(run_sgrl_learning_with_ray, hyper_param_class=base_arg, device='cpu'),
        resources_per_trial={"cpu": hyper_class.CPUS_AVAIL, "gpu": hyper_class.GPUS_AVAIL},
        config=hyper_class.CONFIG,
        num_samples=hyper_class.NUM_SAMPLES,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=os.path.join(identifier, output_path),
        log_to_file=True,
        stop=TimeStopper(),
        resume="AUTO"
    )
    best_trial = result.get_best_trial("val_accuracy", "max", "last")

    print("Best trial config: {}".format(best_trial))
    with open(f'{identifier}_best_result.json', "w") as file:
        json.dump(best_trial.config, file)

    print("Best trial final train loss: {}".format(best_trial.last_result["val_loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["val_accuracy"]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--identifier', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--sign_type', type=str, required=True, choices=["SuP", "KSuP"])

    args = parser.parse_args()
    ray_tune_helper(args.identifier, args.output_path, args.dataset, args.sign_type)
