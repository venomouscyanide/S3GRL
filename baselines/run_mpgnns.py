import argparse

from baselines.gnn_link_pred import train_gnn


class DummyArgs:
    def __init__(self, runs, model, dataset):
        self.res_dir = ""
        self.use_feature = True

        self.model = model
        self.hidden_channels = 32
        self.lr = 0.01
        self.epochs = 50
        self.eval_steps = 1
        self.log_steps = 1
        self.dropout = 0.5
        self.fast_split = False

        # important hyperparameters
        self.split_val_ratio = 0.05
        self.split_test_ratio = 0.1
        self.neg_ratio = 1
        self.runs = runs
        self.dataset = dataset


def run_MPGNNs(dataset, model, runs):
    args = DummyArgs(runs, model, dataset)
    train_gnn(device='cpu', args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--model', type=str, required=True, choices=["GCN", "SAGE", "GIN"])

    args = parser.parse_args()

    run_MPGNNs(args.dataset, args.model, args.runs)
