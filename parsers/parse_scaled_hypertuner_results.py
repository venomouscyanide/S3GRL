import argparse

import numpy as np


def parse_hyper_tuner_results(file_name):
    with open(file_name, 'r') as hyper_file:
        results = hyper_file.readlines()

    max_lines = len(results)
    index = 0

    parsed_mean = {
        "PB": {
            "AUC_MEAN": [],
            "AP_MEAN": [],
            "Time_MEAN": []
        },
        "Ecoli": {
            "AUC_MEAN": [],
            "AP_MEAN": [],
            "Time_MEAN": []
        },
    }

    parsed_results = {
        "PB": {
            "AUC": {
                (2, 2): [],
                (2, 5): [],
                (2, 10): [],
                (2, 20): [],
                (3, 2): [],
                (3, 5): [],
                (3, 10): [],
                (3, 20): [],
                (5, 2): [],
                (5, 5): [],
                (5, 10): [],
                (5, 20): [],
                (7, 2): [],
                (7, 5): [],
                (7, 10): [],
                (7, 20): []
            },
            "AP": {
                (2, 2): [],
                (2, 5): [],
                (2, 10): [],
                (2, 20): [],
                (3, 2): [],
                (3, 5): [],
                (3, 10): [],
                (3, 20): [],
                (5, 2): [],
                (5, 5): [],
                (5, 10): [],
                (5, 20): [],
                (7, 2): [],
                (7, 5): [],
                (7, 10): [],
                (7, 20): []
            },
            "Time": {
                (2, 2): [],
                (2, 5): [],
                (2, 10): [],
                (2, 20): [],
                (3, 2): [],
                (3, 5): [],
                (3, 10): [],
                (3, 20): [],
                (5, 2): [],
                (5, 5): [],
                (5, 10): [],
                (5, 20): [],
                (7, 2): [],
                (7, 5): [],
                (7, 10): [],
                (7, 20): []
            },
        },
        "Ecoli": {

            "AUC": {
                (2, 2): [],
                (2, 5): [],
                (2, 10): [],
                (2, 20): [],
                (3, 2): [],
                (3, 5): [],
                (3, 10): [],
                (3, 20): [],
                (5, 2): [],
                (5, 5): [],
                (5, 10): [],
                (5, 20): [],
                (7, 2): [],
                (7, 5): [],
                (7, 10): [],
                (7, 20): []
            },
            "AP": {
                (2, 2): [],
                (2, 5): [],
                (2, 10): [],
                (2, 20): [],
                (3, 2): [],
                (3, 5): [],
                (3, 10): [],
                (3, 20): [],
                (5, 2): [],
                (5, 5): [],
                (5, 10): [],
                (5, 20): [],
                (7, 2): [],
                (7, 5): [],
                (7, 10): [],
                (7, 20): []
            },
            "Time": {
                (2, 2): [],
                (2, 5): [],
                (2, 10): [],
                (2, 20): [],
                (3, 2): [],
                (3, 5): [],
                (3, 10): [],
                (3, 20): [],
                (5, 2): [],
                (5, 5): [],
                (5, 10): [],
                (5, 20): [],
                (7, 2): [],
                (7, 5): [],
                (7, 10): [],
                (7, 20): []
            },
        }
    }

    while index < max_lines:
        line = results[index]
        if line.startswith("Command line input:"):
            dataset = line.split("--dataset")[-1].split("--hidden_channels")[0].strip()
            while 1:
                index += 1
                line = results[index]
                if line.startswith("All runs:"):
                    index += 4
                    line = results[index]
                    auc_score = float(line.split("Highest Test: ")[-1].split("± nan")[0].strip())

                    index += 8
                    line = results[index]
                    ap_score = float(line.split("Highest Test: ")[-1].split("± nan")[0].strip())

                    index += 6
                    line = results[index]
                    time_taken = float(
                        line.split("Time taken for hyper_run ")[-1].split(" seconds")[0].split(": ")[-1].strip())
                    m = int(line.split("with")[-1].split(', ')[0].split(':')[-1].strip())
                    M = int(line.split("with")[-1].split(", ")[-1].split('M:')[-1].split(":")[0].strip())

                    parsed_results[dataset]['AUC'][(m, M)].append(auc_score)
                    parsed_results[dataset]['AP'][(m, M)].append(ap_score)
                    parsed_results[dataset]['Time'][(m, M)].append(time_taken)

                    index += 1
                    break
        index += 1

    for dataset, dict in parsed_results.items():
        for metric, pairs in dict.items():
            for pair_id, values in pairs.items():
                if not values:
                    continue
                values = np.array(values)
                parsed_mean[dataset][f'{metric}_MEAN'].append(values.mean())

    print("Done reading file")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Read the results from hyperparameter logs to get data to run plot.py"
    )
    parser.add_argument('--file_name', type=str)
    args = parser.parse_args()

    parse_hyper_tuner_results(args.file_name)
