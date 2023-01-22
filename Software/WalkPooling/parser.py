import argparse
import json

import numpy as np

dataset_dict = \
    {"AUC": {},
     "AP": {},
     "AUC Mean": {},
     "AP Mean": {}
     }


def parse_results(file_name, num_datasets):
    with open(file_name, 'r', errors='ignore') as hyper_file:
        results = hyper_file.readlines()

    max_lines = len(results)
    index = 0
    auc_score = []
    ap_score = []

    while index < max_lines:
        line = results[index]
        if line.startswith("From AUC: Final Test AUC"):
            split = line.split("From AUC: Final Test AUC: ")[-1].split(f', Final Test AP: ')
            auc_score.append(float(split[0].strip()) * 100)
            ap_score.append(float(split[1].strip()) * 100)

        index += 1
    print("Done reading file")

    dataset_dict['AUC'] = auc_score
    dataset_dict['AP'] = ap_score
    auc = np.array(dataset_dict['AUC'])
    ap = np.array(dataset_dict['AP'])
    auc_mean = []
    ap_mean = []
    for index in range(0, num_datasets * 10, 10):
        auc_mean.append(f'{auc[index: index + 10].mean():.2f} ± {auc[index: index + 10].std():.2f}')
        ap_mean.append(f'{ap[index: index + 10].mean():.2f} ± {ap[index: index + 10].std():.2f}')
    dataset_dict["AUC Mean"] = auc_mean
    dataset_dict["AP Mean"] = ap_mean
    with open(f'final-result.json', 'w', encoding='utf-8') as fp:
        json.dump(dataset_dict, fp, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Read the results from WalkPool"
    )
    parser.add_argument('--file_name', type=str)
    parser.add_argument('--num_datasets', type=int)
    args = parser.parse_args()

    parse_results(args.file_name, args.num_datasets)