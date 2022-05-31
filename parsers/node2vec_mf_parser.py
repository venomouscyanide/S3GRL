import argparse
import json

import numpy as np

node2vec_mf_dict = \
    {
        'USAir':
            {
                'AUC': [],
                'AP': []
            },
        'NS':
            {
                'AUC': [],
                'AP': []
            },
        'Power':
            {
                'AUC': [],
                'AP': []
            },
        'Celegans':
            {
                'AUC': [],
                'AP': []
            },
        'Router':
            {
                'AUC': [],
                'AP': []
            },
        'PB':
            {
                'AUC': [],
                'AP': []
            },
        'Ecoli':
            {
                'AUC': [],
                'AP': []
            },
        'Yeast':
            {
                'AUC': [],
                'AP': []
            }
    }

def parse_n2vec_mf(file_name, method_type):
    with open(file_name, 'r') as hyper_file:
        results = hyper_file.readlines()

    max_lines = len(results)
    index = 0

    while index < max_lines:
        line = results[index]
        if line.startswith("Command line input:"):
            dataset = line.split("--dataset")[-1].split(f'--epochs')[0].strip()
            while 1:
                index += 1
                line = results[index]
                if line.startswith("All runs:"):
                    index += 4
                    line = results[index]
                    auc_score = float(line.split("(Precision of 5)Highest Test: ")[-1].split("± nan")[0].strip())

                    index += 7
                    line = results[index]
                    ap_score = float(line.split("(Precision of 5)Highest Test: ")[-1].split("± nan")[0].strip())

                    node2vec_mf_dict[dataset]['AUC'].append(auc_score)
                    node2vec_mf_dict[dataset]['AP'].append(ap_score)
                    index += 1
                    break
        index += 1
    print("Done reading file")

    for dataset in node2vec_mf_dict.keys():
        auc = np.array(node2vec_mf_dict[dataset]['AUC'])
        ap = np.array(node2vec_mf_dict[dataset]['AP'])
        node2vec_mf_dict[dataset]['AUC Mean'] = f'{auc.mean():.2f} ± {auc.std():.2f}'
        node2vec_mf_dict[dataset]['AP Mean'] = f'{ap.mean():.2f} ± {ap.std():.2f}'
    with open(f'{method_type}-result.json', 'w', encoding='utf-8') as fp:
        json.dump(node2vec_mf_dict, fp, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Read the results from heuristics and gae methods to record baselines"
    )
    parser.add_argument('--file_name', type=str)
    parser.add_argument('--method_type', type=str)
    args = parser.parse_args()

    parse_n2vec_mf(args.file_name, args.method_type)
