import argparse
import json

import numpy as np

heuristic_dict = \
    {
        'USAir':
            {'CN':
                {
                    'AUC': [],
                    'AP': []
                },
                'AA':
                    {
                        'AUC': [],
                        'AP': []
                    },
                'PPR':
                    {
                        'AUC': [],
                        'AP': []
                    },
            },
        'NS':
            {'CN':
                {
                    'AUC': [],
                    'AP': []
                },
                'AA':
                    {
                        'AUC': [],
                        'AP': []
                    },
                'PPR':
                    {
                        'AUC': [],
                        'AP': []
                    },
            },
        'Power':
            {'CN':
                {
                    'AUC': [],
                    'AP': []
                },
                'AA':
                    {
                        'AUC': [],
                        'AP': []
                    },
                'PPR':
                    {
                        'AUC': [],
                        'AP': []
                    },
            },
        'Celegans':
            {'CN':
                {
                    'AUC': [],
                    'AP': []
                },
                'AA':
                    {
                        'AUC': [],
                        'AP': []
                    },
                'PPR':
                    {
                        'AUC': [],
                        'AP': []
                    },
            },
        'Router':
            {'CN':
                {
                    'AUC': [],
                    'AP': []
                },
                'AA':
                    {
                        'AUC': [],
                        'AP': []
                    },
                'PPR':
                    {
                        'AUC': [],
                        'AP': []
                    },
            },
        'PB':
            {'CN':
                {
                    'AUC': [],
                    'AP': []
                },
                'AA':
                    {
                        'AUC': [],
                        'AP': []
                    },
                'PPR':
                    {
                        'AUC': [],
                        'AP': []
                    },
            },
        'Ecoli':
            {'CN':
                {
                    'AUC': [],
                    'AP': []
                },
                'AA':
                    {
                        'AUC': [],
                        'AP': []
                    },
                'PPR':
                    {
                        'AUC': [],
                        'AP': []
                    },
            },
        'Yeast':
            {'CN':
                {
                    'AUC': [],
                    'AP': []
                },
                'AA':
                    {
                        'AUC': [],
                        'AP': []
                    },
                'PPR':
                    {
                        'AUC': [],
                        'AP': []
                    },
            },
        'Cora':
            {'CN':
                {
                    'AUC': [],
                    'AP': []
                },
                'AA':
                    {
                        'AUC': [],
                        'AP': []
                    },
                'PPR':
                    {
                        'AUC': [],
                        'AP': []
                    },
            },
        'CiteSeer':
            {'CN':
                {
                    'AUC': [],
                    'AP': []
                },
                'AA':
                    {
                        'AUC': [],
                        'AP': []
                    },
                'PPR':
                    {
                        'AUC': [],
                        'AP': []
                    },
            },
        'ogbl-collab':
            {'CN':
                {
                    'AUC': [],
                    'AP': []
                },
                'AA':
                    {
                        'AUC': [],
                        'AP': []
                    },
                'PPR':
                    {
                        'AUC': [],
                        'AP': []
                    },
            }
    }

gae_dict = \
    {
        'USAir':
            {'GCN':
                {
                    'AUC': [],
                    'AP': []
                },
                'SAGE':
                    {
                        'AUC': [],
                        'AP': []
                    },
                'GIN':
                    {
                        'AUC': [],
                        'AP': []
                    },
            },
        'NS':
            {'GCN':
                {
                    'AUC': [],
                    'AP': []
                },
                'SAGE':
                    {
                        'AUC': [],
                        'AP': []
                    },
                'GIN':
                    {
                        'AUC': [],
                        'AP': []
                    },
            },
        'Power':
            {'GCN':
                {
                    'AUC': [],
                    'AP': []
                },
                'SAGE':
                    {
                        'AUC': [],
                        'AP': []
                    },
                'GIN':
                    {
                        'AUC': [],
                        'AP': []
                    },
            },
        'Celegans':
            {'GCN':
                {
                    'AUC': [],
                    'AP': []
                },
                'SAGE':
                    {
                        'AUC': [],
                        'AP': []
                    },
                'GIN':
                    {
                        'AUC': [],
                        'AP': []
                    },
            },
        'Router':
            {'GCN':
                {
                    'AUC': [],
                    'AP': []
                },
                'SAGE':
                    {
                        'AUC': [],
                        'AP': []
                    },
                'GIN':
                    {
                        'AUC': [],
                        'AP': []
                    },
            },
        'PB':
            {'GCN':
                {
                    'AUC': [],
                    'AP': []
                },
                'SAGE':
                    {
                        'AUC': [],
                        'AP': []
                    },
                'GIN':
                    {
                        'AUC': [],
                        'AP': []
                    },
            },
        'Ecoli':
            {'GCN':
                {
                    'AUC': [],
                    'AP': []
                },
                'SAGE':
                    {
                        'AUC': [],
                        'AP': []
                    },
                'GIN':
                    {
                        'AUC': [],
                        'AP': []
                    },
            },
        'Yeast':
            {'GCN':
                {
                    'AUC': [],
                    'AP': []
                },
                'SAGE':
                    {
                        'AUC': [],
                        'AP': []
                    },
                'GIN':
                    {
                        'AUC': [],
                        'AP': []
                    },
            },
        'Cora':
            {'GCN':
                {
                    'AUC': [],
                    'AP': []
                },
                'SAGE':
                    {
                        'AUC': [],
                        'AP': []
                    },
                'GIN':
                    {
                        'AUC': [],
                        'AP': []
                    },
            },
        'CiteSeer':
            {'GCN':
                {
                    'AUC': [],
                    'AP': []
                },
                'SAGE':
                    {
                        'AUC': [],
                        'AP': []
                    },
                'GIN':
                    {
                        'AUC': [],
                        'AP': []
                    },
            },
        'ogbl-collab':
            {'GCN':
                {
                    'AUC': [],
                    'AP': []
                },
                'SAGE':
                    {
                        'AUC': [],
                        'AP': []
                    },
                'GIN':
                    {
                        'AUC': [],
                        'AP': []
                    },
            }
    }


def parse_baseline_results(file_name, method_type):
    with open(file_name, 'r') as hyper_file:
        results = hyper_file.readlines()

    max_lines = len(results)
    index = 0

    while index < max_lines:
        line = results[index]
        if line.startswith("Command line input:"):
            dataset = line.split("--dataset")[-1].split(f'--{method_type}')[0].strip()
            if method_type == 'use_heuristic':
                model_name = line.split(f'--{method_type}')[-1].split(f'--')[0].strip()
                result_dict = heuristic_dict
            else:
                model_name = line.split(f'--model')[-1].split(f'--')[0].strip()
                result_dict = gae_dict
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

                    result_dict[dataset][model_name]['AUC'].append(auc_score)
                    result_dict[dataset][model_name]['AP'].append(ap_score)
                    index += 1
                    break
        index += 1
    print("Done reading file")

    for dataset in result_dict.keys():
        baseline_methods = result_dict[dataset].keys()
        for key in baseline_methods:
            result_key = result_dict[dataset][key]
            auc = np.array(result_key['AUC'])
            ap = np.array(result_key['AP'])
            result_key['AUC Mean'] = f'{auc.mean():.2f} ± {auc.std():.2f}'
            result_key['AP Mean'] = f'{ap.mean():.2f} ± {ap.std():.2f}'
    with open(f'{method_type}-result.json', 'w', encoding='utf-8') as fp:
        json.dump(result_dict, fp, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Read the results from heuristics and gae methods to record baselines"
    )
    parser.add_argument('--file_name', type=str)
    parser.add_argument('--method_type', type=str)
    args = parser.parse_args()

    parse_baseline_results(args.file_name, args.method_type)
