import argparse
import json

import numpy as np
from torch_geometric.profile.utils import byte_to_megabyte

from utils import human_format

seal_dict = \
    {
        'Cora':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'CiteSeer':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'Pubmed':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'attributed-Facebook':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'Corak3':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'CiteSeerk3':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'Corak-':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'CiteSeerk-':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'Pubmedk3':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'Pubmedk-':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'attributed-Facebookk1':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'attributed-Facebookk-':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
    }

scaled_dict = \
    {
        'Corak3_1_10':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'Corak3_1_20':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'Corak-1_1_10':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'Corak-1_1_20':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'CiteSeerk3_1_10':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'CiteSeerk3_1_20':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'CiteSeerk-1_1_10':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'CiteSeerk-1_1_20':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'Pubmedk3_1_10':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'Pubmedk3_1_20':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'Pubmedk-1_1_10':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'Pubmedk-1_1_20':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'attributed-Facebookk3_1_10':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'attributed-Facebookk3_1_20':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'attributed-Facebookk-1_1_10':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },
        'attributed-Facebookk-1_1_20':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             'Max allocated CUDA': [],
             'Model size': [],
             'Parameters': []
             },

    }


def parse_attributed_results(file_name, method_type):
    with open(file_name, 'r') as hyper_file:
        results = hyper_file.readlines()

    max_lines = len(results)
    index = 0

    while index < max_lines:
        line = results[index]
        if line.startswith("Command line input:"):
            if method_type == 'seal':
                dataset = line.split("--dataset")[-1].split(f'--epochs')[0].strip()
                suffix = ''
                if '--sign_k' in line:
                    suffix = 'k' + line.split('--sign_k ')[-1][0]
                dataset += suffix
                result_dict = seal_dict
            else:
                dataset = line.split("--dataset")[-1].split(f'--epochs')[0].strip()
                if dataset == 'attributed-Facebook':
                    identifier = line.split("--data_appendix ")[-1].split(f'--num_layers ')[0][len(dataset.split('-')[-1]) + 1: -3]
                else:
                    identifier = line.split("--data_appendix ")[-1].split(f'--model ')[0][len(dataset) + 1: -3]
                dataset += identifier
                result_dict = scaled_dict
            while 1:
                index += 1
                line = results[index]
                if line.startswith("Summarized stats:"):
                    max_allocated_cuda = float(line.split("max_allocated_cuda=")[-1].split(f',')[0].strip())
                    max_nvidia_smi_cuda = float(line.split("max_nvidia_smi_used_cuda=")[-1].split(f')')[0].strip())
                    result_dict[dataset]['Max allocated CUDA'].append(max_allocated_cuda)
                    index += 1
                if line.startswith("Model size"):
                    model_size = float(line.split("Model size: ")[-1])
                    index += 1
                    line = results[index]
                    parameters = float(line.split("Parameters: ")[-1])
                    result_dict[dataset]['Model size'].append(byte_to_megabyte(model_size))
                    result_dict[dataset]['Parameters'].append(parameters)
                if line.startswith("All runs:"):
                    index += 4
                    line = results[index]
                    auc_score = float(line.split("(Precision of 5)Highest Test: ")[-1].split("± nan")[0].strip())

                    index += 8
                    line = results[index]
                    ap_score = float(line.split("(Precision of 5)Highest Test: ")[-1].split("± nan")[0].strip())

                    result_dict[dataset]['AUC'].append(auc_score)
                    result_dict[dataset]['AP'].append(ap_score)
                    index += 3
                    line = results[index]
                    parameters = int(line.split("Total number of parameters is ")[-1])
                    result_dict[dataset]['Parameters'].append(parameters)

                    index += 3
                    line = results[index]
                    time_taken_for_run = float(line.split("Time taken for run: ")[-1].split("seconds")[0].strip())
                    result_dict[dataset]['Time taken (per run)'].append(time_taken_for_run)

                    index += 1
                    break
        index += 1
    print("Done reading file")

    for dataset in result_dict.keys():
        result_key = result_dict[dataset]
        auc = np.array(result_key['AUC'])
        if not result_key['AUC']:
            continue
        ap = np.array(result_key['AP'])
        time_per_run = np.array(result_key['Time taken (per run)'])
        allocated_cuda = np.array(result_key['Max allocated CUDA'])
        model_size = np.array(result_key['Model size'])
        parameters = np.array(result_key['Parameters'])
        result_key['AUC Mean'] = f'{auc.mean():.2f} ± {auc.std():.2f}'
        result_key['AP Mean'] = f'{ap.mean():.2f} ± {ap.std():.2f}'
        result_key['Time taken (per run) Mean'] = f'{round(time_per_run.mean())}'
        # result_key['Max allocated CUDA Mean'] = f'{round(allocated_cuda.mean())}'
        # result_key['Model size Mean'] = f'{model_size.mean():.2f}'
        result_key['Parameters Mean'] = f'{human_format(parameters.mean())}'
    with open(f'attributed-{method_type}-result.json', 'w', encoding='utf-8') as fp:
        json.dump(result_dict, fp, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Read the results from SEAL and ScaLed log file "
    )
    parser.add_argument('--file_name', type=str)
    parser.add_argument('--method_type', type=str)
    args = parser.parse_args()

    parse_attributed_results(args.file_name, args.method_type)
