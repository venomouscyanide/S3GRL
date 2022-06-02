import argparse
import json

import numpy as np

# only retain data required for SEAL for hypertuning plots
seal_dict = \
    {

        'PB':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             },
        'Ecoli':
            {'AUC': [],
             'AP': [],
             'Time taken (per run)': [],
             },

    }


def parse_non_attributed_results(file_name):
    with open(file_name, 'r') as hyper_file:
        results = hyper_file.readlines()

    max_lines = len(results)
    index = 0

    while index < max_lines:
        line = results[index]
        if line.startswith("Command line input:"):
            dataset = line.split("--dataset")[-1].split(f'--epochs')[0].strip()
            result_dict = seal_dict

            while 1:
                index += 1
                line = results[index]
                if line.startswith("Summarized stats:"):
                    index += 1
                if line.startswith("All runs:"):
                    index += 4
                    line = results[index]
                    auc_score = float(line.split("(Precision of 5)Highest Test: ")[-1].split("± nan")[0].strip())

                    index += 8
                    line = results[index]
                    ap_score = float(line.split("(Precision of 5)Highest Test: ")[-1].split("± nan")[0].strip())

                    result_dict[dataset]['AUC'].append(auc_score)
                    result_dict[dataset]['AP'].append(ap_score)
                    index += 6
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
        ap = np.array(result_key['AP'])
        time_per_run = np.array(result_key['Time taken (per run)'])
        result_key['AUC Mean'] = f'{auc.mean():.2f} ± {auc.std():.2f}'
        result_key['AP Mean'] = f'{ap.mean():.2f} ± {ap.std():.2f}'
        result_key['Time taken (per run) Mean'] = f'{time_per_run.mean()}'
    with open(f'non-attributed-SEAL-tuner-result.json', 'w', encoding='utf-8') as fp:
        json.dump(result_dict, fp, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Read the results from SEAL and ScaLed log file "
    )
    parser.add_argument('--file_name', type=str)
    args = parser.parse_args()

    parse_non_attributed_results(args.file_name)
