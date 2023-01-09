import argparse
import json
from collections import defaultdict

from os import listdir
from os.path import isfile, join

import numpy as np


def parse_data(folder_path):
    # https://stackoverflow.com/a/3207973
    json_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    combined_profiling_results = {}

    mapp = defaultdict(list)
    for file in json_files:
        if file.split('_')[1] == 'LinkPred()':  # WP
            unique_id = f"{file.split('_')[1]}_{file.split('_')[2]}"
        else:
            unique_id = f"{file.split('_')[1]}_{file.split('_')[2]}_{file.split('_')[-1]}"
        with open(join(folder_path, file), "r") as json_file:
            mapp[unique_id].append(json.load(json_file))
    sorted_map = {k: v for k, v in sorted(mapp.items(), key=lambda x: x[0])}  # for readability

    for model_ds, profiled_info in sorted_map.items():
        avg_train_times = []
        avg_inf_times = []

        ds_prep_times = []

        max_allocated_cuda = []
        max_reserved_cuda = []
        max_active_cuda = []

        min_smi_free = []
        max_smi_used = []
        model_size = []

        num_params = []
        for each_seed_data in profiled_info:
            avg_train_times.append((float(each_seed_data['Average Train Time(in seconds)'].split('±')[0]),
                                    float(each_seed_data['Average Train Time(in seconds)'].split('±')[1])))
            avg_inf_times.append((float(each_seed_data['Average Inference Time(in seconds)'].split('±')[0])
                                  , float(each_seed_data['Average Inference Time(in seconds)'].split('±')[1])))

            ds_prep_times.append(float(each_seed_data['Dataset Prep Time(in seconds)']))

            max_allocated_cuda.append(float(each_seed_data['Max Allocated CUDA (in MegaBytes)']))
            max_reserved_cuda.append(float(each_seed_data['Max Reserved CUDA (in MegaBytes)']))
            max_active_cuda.append(float(each_seed_data['Max Active CUDA (in MegaBytes)']))

            min_smi_free.append(float(each_seed_data['Min NVIDIA SMI Free CUDA Memory (in MegaBytes)']))
            max_smi_used.append(float(each_seed_data['Max NVIDIA SMI Used CUDA Memory (in MegaBytes)']))
            model_size.append(float(each_seed_data['Model size (in MegaBytes)']))

            num_params.append(float(each_seed_data['Number of Model Parameters']))

        combined_profiling_results[model_ds] = {
            "Avg. Train Times": f"{np.array(list(map(lambda x: x[0], avg_train_times))).mean()} ± {np.array(list(map(lambda x: x[1], avg_train_times))).mean()}",
            "Avg. Inference Times": f"{np.array(list(map(lambda x: x[0], avg_inf_times))).mean()} ± {np.array(list(map(lambda x: x[1], avg_inf_times))).mean()}",
            "Avg. Dataset Prep Time": f"{np.array(ds_prep_times).mean()} ± {np.array(ds_prep_times).std()}",
            "Avg. Max. Allocated CUDA": f"{np.array(max_allocated_cuda).mean()} ± {np.array(max_allocated_cuda).std()}",
            "Avg. Reserved. Allocated CUDA": f"{np.array(max_reserved_cuda).mean()} ± {np.array(max_reserved_cuda).std()}",
            "Avg. Active. Allocated CUDA": f"{np.array(max_active_cuda).mean()} ± {np.array(max_active_cuda).std()}",
            "Avg. Min. SMI. Free CUDA": f"{np.array(min_smi_free).mean()} ± {np.array(min_smi_free).std()}",
            "Avg. Max. SMI. Used CUDA": f"{np.array(max_smi_used).mean()} ± {np.array(max_smi_used).std()}",
            "Avg. Model Size": f"{np.array(model_size).mean()} ± {np.array(model_size).std()}",
            "Avg. Num Params": f"{np.array(num_params).mean()} ± {np.array(num_params).std()}",
        }
        for key, value in combined_profiling_results[model_ds].items():
            combined_profiling_results[model_ds][
                key] = f"{float(value.split('±')[0]):.2f} ± {float(value.split('±')[1]):.2f}"

    with open(join("combined_profiling_results.json"), "w") as json_out:
        json.dump(combined_profiling_results, json_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)

    args = parser.parse_args()

    parse_data(args.folder)
