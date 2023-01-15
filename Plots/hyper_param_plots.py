import warnings
from matplotlib import MatplotlibDeprecationWarning

warnings.simplefilter('ignore', MatplotlibDeprecationWarning)

import argparse
import json

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# enforce 3 fonts
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def plot_helper(result_json):
    with open('sensitivity_graph_data_pos.json', 'rb') as results_json:
        results_dict_ksup = json.load(results_json)
    with open('sensitivity_graph_data_k_sup.json', 'rb') as results_json:
        results_dict_pos = json.load(results_json)
    results_dict_ksup.update(results_dict_pos)
    results_dict = results_dict_ksup
    # multi-line graph plots for hyperparameter tuning results
    all_plot_data = defaultdict(list)
    for identifier, values in results_dict.items():
        key = identifier.split('_')[0] + '_' + identifier.split('_')[1] + '_' + identifier.split('_')[-2]
        dataset_prep_time = float(values['results']['Average Dataset Prep Time'].split('±')[0])
        acc = float(values['results']['Average Test AUC'].split('±')[0])
        all_plot_data[key] += [(acc, dataset_prep_time)]

    plot_ds(y_label="AUC %", data_index=0, dataset_name='Cora', all_plot_data=all_plot_data)
    plot_ds(y_label="Dataset Prep. Time (sec.)", data_index=1, dataset_name='Cora', all_plot_data=all_plot_data)
    plot_ds(y_label="AUC %", data_index=0, dataset_name='CiteSeer', all_plot_data=all_plot_data)
    plot_ds(y_label="Dataset Prep. Time (sec.)", data_index=1, dataset_name='CiteSeer', all_plot_data=all_plot_data)


def plot_ds(y_label, data_index, dataset_name, all_plot_data):
    # colors = ['b', 'g', 'r', 'm', 'c', 'y', 'k', 'w']
    # slice_length = 5
    # cmap = [plt.cm.get_cmap("Reds"), plt.cm.get_cmap("Greens")]
    # slicedCM = [cmap[0](np.linspace(0.4, 0.75, slice_length)), cmap[1](np.linspace(0.5, 0.8, slice_length))]
    line_style = ['solid', 'dotted', 'dashed', 'dashdot']
    marker_style = ['D', 's', 'o', '^']

    f = plt.figure()
    x = [1, 2, 3, 4, 5]
    default_x_ticks = range(len(x))
    plt.rcParams.update({'font.size': 16.5})
    plt.xticks(default_x_ticks, x)

    if y_label != "Dataset Prep. Time (sec.)":
        plt.yticks(np.arange(80, 99, 1))
        plt.ylim(85)
    else:
        plt.yticks(np.arange(15, 200, 10))
        # plt.ylim(150)

    all_plot_data = {k: v for k, v in filter(lambda x: x[0].split('_')[0] == dataset_name, all_plot_data.items())}
    for index, (dataset, results) in enumerate(all_plot_data.items()):
        dataset_name = dataset.split('_')[0]
        model_name = dataset.split('_')[1]
        if model_name == 'K':
            model_name = "SuP+"

        y_plots = list(map(lambda x: x[data_index], results))
        if model_name == 'PoS':
            index = 1
        plt.plot(default_x_ticks, y_plots, label=f"{model_name} h{index}",
                 linestyle=line_style[index],
                 marker=marker_style[index], linewidth=2, markersize=10)

    plt.ylabel(y_label, labelpad=0)
    plt.xlabel('Num. Operators (R)', labelpad=0)
    if y_label != "Dataset Prep. Time (sec.)":
        file_name = 'auc'
        plt.legend(loc="lower left", ncol=2, borderpad=0.2, labelspacing=0.1, borderaxespad=0.1, columnspacing=0.8)
    else:
        file_name = 'ds_prep'
        plt.legend(loc="upper left", ncol=2, borderpad=0.2, labelspacing=0.1, borderaxespad=0.1, columnspacing=0.8)

    plt.show()
    f.savefig(f"{dataset_name.lower()}_{file_name}.pdf", bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_json', type=str, required=True)

    args = parser.parse_args()

    plot_helper(args.results_json)
