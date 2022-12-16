import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# enforce 3 fonts
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def plot_helper(result_json):
    with open('3_vs_4_vs_4(1).json', 'rb') as results_json:
        results_dict = json.load(results_json)

    # multi-line graph plots for hyperparameter tuning results
    all_plot_data = defaultdict(list)
    for identifier, values in results_dict.items():
        key = f"{identifier.split('_')[0]}_{identifier.split('_')[-2]}_{identifier.split('_')[-1]}"
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
    line_style = ['solid', 'dashed', 'dashed', 'dashdot', '-.', ':']
    marker_style = ['D', 's', 'o', '^', 'X', 'h']

    f = plt.figure()
    x = (list(range(1, 6, 1)))
    # p = [2, 3, 4, 5]
    labels = ['(3) Mean', '(3) Sum', '(4) Mean', '(4) Sum', '(4.1) Mean', '(4.1) Sum']
    default_x_ticks = range(len(x))
    plt.rcParams.update({'font.size': 16.5})
    plt.xticks(default_x_ticks, x)

    # plt.yticks(np.arange(0, 99, 0.5))
    # plt.ylim(79)

    all_plot_data = {k: v for k, v in filter(lambda x: x[0].split('_')[0] == dataset_name, all_plot_data.items())}
    for index, (dataset, results) in enumerate(all_plot_data.items()):
        dataset = dataset.split('_')[0]

        y_plots = list(map(lambda x: x[data_index], results))

        plt.plot(default_x_ticks, y_plots, label=f"{labels[index]}",
                 linestyle=line_style[index],
                 marker=marker_style[index], linewidth=2, markersize=10)

    plt.ylabel(y_label, labelpad=0)
    plt.xlabel('Num. operators', labelpad=0)
    plt.legend(loc="best", ncol=2, borderpad=0.2, labelspacing=0.1, borderaxespad=0.1)
    plt.show()
    f.savefig(f"{dataset}_hypertuner_{y_label.lower()}.pdf", bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_json', type=str, required=False)

    args = parser.parse_args()

    plot_helper(args.results_json)
