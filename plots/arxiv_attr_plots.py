import matplotlib.pyplot as plt
import numpy as np


# omit CiteSeer for paper
class HyperTunerResults:
    M = [2, 5, 10, 20, 40]
    m = [1, 2, 3, 5, 7]

    RESULTS_NON = {'Cora': {
        'AUC_MEAN': [78.657398, 85.77736000000002, 87.1088, 87.45698, 88.13490999999999, 82.765792, 87.47923200000001,
                     88.650012, 90.10898999999999, 89.874088, 84.74376199999999, 87.77750999999999, 90.21564199999999,
                     90.552734, 90.628634, 86.15254200000001, 89.210204, 90.243368, 90.291398, 90.814354,
                     86.82888799999999, 89.5236, 90.06686599999999, 90.10963999999998, 90.265906],
        'AP_MEAN': [82.903088, 88.027448, 89.061708, 89.35637399999999, 89.870742, 85.64673, 89.324092,
                    90.46394599999999, 91.178108, 91.243876, 87.63453, 89.70203199999999, 91.41711999999998, 91.370374,
                    91.551234, 88.51356999999999, 90.376946, 91.409238, 91.13826200000001, 91.468296, 88.49256,
                    90.40113999999998, 91.332562, 91.00063, 91.215956],
        'Time_MEAN': [90.36000000000001, 90.422, 96.17, 88.8, 99.454, 94.538, 102.354, 98.164, 101.702,
                      107.31200000000001, 98.068, 94.364, 98.97, 105.96199999999999, 118.38600000000001, 107.056,
                      96.074, 111.96799999999999, 127.8, 137.048, 95.21600000000001, 104.50399999999999, 121.35,
                      144.48600000000002, 152.55]}, 'CiteSeer': {
        'AUC_MEAN': [82.278176, 85.310082, 86.13404, 85.4861, 85.856394, 85.073202, 86.117424, 87.196234,
                     87.15797799999999, 86.330832, 85.804516, 86.459608, 87.76186399999999, 87.67249999999999,
                     87.726508, 85.842288, 86.90100000000001, 88.03265400000001, 87.834564, 88.11032399999999,
                     86.64992000000001, 88.08839599999999, 87.91779, 88.75420799999999, 88.10674800000001],
        'AP_MEAN': [85.142882, 88.12067199999998, 88.607684, 88.3067, 88.236426, 87.358022, 88.331174,
                    89.48388800000001, 89.520364, 88.993114, 88.180022, 88.707264, 89.82346399999999, 89.72636400000002,
                    89.7126, 88.257678, 89.419182, 89.506942, 89.71539200000001, 89.968366, 89.07495800000001,
                    89.82086799999999, 89.84497200000001, 90.484148, 89.92393999999999],
        'Time_MEAN': [95.146, 88.55199999999999, 92.22800000000001, 90.288, 86.422, 88.77799999999999,
                      91.94200000000001, 89.77000000000001, 98.63, 103.58600000000001, 91.762, 93.94399999999999,
                      98.186, 99.224, 111.76400000000001, 90.91199999999999, 101.52799999999999, 110.09200000000001,
                      118.80799999999999, 129.384, 96.156, 102.112, 119.80999999999999, 126.596, 148.11599999999999]}}
    SEAL = {'Cora': {'AUC': [90.93577, 87.77513, 89.90779, 92.03936, 90.06226],
                     'AP': [91.32137, 88.10345, 90.76731, 93.68123, 89.95426],
                     'Time taken (per run)': [237.07, 247.75, 249.54, 285.63, 251.94], 'AUC Mean': [90.14],
                     'AP Mean': '90.77 ± 1.82', 'Time taken (per run) Mean': [254.38600000000002]},
            'CiteSeer': {'AUC': [87.36553, 88.52554, 90.39971, 89.096, 87.21966],
                         'AP': [89.14829, 90.63891, 91.90871, 90.62441, 88.48923],
                         'Time taken (per run)': [165.44, 191.32, 200.13, 200.27, 216.85], 'AUC Mean': [88.52],
                         'AP Mean': '90.16 ± 1.21', 'Time taken (per run) Mean': [194.802]}}


if __name__ == '__main__':
    # multi-line graph plots for hyperparameter tuning results
    slice_length = len(HyperTunerResults.m)
    # colors = ['b', 'g', 'r', 'm', 'c', 'y', 'k', 'w']
    cmap = [plt.cm.get_cmap("Reds"), plt.cm.get_cmap("Greens")]
    slicedCM = [cmap[0](np.linspace(0.4, 0.75, slice_length)), cmap[1](np.linspace(0.5, 0.8, slice_length))]
    # line_style = ['solid', 'dotted', 'dashed', 'dashdot', ]
    line_style = [('solid', 'solid'),
                  ('dotted', 'dotted'),
                  ('dashed', 'dashed'),
                  ('dashdot', 'dashdot'),
                  ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5)))]
    marker_style = ['D', 's', 'o', '^', '*']
    seal_colors = ['midnightblue', 'darkgreen', ]

    for dataset, results in HyperTunerResults.RESULTS_NON.items():
        all_auc = results['AUC_MEAN']
        all_ap = results['AP_MEAN']
        all_times = results['Time_MEAN']

        auc_m_results = [all_auc[i:i + slice_length] for i in range(0, len(all_auc), slice_length)]
        time_m_results = [all_times[i:i + slice_length] for i in range(0, len(all_times), slice_length)]

        HyperTunerResults.RESULTS_NON[dataset].update(
            {'auc_m_results': auc_m_results, 'time_m_results': time_m_results}
        )

    ylimit = [70, 79]
    skip_axis = [2, 2]

    for index, (dataset, results) in enumerate(HyperTunerResults.RESULTS_NON.items()):
        f = plt.figure()
        x = HyperTunerResults.M
        default_x_ticks = range(len(x))
        plt.rcParams.update({'font.size': 16.5})
        plt.xticks(default_x_ticks, x)
        plt.yticks(np.arange(0, 95, skip_axis[index]))
        plt.ylim(ylimit[index])
        # SEAL line
        auc_SEAL_results = HyperTunerResults.SEAL[dataset]['AUC Mean'] * 5
        plt.plot(default_x_ticks, auc_SEAL_results, label=f"SEAL h=2", color=seal_colors[0],
                 linestyle='-', linewidth=2, markersize=10)

        auc_m_results = results['auc_m_results']

        for inner_index, m_values in enumerate(auc_m_results):
            plt.plot(default_x_ticks, m_values, label=f"ScaLed h={HyperTunerResults.m[inner_index]}",
                     color=slicedCM[0][inner_index],
                     linestyle=line_style[inner_index][1], marker=marker_style[inner_index], linewidth=2, markersize=10)

        plt.ylabel('AUC Scores')
        plt.xlabel('k: Number of Walks')
        plt.legend(loc="lower right", ncol=2, borderpad=0.2, labelspacing=0.25, borderaxespad=0.25)
        plt.tight_layout()
        plt.show()
        f.savefig(f"{dataset}_hypertuner_auc.pdf", bbox_inches='tight')

    ylimit = [20, 30]
    max_limit = [300, 230]
    skip_axis = [30, 30]

    for index, (dataset, results) in enumerate(HyperTunerResults.RESULTS_NON.items()):
        f = plt.figure()
        x = HyperTunerResults.M
        default_x_ticks = range(len(x))
        plt.rcParams.update({'font.size': 16.5})
        plt.xticks(default_x_ticks, x)
        plt.yticks(np.arange(0, max_limit[index], skip_axis[index]))
        plt.ylim(ylimit[index])
        # SEAL line
        auc_SEAL_results = HyperTunerResults.SEAL[dataset]['Time taken (per run) Mean'] * 5
        plt.plot(default_x_ticks, auc_SEAL_results, label=f"SEAL h=2", color=seal_colors[0],
                 linestyle='-', linewidth=2, markersize=10)

        auc_m_results = results['time_m_results']

        for inner_index, m_values in enumerate(auc_m_results):
            plt.plot(default_x_ticks, m_values, label=f"ScaLed h={HyperTunerResults.m[inner_index]}",
                     color=slicedCM[0][inner_index],
                     linestyle=line_style[inner_index][1], marker=marker_style[inner_index], linewidth=2, markersize=10)

        plt.ylabel('Runtime (sec)')
        plt.xlabel('k: Number of Walks')
        plt.legend(loc="lower right", ncol=2, borderpad=0.2, labelspacing=0.25, borderaxespad=0.25)
        plt.tight_layout()
        plt.show()
        f.savefig(f"{dataset}_hypertuner_time.pdf", bbox_inches='tight')
