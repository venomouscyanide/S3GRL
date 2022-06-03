import matplotlib.pyplot as plt
import numpy as np


# omit CiteSeer for paper
class HyperTunerResults:
    M = [2, 5, 10, 20]
    m = [2, 3, 5, 7]
    RESULTS_NON = {
        'Cora': {
            'AUC_MEAN': [82.76017199999998, 87.487588, 88.65829599999999, 90.11079, 84.747722, 87.780028, 90.22709,
                         90.57181399999999, 86.170114, 89.230798, 90.51989599999999, 90.35685799999999, 86.692998,
                         89.488528, 90.05988, 90.15644999999999],
            'AP_MEAN': [85.65929, 89.32734599999999, 90.479214, 91.173516, 87.654978, 89.718496, 91.420358, 91.392862,
                        88.585766, 90.39572, 91.406922, 91.088056, 88.461682, 90.37998999999999, 91.12392600000001,
                        91.008716],
            'Time_MEAN': [100.79, 101.43199999999999, 96.05600000000001, 97.9, 98.32000000000001, 94.14200000000001,
                          97.922,
                          104.936, 101.85799999999999, 99.46199999999999, 105.138, 116.602, 96.34, 103.88399999999999,
                          119.73800000000001, 134.08800000000002]
        },
        # 'CiteSeer': {
        #     'AUC_MEAN': [85.08373399999999, 86.104286, 87.15218, 87.177106, 85.804516, 86.45613, 87.81770200000001,
        #                  87.696074, 85.83997, 86.89675000000001, 88.01285200000001, 87.871078, 86.656876,
        #                  88.07767199999998,
        #                  87.94512999999999, 88.747158],
        #     'AP_MEAN': [87.368156, 88.33165399999999, 89.46185599999998, 89.532334, 88.180022, 88.854016, 89.849406,
        #                 89.740984, 88.258682, 89.41571600000002, 89.51878199999999, 89.72151999999998,
        #                 89.08202400000002,
        #                 89.82256, 89.86403600000001, 90.475782],
        #     'Time_MEAN': [96.97999999999999, 94.96399999999998, 93.978, 96.912, 91.71799999999999, 97.24399999999999,
        #                   99.49,
        #                   107.628, 92.07200000000002, 96.85999999999999, 107.048, 116.96, 93.804, 107.81199999999998,
        #                   119.17, 131.362]
        # }
    }
    SEAL = {
        'Cora': {
            'AUC_MEAN': [90.13],
            'AP_MEAN': [90.71],
            'Time_MEAN': [258.184]},
        # 'CiteSeer': {
        #     'AUC_MEAN': [88.55],
        #     'AP_MEAN': [90.16],
        #     'Time_MEAN': [202.076]
        # }
    }


if __name__ == '__main__':
    # multi-line graph plots for hyperparameter tuning results
    slice_length = len(HyperTunerResults.m)
    # colors = ['b', 'g', 'r', 'm', 'c', 'y', 'k', 'w']
    cmap = [plt.cm.get_cmap("Reds"), plt.cm.get_cmap("Greens")]
    slicedCM = [cmap[0](np.linspace(0.4, 0.75, slice_length)), cmap[1](np.linspace(0.5, 0.8, slice_length))]
    line_style = ['solid', 'dotted', 'dashed', 'dashdot']
    marker_style = ['D', 's', 'o', '^']
    seal_colors = ['midnightblue', 'darkgreen', ]

    for dataset, results in HyperTunerResults.RESULTS_NON.items():
        all_auc = results['AUC_MEAN']
        all_ap = results['AP_MEAN']
        all_times = results['Time_MEAN']

        time_SEAL_results = HyperTunerResults.SEAL[dataset]['Time_MEAN'] * 4

        auc_m_results = [all_auc[i:i + slice_length] for i in range(0, len(all_auc), slice_length)]
        time_m_results = [all_times[i:i + slice_length] for i in range(0, len(all_times), slice_length)]

        HyperTunerResults.RESULTS_NON[dataset].update(
            {'auc_m_results': auc_m_results, 'time_m_results': time_m_results}
        )

    f = plt.figure()
    x = HyperTunerResults.M
    default_x_ticks = range(len(x))
    plt.rcParams.update({'font.size': 16.5})
    plt.xticks(default_x_ticks, x)
    plt.yticks(np.arange(0, 92, 1))
    plt.ylim(79)
    for index, (dataset, results) in enumerate(HyperTunerResults.RESULTS_NON.items()):
        # SEAL line
        auc_SEAL_results = HyperTunerResults.SEAL[dataset]['AUC_MEAN'] * 4
        plt.plot(default_x_ticks, auc_SEAL_results, label=f"SEAL h=3", color=seal_colors[index],
                 linestyle='-', linewidth=2, markersize=10)

        auc_m_results = results['auc_m_results']
        for inner_index, m_values in enumerate(auc_m_results):
            plt.plot(default_x_ticks, m_values, label=f"ScaLed h={HyperTunerResults.m[inner_index]}",
                     color=slicedCM[index][inner_index], linestyle=line_style[inner_index],
                     marker=marker_style[inner_index], linewidth=2, markersize=10)

    plt.ylabel('AUC Scores')
    plt.xlabel('k: Number of Walks')
    plt.legend(loc="lower right", ncol=2, borderpad=0.2, labelspacing=0.25, borderaxespad=0.25)
    plt.show()
    f.savefig("hypertuner_attr_auc.pdf", bbox_inches='tight')

    f = plt.figure()
    x = HyperTunerResults.M
    default_x_ticks = range(len(x))
    plt.rcParams.update({'font.size': 16.5})
    plt.xticks(default_x_ticks, x)
    plt.yticks(np.arange(0, 300, 25))
    plt.ylim(0)
    for index, (dataset, results) in enumerate(HyperTunerResults.RESULTS_NON.items()):
        # SEAL line
        auc_SEAL_results = HyperTunerResults.SEAL[dataset]['Time_MEAN'] * 4
        plt.plot(default_x_ticks, auc_SEAL_results, label=f"SEAL h=3", color=seal_colors[index],
                 linestyle='-', linewidth=2, markersize=10)

        auc_m_results = results['time_m_results']
        for inner_index, m_values in enumerate(auc_m_results):
            plt.plot(default_x_ticks, m_values, label=f"ScaLed h={HyperTunerResults.m[inner_index]}",
                     color=slicedCM[index][inner_index],
                     linestyle=line_style[inner_index], marker=marker_style[inner_index], linewidth=2, markersize=10)

    plt.ylabel('Runtime (sec)')
    plt.xlabel('k: Number of Walks')
    plt.legend(loc="lower right", ncol=2, borderpad=0.2, labelspacing=0.25, borderaxespad=0.25)
    plt.show()
    f.savefig("hypertuner_attr_time.pdf", bbox_inches='tight')
