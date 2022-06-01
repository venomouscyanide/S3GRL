import matplotlib.pyplot as plt
import numpy as np


class HyperTunerResults:
    M = [2, 5, 10, 20]
    m = [2, 3, 5, 7]
    RESULTS = {
        'PB': {
            'AUC_MEAN': [86.95306200000002, 91.919426, 93.503864, 94.188968, 88.041946, 92.15317, 93.88848, 94.212748,
                         89.777362, 92.67347, 93.89034000000001, 94.099282, 90.51614, 92.896852, 93.80544799999998,
                         93.962928],
            'AP_MEAN': [87.197488, 91.49574, 93.15968, 93.832426, 88.13949, 91.916772, 93.56088, 93.91488,
                        89.80645799999999, 92.52873400000001, 93.68607999999999, 93.86442000000001, 90.58420400000001,
                        92.826278, 93.629088, 93.843884],
            'Time_MEAN': [497.05, 482.36800000000005, 482.36, 472.59800000000007, 446.838, 454.36, 465.054,
                          488.28999999999996, 450.816, 470.18600000000004, 495.264, 525.884, 458.56399999999996, 478.35,
                          504.11, 561.1859999999999]},
        'Ecoli': {
            'AUC_MEAN': [],
            'AP_MEAN': [],
            'Time_MEAN': []
        }
    }
    SEAL = {
        'PB': {
            'AUC_MEAN': [94.44],
            'AP_MEAN': [94.08],
            'Time_MEAN': [1015.35]},
        'Ecoli': {
            'AUC_MEAN': [],
            'AP_MEAN': [],
            'Time_MEAN': []
        }
    }


if __name__ == '__main__':
    # multi-line graph plots for hyperparameter tuning results
    slice_length = len(HyperTunerResults.m)
    # colors = ['b', 'g', 'r', 'm', 'c', 'y', 'k', 'w']
    cmap = plt.cm.get_cmap("OrRd")
    slicedCM = cmap(np.linspace(0.5, 1, slice_length))
    line_style = ['solid', 'dotted', 'dashed', 'dashdot']
    marker_style = ['D', 's', 'o', '^']
    for dataset, results in HyperTunerResults.RESULTS.items():
        x = HyperTunerResults.M

        all_auc = results['AUC_MEAN']
        all_ap = results['AP_MEAN']
        all_times = results['Time_MEAN']

        auc_SEAL_results = HyperTunerResults.SEAL[dataset]['AUC_MEAN'] * 4
        ap_SEAL_results = HyperTunerResults.SEAL[dataset]['AUC_MEAN'] * 4
        time_SEAL_results = HyperTunerResults.SEAL[dataset]['Time_MEAN'] * 4

        auc_m_results = [all_auc[i:i + slice_length] for i in range(0, len(all_auc), slice_length)]
        ap_m_results = [all_ap[i:i + slice_length] for i in range(0, len(all_ap), slice_length)]
        time_m_results = [all_times[i:i + slice_length] for i in range(0, len(all_times), slice_length)]

        default_x_ticks = range(len(x))
        plt.xticks(default_x_ticks, x)
        # SEAL line
        plt.plot(default_x_ticks, auc_SEAL_results, label=f"SEAL", color='b', linestyle='-')
        for index, m_values in enumerate(auc_m_results):
            plt.plot(default_x_ticks, m_values, label=f"h={HyperTunerResults.m[index]}", color=slicedCM[index],
                     linestyle=line_style[index], marker=marker_style[index])

        plt.ylabel('AUC')
        plt.xlabel('k')
        plt.legend(loc="lower right")
        plt.title(f"{dataset} AUC vs. (h, k)")
        plt.show()

        default_x_ticks = range(len(x))
        plt.xticks(default_x_ticks, x)
        # SEAL line
        plt.plot(default_x_ticks, ap_SEAL_results, label=f"SEAL", color='b', linestyle='-')
        for index, m_values in enumerate(ap_m_results):
            plt.plot(default_x_ticks, m_values, label=f"h={HyperTunerResults.m[index]}", color=slicedCM[index],
                     linestyle=line_style[index], marker=marker_style[index])

        plt.ylabel('AP')
        plt.xlabel('k')
        plt.legend(loc="lower right")
        plt.title(f"{dataset} AP vs. (h, k)")
        plt.show()

        default_x_ticks = range(len(x))
        plt.xticks(default_x_ticks, x)
        # SEAL line
        plt.plot(default_x_ticks, time_SEAL_results, label=f"SEAL", color='b', linestyle='-')
        for index, m_values in enumerate(time_m_results):
            plt.plot(default_x_ticks, m_values, label=f"h={HyperTunerResults.m[index]}", color=slicedCM[index],
                     linestyle=line_style[index], marker=marker_style[index])

        plt.ylabel('Time per run')
        plt.xlabel('k')
        plt.legend(loc="lower right")
        plt.title(f"{dataset} Time per run(in sec) vs. (h, k)")
        plt.show()
