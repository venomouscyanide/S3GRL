import matplotlib.pyplot as plt
import numpy as np


class HyperTunerResults:
    M = [2, 5, 10, 20]
    m = [2, 3, 5, 7]
    RESULTS_NON = {

        'Cora': {
            'AUC_MEAN': [82.795892, 87.50818399999999, 88.65851, 90.108126, 84.739438, 87.784636, 90.21895400000001,
                         90.576714, 86.17458, 89.20602799999999, 90.269076, 90.286718, 86.71763000000001, 89.521512,
                         90.061032, 90.11467999999999],
            'AP_MEAN': [85.65971400000001, 89.324008, 90.479642, 91.191348, 87.64832399999999, 89.728382, 91.410282,
                        91.393292, 88.51729399999999, 90.36761600000001, 91.41886400000001, 91.129458,
                        88.49078799999998, 90.40688800000001, 91.177946, 90.991376],
            'Time_MEAN': [90.34599999999999, 98.786, 87.256, 93.15599999999999, 99.93800000000002, 91.304, 90.17,
                          95.03799999999998, 100.95599999999999, 91.01, 103.51999999999998, 115.81999999999998,
                          87.90599999999999, 98.372, 105.024, 119.71599999999998]},
        'CiteSeer': {
            'AUC_MEAN': [85.08412, 86.107088, 87.200388, 87.15836399999999, 85.809636, 86.51254800000001,
                         87.80968399999999, 87.698296, 85.83997, 86.925442, 88.031304, 87.894262, 86.656682,
                         88.080136,
                         87.91431, 88.783964],
            'AP_MEAN': [87.36841799999999, 88.331196, 89.42752, 89.52173, 88.18007800000001, 88.879274, 89.840972,
                        89.739004, 88.258682, 89.42994600000002, 89.52581799999999, 89.751534, 89.077964, 89.825172,
                        89.851604, 90.49150599999999],
            'Time_MEAN': [70.86800000000001, 70.468, 79.068, 80.622, 71.54, 76.13, 84.54599999999999, 91.784,
                          74.382,
                          82.24000000000001, 90.67, 100.51400000000001, 79.448, 86.15799999999999,
                          95.65799999999999,
                          110.768]}
    }

    SEAL = {
        'Cora': {
            'AUC_MEAN': [90.19],
            'AP_MEAN': [90.80],
            'Time_MEAN': [234.98999999999995]},
        'CiteSeer': {
            'AUC_MEAN': [88.48],
            'AP_MEAN': [90.13],
            'Time_MEAN': [190.35999999999999]
        }
    }


if __name__ == '__main__':
    # multi-line graph plots for hyperparameter tuning results
    slice_length = len(HyperTunerResults.m)
    # colors = ['b', 'g', 'r', 'm', 'c', 'y', 'k', 'w']
    cmap = [plt.cm.get_cmap("OrRd"), plt.cm.get_cmap("BuGn")]
    slicedCM = [cmap[0](np.linspace(0.5, 1, slice_length)), cmap[1](np.linspace(0.5, 1, slice_length))]
    line_style = ['solid', 'dotted', 'dashed', 'dashdot']
    marker_style = ['D', 's', 'o', '^']
    seal_colors = ['b', 'm']

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
    plt.xticks(default_x_ticks, x)
    for index, (dataset, results) in enumerate(HyperTunerResults.RESULTS_NON.items()):
        # SEAL line
        auc_SEAL_results = HyperTunerResults.SEAL[dataset]['AUC_MEAN'] * 4
        plt.plot(default_x_ticks, auc_SEAL_results, label=f"{dataset} SEAL", color=seal_colors[index], linestyle='-')

        auc_m_results = results['auc_m_results']
        for inner_index, m_values in enumerate(auc_m_results):
            plt.plot(default_x_ticks, m_values, label=f"{dataset} h={HyperTunerResults.m[inner_index]}",
                     color=slicedCM[index][inner_index],
                     linestyle=line_style[inner_index], marker=marker_style[inner_index])

    plt.ylabel('AUC Scores on testing split')
    plt.xlabel('k: Number of walks')
    plt.legend(loc="lower right")
    plt.title(f"Attributed Datasets AUC vs. (h, k)")
    plt.show()
    f.savefig("hypertuner_attr_auc.pdf", bbox_inches='tight')

    f = plt.figure()
    x = HyperTunerResults.M
    default_x_ticks = range(len(x))
    plt.xticks(default_x_ticks, x)
    for index, (dataset, results) in enumerate(HyperTunerResults.RESULTS_NON.items()):
        # SEAL line
        auc_SEAL_results = HyperTunerResults.SEAL[dataset]['Time_MEAN'] * 4
        plt.plot(default_x_ticks, auc_SEAL_results, label=f"{dataset} SEAL", color=seal_colors[index], linestyle='-')

        auc_m_results = results['time_m_results']
        for inner_index, m_values in enumerate(auc_m_results):
            plt.plot(default_x_ticks, m_values, label=f"{dataset} h={HyperTunerResults.m[inner_index]}",
                     color=slicedCM[index][inner_index],
                     linestyle=line_style[inner_index], marker=marker_style[inner_index])

    plt.ylabel('Time taken per run in seconds')
    plt.xlabel('k: Number of walks')
    plt.legend(loc="upper right")
    plt.title(f"Attributed Datasets Time per run vs. (h, k)")
    plt.show()
    f.savefig("hypertuner_attr_time.pdf", bbox_inches='tight')
