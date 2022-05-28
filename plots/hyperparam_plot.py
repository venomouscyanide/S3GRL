import matplotlib.pyplot as plt
import numpy as np


class HyperTunerResults:
    M = [2, 5, 10, 20]
    m = [2, 3, 5, 7]
    RESULTS = {
        "USAir": {
            "AUC": [90.99, 94.45, 94.63, 95.11, 90.52, 94.97, 94.55, 95.57, 92.50, 95.24, 95.22, 95.53, 93.54, 95.27,
                    95.29, 95.05],
            "AP": [91.65, 94.58, 94.82, 95.21, 91.04, 94.90, 94.80, 95.47, 92.68, 95.25, 95.25, 95.56, 93.76, 95.27,
                   95.28, 95.16],
            "Average Time": [65.85, 67.45, 66.81, 64.00, 62.46, 62.53, 65.22, 64.89, 67.89, 70.08, 71.66, 69.50, 69.70,
                             71.00, 72.26, 75.88]
        },
        "NS": {
            "AUC": [98.56, 99.23, 98.99, 98.94, 98.96, 98.99, 99.08, 97.98, 98.89, 98.90, 99.02, 98.86, 98.79, 97.77,
                    97.89, 98.94],
            "AP": [98.92, 99.40, 99.24, 99.21, 99.22, 99.24, 99.25, 98.54, 99.18, 99.18, 99.22, 99.07, 99.11, 98.34,
                   98.29, 99.08],
            "Average Time": [90.97, 81.18, 76.04, 83.18, 82.43, 79.83, 80.25, 80.26, 79.21, 80.00, 88.06, 88.84, 86.47,
                             86.63, 86.60, 81.04]
        },
        "Power": {
            "AUC": [74.64, 82.50, 81.80, 84.93, 77.02, 83.17, 84.15, 85.58, 79.01, 84.38, 86.12, 88.08, 79.98, 85.45,
                    87.25, 88.93],
            "AP": [79.04, 85.08, 84.52, 86.92, 81.10, 85.63, 86.44, 87.93, 82.93, 86.61, 88.12, 89.57, 83.50, 87.36,
                   88.91, 90.16],
            "Average Time": [181.57, 195.48, 194.62, 210.35, 208.60, 211.45, 194.01, 197.67, 206.31, 214.22, 210.30,
                             195.83, 203.17, 194.30, 191.50, 213.19]
        },
        "Celegans": {
            "AUC": [66.91, 78.23, 82.58, 82.36, 70.19, 78.67, 83.85, 83.87, 70.39, 80.66, 84.77, 85.82, 72.92, 80.97,
                    84.90, 85.68],
            "AP": [67.33, 77.01, 81.59, 82.61, 70.30, 76.54, 82.91, 83.43, 70.58, 79.06, 84.27, 85.75, 72.55, 80.74,
                   84.41, 85.84],
            "Average Time": [62.76, 60.49, 61.88, 61.83, 60.61, 61.76, 62.43, 63.29, 59.63, 61.46, 62.43, 63.64, 60.71,
                             61.11, 61.47, 66.11]
        },
        "Router": {
            "AUC": [88.92, 92.69, 93.77, 93.29, 91.17, 93.82, 93.31, 93.52, 90.83, 93.03, 92.95, 92.11, 92.92, 92.11,
                    92.53, 93.18],
            "AP": [87.95, 92.87, 93.99, 93.60, 90.70, 94.09, 93.45, 93.77, 90.44, 93.47, 93.18, 93.42, 92.76, 92.45,
                   92.95, 93.49],
            "Average Time": [179.71, 171.15, 171.44, 172.18, 173.09, 171.48, 172.29, 171.57, 169.04, 168.90, 172.55,
                             175.23, 169.51, 172.88, 173.24, 174.95]
        },
        "PB": {
            "AUC": [85.99, 91.50, 93.10, 93.95, 87.33, 92.04, 93.46, 94.21, 89.34, 92.39, 93.70, 94.11, 90.29, 92.63,
                    93.64, 94.27],
            "AP": [85.83, 91.10, 92.56, 93.52, 87.30, 91.63, 93.04, 93.89, 89.63, 92.07, 93.48, 93.92, 90.55, 92.40,
                   93.44, 94.03],
            "Average Time": [466.15, 494.92, 527.70, 537.90, 492.15, 524.50, 511.41, 531.90, 494.96, 512.28, 534.60,
                             535.95, 504.05, 473.27, 502.16, 566.59]
        },
        "Ecoli": {
            "AUC": [93.05, 95.61, 96.12, 96.48, 93.52, 96.19, 96.65, 96.81, 94.35, 96.39, 96.83, 96.78, 94.65, 96.26,
                    96.45, 96.77],
            "AP": [94.33, 96.46, 96.86, 97.18, 94.63, 96.84, 97.24, 97.38, 95.42, 97.01, 97.58, 97.41, 95.61, 96.98,
                   97.30, 97.48],
            "Average Time": [428.34, 422.87, 428.57, 487.31, 433.96, 434.65, 470.98, 492.37, 422.92, 447.43, 431.45,
                             471.01, 412.03, 449.14, 479.10, 564.36]
        },
        "Yeast": {
            "AUC": [92.69, 96.18, 97.07, 97.17, 92.75, 96.34, 97.12, 97.22, 94.17, 96.45, 97.05, 97.07, 93.94, 96.47,
                    96.86, 96.88],
            "AP": [94.50, 96.90, 97.62, 97.66, 94.80, 97.01, 97.62, 97.75, 95.52, 97.03, 97.56, 97.62, 95.65, 97.15,
                   97.44, 97.42],
            "Average Time": [342.97, 347.33, 340.00, 345.97, 341.35, 340.08, 353.88, 353.75, 343.83, 330.72, 360.99,
                             355.29, 353.38, 367.46, 353.87, 386.02]
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

        all_auc = results['AUC']
        all_ap = results['AP']
        all_times = results['Average Time']

        auc_m_results = [all_auc[i:i + slice_length] for i in range(0, len(all_auc), slice_length)]
        ap_m_results = [all_ap[i:i + slice_length] for i in range(0, len(all_ap), slice_length)]
        time_m_results = [all_times[i:i + slice_length] for i in range(0, len(all_times), slice_length)]

        default_x_ticks = range(len(x))
        plt.xticks(default_x_ticks, x)
        for index, m_values in enumerate(auc_m_results):
            plt.plot(default_x_ticks, m_values, label=f"m={HyperTunerResults.m[index]}", color=slicedCM[index],
                     linestyle=line_style[index], marker=marker_style[index])

        plt.ylabel('AUC')
        plt.xlabel('M')
        plt.legend(loc="lower right")
        plt.title(f"{dataset} AUC vs. m")
        plt.show()

        default_x_ticks = range(len(x))
        plt.xticks(default_x_ticks, x)
        for index, m_values in enumerate(ap_m_results):
            plt.plot(default_x_ticks, m_values, label=f"m={HyperTunerResults.m[index]}", color=slicedCM[index],
                     linestyle=line_style[index], marker=marker_style[index])

        plt.ylabel('AP')
        plt.xlabel('M')
        plt.legend(loc="lower right")
        plt.title(f"{dataset} AP vs. m")
        plt.show()

        default_x_ticks = range(len(x))
        plt.xticks(default_x_ticks, x)
        for index, m_values in enumerate(time_m_results):
            plt.plot(default_x_ticks, m_values, label=f"m={HyperTunerResults.m[index]}", color=slicedCM[index],
                     linestyle=line_style[index], marker=marker_style[index])

        plt.ylabel('Time per run')
        plt.xlabel('M')
        plt.legend(loc="lower right")
        plt.title(f"{dataset} Time per run(in sec) vs. m")
        plt.show()
