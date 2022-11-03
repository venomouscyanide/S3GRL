import matplotlib.pyplot as plt


def plot_efficacy():
    us_air_auc_x_h1 = [92.86, 93.74, 93.99, 94.13, 94.19]
    us_air_auc_y_h1 = [_ for _ in range(1, 6, 1)]

    us_air_auc_x_h2 = [93.03, 93.91, 94.03, 94.10, 94.11, 94.05, 94.28, 94.25, 94.16]
    us_air_auc_y_h2 = [_ for _ in range(1, 10, 1)]

    cele_auc_x_h1 = [85.48, 86.69, 87.14, 87.43, 87.24]
    cele_auc_y_h1 = [_ for _ in range(1, 6, 1)]

    cele_auc_x_h2 = [86.33, 87.56, 87.42, 87.51, 87.41, 87.38, 87.69, 87.58, 87.69]
    cele_auc_y_h2 = [_ for _ in range(1, 10, 1)]

    ns_auc_x_h1 = [91.78, 93.37, 94.82, 95.55, 96.00]
    ns_auc_y_h1 = [_ for _ in range(1, 6, 1)]

    ns_auc_x_h2 = [91.65, 92.41, 93.29, 94.47, 94.63, 95.00, 95.34, 95.26, 95.40]
    ns_auc_y_h2 = [_ for _ in range(1, 10, 1)]

    fig, ax = plt.subplots()
    plt.plot(us_air_auc_y_h1, us_air_auc_x_h1, label="US Air SUP h=1")
    plt.plot(us_air_auc_y_h2, us_air_auc_x_h2, label="US Air SUP h=2")
    plt.plot(cele_auc_y_h1, cele_auc_x_h1, label="Celegans SUP h=1")
    plt.plot(cele_auc_y_h2, cele_auc_x_h2, label="Celegans SUP h=2")
    plt.plot(ns_auc_y_h1, ns_auc_x_h1, label="NS SUP h=1")
    plt.plot(ns_auc_y_h2, ns_auc_x_h2, label="NS SUP h=2")

    plt.legend()
    ax.set_ylabel("AUC")
    ax.set_xlabel("p number of SuP operators")
    fig.suptitle('SuP AUC vs (h,p) values',
                 fontsize=14, fontweight='bold')
    plt.show()


def plot_ds_prep_time():
    us_air_ds_x_h1 = [21.03, 23.93, 31.52, 34.43, 37.71]
    us_air_ds_y_h1 = [_ for _ in range(1, 6, 1)]

    us_air_ds_x_h2 = [28.52, 39.28, 50.29, 65.44, 92.72, 107.48, 116.76, 126.21, 137.98]
    us_air_ds_y_h2 = [_ for _ in range(1, 10, 1)]

    cele_ds_x_h1 = [17.64, 24.31, 27.49, 31.81, 33.90]
    cele_ds_y_h1 = [_ for _ in range(1, 6, 1)]

    cele_ds_x_h2 = [23.88, 37.29, 46.01, 57.89, 74.16, 79.00, 94.58, 106.06, 123.35]
    cele_ds_y_h2 = [_ for _ in range(1, 10, 1)]

    ns_ds_x_h1 = [23.90, 28.71, 31.17, 36.84, 41.99]
    ns_ds_y_h1 = [_ for _ in range(1, 6, 1)]

    ns_ds_x_h2 = [25.78, 30.77 , 34.13, 38.86, 45.05, 46.67, 48.66, 52.85, 56.65]
    ns_ds_y_h2 = [_ for _ in range(1, 10, 1)]

    fig, ax = plt.subplots()
    plt.plot(us_air_ds_y_h1, us_air_ds_x_h1, label="US Air SUP h=1")
    plt.plot(us_air_ds_y_h2, us_air_ds_x_h2, label="US Air SUP h=2")
    plt.plot(cele_ds_y_h1, cele_ds_x_h1, label="Celegans SUP h=1")
    plt.plot(cele_ds_y_h2, cele_ds_x_h2, label="Celegans SUP h=2")
    plt.plot(ns_ds_y_h1, ns_ds_x_h1, label="NS SUP h=1")
    plt.plot(ns_ds_y_h2, ns_ds_x_h2, label="NS SUP h=2")

    plt.legend()
    ax.set_ylabel("DS Prep Time (s)")
    ax.set_xlabel("p number of SuP operators")
    fig.suptitle('SuP DS Prep Time vs (h,p) values',
                 fontsize=14, fontweight='bold')
    plt.show()


if __name__ == '__main__':
    plot_efficacy()
    plot_ds_prep_time()
