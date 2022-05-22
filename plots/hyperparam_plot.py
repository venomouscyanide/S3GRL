import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == '__main__':
    # graph plot
    x = [5, 10, 20, 50, 100]

    m5_acc = [10, 25, 12, 40.99, 40]
    m2_acc = [5, 15, 14, 10, 35]
    default_x_ticks = range(len(x))

    plt.xticks(default_x_ticks, x)
    # plt.yticks(np.arange(0, 100, 5))
    plt.plot(default_x_ticks, m5_acc, label="m=5", color="purple", linestyle="-")
    plt.plot(default_x_ticks, m2_acc, label="m=2", color="red", linestyle="-")
    plt.ylabel('Accuracy')
    plt.xlabel('M')
    plt.legend(loc="upper right")
    plt.show()

    # heatmap
    uniform_data = np.random.rand(10, 12)

    ax = sns.heatmap(uniform_data, annot=True, cmap="YlGnBu")
    plt.show()
