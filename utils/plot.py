import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_curve(values, title, ylabel, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure()
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def moving_average(values, window=20):
    if len(values) < window:
        return values

    result = []

    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start : i + 1]) / len(values[start : i + 1]))

    return result
