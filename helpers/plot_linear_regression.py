from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def plot_linear_regression(y_true, y_pred):
    y_true_exp = np.exp(y_true)
    y_pred_exp = np.exp(y_pred)
    # Plot the model
    scatter_points = {"s": .1, "alpha": .3}
    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches([12, 6])
    sns.regplot(y_true, y_pred, ax=axes[0][0], scatter_kws=scatter_points, lowess=True)
    axes[0][0].set(xlabel="track_listens_log")
    sns.residplot(y_true, y_pred, ax=axes[0][1], scatter_kws=scatter_points)
    axes[0][1].set(xlabel="track_listens_log")
    sns.regplot(y_true_exp, y_pred_exp, ax=axes[1][0], scatter_kws=scatter_points)
    axes[1][0].set(xlabel="track_listens")
    sns.residplot(y_true_exp, y_pred_exp, ax=axes[1][1], scatter_kws=scatter_points)
    axes[1][1].set(xlabel="track_listens")
    for row in axes:
        for ax in row:
            ax.set(ylabel="predictions")
    plt.show()
    plt.close()
