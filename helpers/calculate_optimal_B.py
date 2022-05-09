from GLOBAL_VARS import BOOTSTRAP_B, LOG

from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import scipy


def custom_mode(array):
    return scipy.stats.mode(array)


def calculate_optimal_B(B, predictions, y_true):

    R2_lst = {}
    pred_error_lst = {}
    B_lst = np.array(range(BOOTSTRAP_B)) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    plt.xlabel("Bootstrap B")
    fig.suptitle("Calculating optimal B")

    ax1.set_ylabel("R2")
    ax1.set_ylim([0, 1])
    ax2.set_ylabel("Std")
    LOG.process(f"Evaluating optimal B with max B: {BOOTSTRAP_B}")

    for agg_name, agg_func in zip(
        ["mean", "median", "mode"], [np.mean, np.median, scipy.stats.mode]
    ):
        y_true_exp = np.exp(y_true)
        R2_lst[agg_name] = []
        pred_error_lst[agg_name] = []
        for i in range(BOOTSTRAP_B):
            preds_of_interest = predictions[
                : (i + 1),
            ]

            agg_predictions = agg_func(
                preds_of_interest,
                axis=0,
            )
            if agg_name == "mode":
                agg_predictions = agg_predictions[0][0]

            agg_predictions_exp = np.exp(agg_predictions)
            R2 = np.square(np.corrcoef(agg_predictions_exp, y_true_exp))[0, 1]
            R2_lst[agg_name].append(R2)

            preds_centered = agg_predictions_exp - y_true_exp
            pred_error = np.std(preds_centered)
            pred_error_lst[agg_name].append(pred_error)
            # LOG.process(f"i: {i} R2: {R2}, pred_error:{pred_error}")

        ax1.plot(B_lst, R2_lst[agg_name], label=agg_name)
        ax2.plot(B_lst, pred_error_lst[agg_name], label=agg_name)

    ax1.legend()
    ax2.legend()
    plt.savefig("plots/calculate_optimal_B.png")
    plt.clf()
