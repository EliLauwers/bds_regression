from helpers.calculate_mahalanobis_distance import mahalanobis_iterative
import pickle
import json
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
from GLOBAL_VARS import FHEIGHT, FWIDTH

if __name__ == "__main__":
    with open("data/intermediate/track_listens/X_train.pk", "rb") as infile:
        X_train = pickle.load(infile)

    fig, rows = plt.subplots(
        2, 2, figsize=(FWIDTH * 3, FHEIGHT * 2 + 2), sharex=False, sharey=False
    )

    for i, var in enumerate(["track_listens", "album_date_released"]):
        ax1 = rows[0][i]
        ax2 = rows[1][i]
        with open(f"logs/{var}/mahal.json", "rb") as infile:
            mahal_distances = json.load(infile)

        p_values = sorted(
            1 - scipy.stats.chi2.cdf(mahal_distances, X_train.shape[1] - 1)
        )
        ax1.hist(p_values, bins=30)
        ax1.set_title(f"Variable: {var}")
        ax1.set_xlabel("Mahal D")
        ax1.set_ylabel("Absolute Frequency")

        count, bins_count = np.histogram(p_values, bins=len(X_train))
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)

        ax2.plot(bins_count[1:], cdf)
        ax2.set_title("Cumulative Frequencies")
        ax2.set_ylabel("Relative Frequencies")
        ax2.set_xlabel("P value")
        ax2.axvline(x=0.05, color="r", linestyle=":")
        ax2.annotate(".05", (0.06, 0.1), color="r")
        ax2.set_ylim(0, 1)

    plt.savefig("plots/rs1_mahal/mahal.png")
    plt.clf()
