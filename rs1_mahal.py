from helpers.calculate_mahalanobis_distance import mahalanobis_iterative
import pickle
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats

if __name__ == "__main__":
    with open("data/intermediate/track_listens/X_train.pk", "rb") as infile:
        X_train = pickle.load(infile)

    mahal_distances = mahalanobis_iterative(X_train)
    p_values = sorted(1 - scipy.stats.chi2.cdf(mahal_distances, X_train.shape[1] - 1))
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.hist(p_values, bins=30)
    plt.savefig("plots/rs1_mahal/mahal_hist.png")
    plt.clf()

    count, bins_count = np.histogram(p_values, bins=len(X_train))
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    plt.plot(bins_count[1:], cdf)
    plt.title("Cumulative Frequencies")
    plt.xaxis("P value")
    plt.savefig("plots/rs1_mahal/mahal_cumul.png")
