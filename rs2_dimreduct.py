import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import IncrementalPCA, KernelPCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.manifold import Isomap
import sklearn
from GLOBAL_VARS import RANDOM_STATE
import json
import pandas as pd


def simulate_mutual_information():
    results = []
    for n_neighbors in [1, 3, 5, 7, 9, 11]:
        print(f"trying {n_neighbors} neighbors")
        mi = mutual_info_regression(
            X_train, np.log(y_train), n_neighbors=n_neighbors, random_state=RANDOM_STATE
        )

        for mi_cutoff in np.arange(0, 1, 0.001):
            mi_cutoff = round(mi_cutoff, 3)
            mi_logvec = np.where(mi >= mi_cutoff)
            if len(mi_logvec[0]) == 0:
                continue
            model = LinearRegression().fit(
                X_train.iloc[:, mi_logvec[0]], np.log(y_train)
            )
            predictions = model.predict(X_test.iloc[:, mi_logvec[0]])
            RMSE = sklearn.metrics.mean_squared_error(
                np.log(y_test), predictions, squared=False
            )
            R2 = sklearn.metrics.r2_score(np.log(y_test), predictions)
            results.append(
                {
                    "n_neighbors": n_neighbors,
                    "mi_cutoff": mi_cutoff,
                    "RMSE": RMSE,
                    "R2": R2,
                }
            )

    with open("logs/rs2_dimreduct/mutual_information.json", "w") as outfile:
        outfile.write(json.dumps(results))
    return


if __name__ == "__main__":

    ####
    #
    #  DECISIONS:
    # └ IPCA with 95% Explained variance => 158 Components
    # └ Mutual information with 5 neighbors and cutoff on .005
    # └ Isomap with 5
    #
    ####

    input_path = "data/intermediate/track_listens/"
    with open(input_path + "X_train.pk", "rb") as infile:
        X_train = pickle.load(infile)
    with open(input_path + "y_train.pk", "rb") as infile:
        y_train = pickle.load(infile)
    with open(input_path + "X_test.pk", "rb") as infile:
        X_test = pickle.load(infile)
    with open(input_path + "y_test.pk", "rb") as infile:
        y_test = pickle.load(infile)

    scaler = MinMaxScaler().fit(X_train)
    X_train[:] = scaler.transform(X_train)
    X_test[:] = scaler.transform(X_test)

    # Elastic net?
    RMSES = []
    R2S = []
    iS = np.arange(50, 1000, 50)

    for i, n_alpha in enumerate(iS):

        print(f"\rn_alphas {i + 1} of {len(iS)}", end=" " * 10)

        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
            n_alphas=n_alpha,
            cv=10,
            random_state=RANDOM_STATE,
            max_iter = 1000000
        ).fit(X_train, np.log(y_train))
        predictions = model.predict(X_test)
        RMSE = sklearn.metrics.mean_squared_error(
            np.log(y_test), predictions, squared=False
        )
        R2 = sklearn.metrics.r2_score(np.log(y_test), predictions)
        RMSES.append(RMSE)
        R2S.append(R2)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(28, 6))

        plot_is = iS[: len(R2S)]
        ax1.plot(plot_is, R2S)
        ax1.set_xlabel("n_alpha")
        ax1.set_ylabel("R2")

        ax2.plot(plot_is, RMSES)
        ax2.set_xlabel("n_alpha")
        ax2.set_ylabel("RMSE")
        plt.savefig("plots/rs2_dimreduct/elastic_net.png")

    # Incremental PCA
    ipca = IncrementalPCA().fit(X_train)
    fig, ax = plt.subplots(figsize=(12, 6))
    xi = np.arange(len(ipca.explained_variance_ratio_), step=1)
    y = np.cumsum(ipca.explained_variance_ratio_)
    plt.plot(xi, y, marker=".", linestyle="--", color="b")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative variance (%)")
    plt.title("The number of components needed to explain variance")
    plt.axhline(y=0.95, color="r", linestyle="--")
    plt.axhline(y=0.99, color="r", linestyle="--")
    plt.text(0, 0.955, ".95", color="red", fontsize=10)
    plt.text(0, 0.995, ".99", color="red", fontsize=10)
    ax.grid(axis="x")
    plt.savefig("plots/rs2_dimreduct/incremental_pca.png")
    plt.ylim(0.94, 1)
    plt.xlim(125, 325)
    plt.savefig("plots/rs2_dimreduct/incremental_pca_zoom.png")
    plt.clf()

    # mutual information
    # baseline picture

    predictions_base = LinearRegression().fit(X_train, np.log(y_train)).predict(X_test)
    RMSE_base = sklearn.metrics.mean_squared_error(
        np.log(y_test), predictions_base, squared=False
    )
    R2_base = sklearn.metrics.r2_score(np.log(y_test), predictions_base)
    # simulate_mutual_information()
    with open("logs/rs2_dimreduct/mutual_information.json") as infile:
        results = json.load(infile)
        data = pd.DataFrame(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.set_title("R2")
    ax1.set_xlabel("Mutual Information Cutoff")
    ax1.set_ylabel("R2")

    ax1.set_ylim(0, 1.01)
    ax1.axhline(y=R2_base, color="r", linestyle="--")
    ax1.text(0, R2_base + 0.1 * R2_base, "Baseline", color="r", fontsize=10)

    ax2.axhline(y=RMSE_base, color="r", linestyle="--")
    ax2.text(0, RMSE_base + 0.1 * RMSE_base, "Baseline", color="r", fontsize=10)

    ax2.set_title("RMSE")
    ax2.set_xlabel("Mutual Information Cutoff")
    ax2.set_ylabel("RMSE")
    for n_neighbor in data["n_neighbors"].unique():
        tmp = data[data.n_neighbors == n_neighbor]
        ax1.plot(tmp["mi_cutoff"], tmp["R2"], label=n_neighbor)
        ax2.plot(tmp["mi_cutoff"], tmp["RMSE"], label=n_neighbor)
    ax1.set_xlim(0, 0.02)
    ax2.set_xlim(0, 0.02)
    ax1.legend()
    ax2.legend()
    plt.savefig("plots/rs2_dimreduct/mutual_information.png")
    plt.clf()
