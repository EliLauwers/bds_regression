import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV
from mpld3 import save_html

from GLOBAL_VARS import RANDOM_STATE, HYPERCV, NUM_CORES

import pandas as pd
import joblib


if __name__ == "__main__":

    ####
    #
    #  DECISIONS:
    # └ IPCA with 95% Explained variance => 158 Components
    # └ Mutual information with 5 neighbors and cutoff on .005
    # └ Isomap with 5
    #
    ####
    compute_elasticnet = True
    input_path = "data/intermediate/track_listens/"

    with open(input_path + "X_train.pk", "rb") as infile:
        X_train = pickle.load(infile)

    with open(input_path + "y_train.pk", "rb") as infile:
        y_train = pickle.load(infile)

    # Scale data with minmax scaler
    minmaxscaler = MinMaxScaler().fit(X_train)
    X_train[:] = minmaxscaler.transform(X_train)

    out_folder = "logs/rs2_dimreduct/"

    if compute_elasticnet:
        param_grid = {
            "l1_ratio": [.25, .5,.75, 1,]
        }
        model = GridSearchCV(
            estimator=ElasticNetCV(
                max_iter=10000,
                random_state=RANDOM_STATE,
                selection="random"
            ),
            param_grid=param_grid,
            cv=HYPERCV,
            n_jobs=NUM_CORES,
            verbose=3,
            scoring=[
                "neg_root_mean_squared_error",
                "neg_median_absolute_error",
                "r2",
            ],
            refit=False,
        ).fit(X_train, np.log(y_train))

        joblib.dump(model, out_folder + "elasticnet.pkl")

    with open(out_folder + "elasticnet" + ".pkl", "rb") as infile:
        results = pd.DataFrame(joblib.load(infile).cv_results_)

    fig, axes = plt.subplots(3, 1, figsize=(16, 16), sharex=False)

    metric_cols = [
        "test_r2",
        "test_neg_root_mean_squared_error",
        "test_neg_median_absolute_error",
    ]
    metric_names = ["R2", "Neg RMSE", "Neg MAE"]
    collection = zip(axes, metric_cols, metric_names)
    for i, (ax, col, name) in enumerate(collection):
        x_label = "l1_ratio"
        x_param = "param_" + x_label

        ax.set_xlabel("")
        if i == 0:
            ax.set_title("Evaluative Measures for elastict net")
        elif i == 2:
            ax.set_xlabel(x_label)

        interval = 1
        mean_ = results["mean_" + col]
        std_ = results["std_" + col]

        ax.errorbar(
            results[x_param],
            mean_,
            yerr=std_,
            ecolor="r",
            fmt="-",
            elinewidth=1,
            capsize=2.5,
        )
        ax.set_ylabel(name)
        ax.set_xlim(0, max(results[x_param]) + 1)
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, interval))

    plt.savefig(f"plots/rs2_dimreduct/elasticnet_{x_label}.png", dpi=300)
    save_html(fig, f"plots/rs2_dimreduct/elasticnet_{x_label}.html")
    plt.clf()
