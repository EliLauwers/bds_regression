from matplotlib import pyplot as plt
from tkinter import Grid
from GLOBAL_VARS import FHEIGHT, FWIDTH, RANDOM_STATE, HYPERCV, LOG, NUM_CORES

# Normal imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import sklearn
from mpld3 import save_html
import joblib

import pickle
import random

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


if __name__ == "__main__":
    # Conclusions
    # - Decision tree with depth of 5

    min_samples_leaf = False
    max_depth = False
    min_samples_split = False
    full_model = False

    input_path = "data/intermediate/track_listens/"

    with open(input_path + "X_train.pk", "rb") as infile:
        X_train = pickle.load(infile)

    with open(input_path + "y_train.pk", "rb") as infile:
        y_train = pickle.load(infile)

    # Scale data with minmax scaler
    minmaxscaler = MinMaxScaler().fit(X_train)
    X_train[:] = minmaxscaler.transform(X_train)

    # Decision tree min samples leaf

    if min_samples_leaf:

        param_grid = {"min_samples_leaf": range(100, 10001, 100)}
        model = GridSearchCV(
            estimator=DecisionTreeRegressor(),
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

        joblib.dump(
            model.cv_results_, "logs/rs4_estimators/dt_min_samples_leaf.pkl"
        )

    # open results
    results = pd.DataFrame(
        joblib.load("logs/rs4_estimators/dt_min_samples_leaf.pkl")
    )

    fig, axes = plt.subplots(3, 1, figsize=(16, 16), sharex=False)

    metric_cols = [
        "test_r2",
        "test_neg_root_mean_squared_error",
        "test_neg_median_absolute_error",
    ]
    metric_names = ["R2", "Neg RMSE", "Neg MAE"]
    collection = zip(axes, metric_cols, metric_names)
    for i, (ax, col, name) in enumerate(collection):

        x_label = "min_samples_leaf"
        x_param = "param_" + x_label

        ax.set_xlabel("")
        if i == 0:
            ax.set_title("Evaluative Measures for min_samples_leaf")
        elif i == 2:
            ax.set_xlabel(x_label)

        interval = 500
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
        ax.set_xlim(0, max(results[x_param]) + interval)
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, interval))

    plt.savefig(f"plots/rs4_estimators/knn_{x_label}.png", dpi=300)
    save_html(fig, f"plots/rs4_estimators/knn_{x_label}.html")
    plt.clf()

    # Decision tree max depth
    # https://analyticsindiamag.com/guide-to-hyperparameters-tuning-using-gridsearchcv-and-randomizedsearchcv/
    # https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680
    if max_depth:
        param_grid = {"max_depth": range(1, 31)}
        model = GridSearchCV(
            estimator=DecisionTreeRegressor(),
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

        joblib.dump(model.cv_results_, "logs/rs4_estimators/dt_max_depth.pkl")

    results = pd.DataFrame(joblib.load("logs/rs4_estimators/dt_max_depth.pkl"))

    # start plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 16), sharex=False)

    metric_cols = [
        "test_r2",
        "test_neg_root_mean_squared_error",
        "test_neg_median_absolute_error",
    ]
    metric_names = ["R2", "Neg RMSE", "Neg MAE"]
    collection = zip(axes, metric_cols, metric_names)
    for i, (ax, col, name) in enumerate(collection):
        x_label = "max_depth"
        x_param = "param_" + x_label

        ax.set_xlabel("")
        if i == 0:
            ax.set_title("Evaluative Measures for Decision Tree Depth")
        elif i == 2:
            ax.set_xlabel(x_label)

        interval = 5
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

    plt.savefig(f"plots/rs4_estimators/dt_{x_label}.png", dpi=300)
    save_html(fig, f"plots/rs4_estimators/dt_{x_label}.html")
    plt.clf()

    # Decision tree min samples split

    if min_samples_split:
        param_grid = {"min_samples_split": range(100, 10001, 100)}
        model = GridSearchCV(
            estimator=DecisionTreeRegressor(),
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

        joblib.dump(
            model.cv_results_, "logs/rs4_estimators/dt_min_samples_split.pkl"
        )

    results = pd.DataFrame(
        joblib.load("logs/rs4_estimators/dt_min_samples_split.pkl")
    )
    # start plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 16), sharex=False)

    metric_cols = [
        "test_r2",
        "test_neg_root_mean_squared_error",
        "test_neg_median_absolute_error",
    ]
    metric_names = ["R2", "Neg RMSE", "Neg MAE"]
    collection = zip(axes, metric_cols, metric_names)
    for i, (ax, col, name) in enumerate(collection):

        x_label = "min_samples_split"
        x_param = "param_" + x_label

        ax.set_xlabel("")
        if i == 0:
            ax.set_title(
                "Evaluative Measures for Decision Tree min_samples_split")
        elif i == 2:
            ax.set_xlabel(x_label)

        interval = 500
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
        ax.set_xlim(0, max(results[x_param]) + interval)
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, interval))

    plt.savefig(f"plots/rs4_estimators/dt_{x_label}.png", dpi=300)
    save_html(fig, f"plots/rs4_estimators/dt_{x_label}.html")
    plt.clf()

    if full_model:
        # Decision tree grid search
        param_grid = {
            "splitter": ["best", "random"],
            "max_depth": [None, 6],
            "max_features": [0.25, 0.5, 0.75, 1.0],
        }
        model = GridSearchCV(
            estimator=DecisionTreeRegressor(
                criterion="squared_error",
                min_samples_split=1300,
                min_samples_leaf=700,
                random_state=RANDOM_STATE
            ),
            param_grid=param_grid,
            cv=10,
            verbose=3,
            scoring=[
                "neg_root_mean_squared_error",
                "neg_median_absolute_error",
                "r2",
            ],
            n_jobs=NUM_CORES,
            refit=False
        ).fit(X_train, np.log(y_train))

        joblib.dump(
            model, "logs/rs4_estimators/dt_full_model.pkl"
        )

    with open("logs/rs4_estimators/dt_full_model.pkl", "rb") as infile:
        results = pd.DataFrame(joblib.load(infile).cv_results_)
    fig, axes = plt.subplots(3, 2, figsize=(
        FHEIGHT * 3, FWIDTH * 2), sharex=False, sharey="row")

    metric_cols = [
        "test_r2",
        "test_neg_root_mean_squared_error",
        "test_neg_median_absolute_error",
    ]
    metric_names = ["R2", "Neg RMSE", "Neg MAE"]
    collection = zip(axes, metric_cols, metric_names)
    for i, (row, col, name) in enumerate(collection):
        for splitter, ax in zip(["random", "best"], row):

            x_label = "max_features"
            x_param = "param_" + x_label

            ax.set_xlabel("")
            if i == 0:
                ax.set_title("Splitter: " + splitter)
            elif i == 2:
                ax.set_xlabel(x_label)

            interval = .25
            for depth in [None, 6]:
                # import ipdb; ipdb.set_trace()
                logvec = []
                for par in results.params:
                    if par["splitter"] != splitter:
                        logvec.append(False)
                        continue
                    if depth:
                        if par["max_depth"] != depth:
                            logvec.append(False)
                            continue
                    else:
                        if par["max_depth"]:
                            logvec.append(False)
                            continue
                    logvec.append(True)

                mean_ = results.iloc[logvec, :]["mean_" + col]
                std_ = results.iloc[logvec, :]["std_" + col]

                # TODO color every line
                ax.errorbar(
                    results.iloc[logvec, :][x_param],
                    mean_,
                    yerr=std_,
                    ecolor="r" if depth else "gray",
                    elinewidth=1,
                    capsize=2.5,
                    label=f"depth: {depth if depth is not None else 'None'}"
                )
                ax.set_ylabel(name)
                ax.set_xlim(0, 1.25)
                start, end = ax.get_xlim()
                ax.xaxis.set_ticks(np.arange(start, end, interval))
                if i == 0 and splitter == "random":
                    ax.legend()

    plt.savefig(f"plots/rs4_estimators/dt_full_model.png", dpi=300)
    save_html(fig, f"plots/rs4_estimators/dt_full_model.html")
    plt.clf()
