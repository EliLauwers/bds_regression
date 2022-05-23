from tkinter import Grid
from GLOBAL_VARS import RANDOM_STATE

# Normal imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import sklearn

np.random.seed(RANDOM_STATE)

import pickle
import random

random.seed(RANDOM_STATE)

from matplotlib import pyplot as plt

if __name__ == "__main__":
    # Conclusions
    # - Decision tree with depth of 5

    min_samples_leaf = True
    min_samples_split = True
    max_depth = False
    gridsearch = False

    input_path = "data/intermediate/track_listens/"

    with open(input_path + "X_train.pk", "rb") as infile:
        X_train = pickle.load(infile)

    with open(input_path + "y_train.pk", "rb") as infile:
        y_train = pickle.load(infile)

    with open(input_path + "X_test.pk", "rb") as infile:
        X_test = pickle.load(infile)

    with open(input_path + "y_test.pk", "rb") as infile:
        y_test = pickle.load(infile)

    # Scale data with minmax scaler
    minmaxscaler = MinMaxScaler().fit(X_train)
    X_train[:] = minmaxscaler.transform(X_train)
    X_test[:] = minmaxscaler.transform(X_test)

    # Decision tree min samples leaf

    if min_samples_leaf:
        RMSES = []
        R2S = []
        iS = range(100, 10000, 100)
        for min_samples in iS:
            print(
                f"\rTrying min samples leaf {min_samples} with max {max(iS)}",
                end=" " * 8,
            )
            model = DecisionTreeRegressor(min_samples_leaf=min_samples).fit(
                X_train, np.log(y_train)
            )
            predictions = model.predict(X_test)
            RMSE = sklearn.metrics.mean_squared_error(
                np.log(y_test), predictions, squared=False
            )
            R2 = sklearn.metrics.r2_score(np.log(y_test), predictions)
            RMSES.append(RMSE)
            R2S.append(R2)

            plot_is = iS[: len(R2S)]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        ax1.set_xlim(0, max(iS) + 100)
        ax1.plot(plot_is, R2S)
        ax1.set_xlabel("min_samples_leaf")
        ax1.set_ylabel("R2")
        start, end = ax1.get_xlim()
        ax1.xaxis.set_ticks(np.arange(start, end, 500))
        ax1.grid(axis="x")

        ax2.set_xlim(0, max(iS) + 100)
        ax2.plot(plot_is, RMSES)
        ax2.set_xlabel("min_samples_leaf")
        ax2.set_ylabel("RMSE")
        start, end = ax2.get_xlim()
        ax2.xaxis.set_ticks(np.arange(start, end, 500))
        ax2.grid(axis="x")
        plt.savefig("plots/rs4_estimators/Decision_tree_min_samples_leaf.png", dpi=300)
        plt.clf()

    # Decision tree max depth
    # https://analyticsindiamag.com/guide-to-hyperparameters-tuning-using-gridsearchcv-and-randomizedsearchcv/
    # https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680
    if max_depth:
        RMSES = []
        R2S = []
        iS = range(1, 30)
        for depth in iS:
            print(f"\rTrying depth {depth} with max {max(iS)}", end=" " * 8)
            model = DecisionTreeRegressor(max_depth=depth).fit(X_train, np.log(y_train))
            predictions = model.predict(X_test)
            RMSE = sklearn.metrics.mean_squared_error(
                np.log(y_test), predictions, squared=False
            )
            R2 = sklearn.metrics.r2_score(np.log(y_test), predictions)
            RMSES.append(RMSE)
            R2S.append(R2)

            plot_is = iS[: len(R2S)]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        ax1.plot(plot_is, R2S)
        ax1.set_xlabel("max_depth")
        ax1.set_ylabel("R2")
        start, end = ax1.get_xlim()
        ax1.xaxis.set_ticks(np.arange(start, end, 5))
        ax1.grid(axis="x")

        ax2.plot(plot_is, RMSES)
        ax2.set_xlabel("max_depth")
        ax2.set_ylabel("RMSE")
        start, end = ax2.get_xlim()
        ax2.xaxis.set_ticks(np.arange(start, end, 5))
        ax2.grid(axis="x")
        plt.savefig("plots/rs4_estimators/Decision_tree_depth.png", dpi=300)
        plt.clf()

    # Decision tree min samples split
    if min_samples_split:
        RMSES = []
        R2S = []
        iS = range(100, 10000, 100)
        for min_samples in iS:
            print(
                f"\rTrying min samples split {min_samples} with max {max(iS)}",
                end=" " * 8,
            )
            model = DecisionTreeRegressor(min_samples_split=min_samples).fit(
                X_train, np.log(y_train)
            )
            predictions = model.predict(X_test)
            RMSE = sklearn.metrics.mean_squared_error(
                np.log(y_test), predictions, squared=False
            )
            R2 = sklearn.metrics.r2_score(np.log(y_test), predictions)
            RMSES.append(RMSE)
            R2S.append(R2)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        ax1.set_xlim(0, max(iS) + 100)
        ax1.plot(iS, R2S)
        ax1.set_xlabel("min_samples_split")
        ax1.set_ylabel("R2")
        start, end = ax1.get_xlim()
        ax1.xaxis.set_ticks(np.arange(start, end, 500))
        ax1.grid(axis="x")

        ax2.set_xlim(0, max(iS) + 100)
        ax2.plot(iS, RMSES)
        ax2.set_xlabel("min_samples_split")
        ax2.set_ylabel("RMSE")
        start, end = ax2.get_xlim()
        ax2.xaxis.set_ticks(np.arange(start, end, 500))
        ax2.grid(axis="x")

        plt.savefig("plots/rs4_estimators/Decision_tree_min_samples_split.png", dpi=300)
        plt.clf()

    if gridsearch:
        # Decision tree grid search
        param_grid = {
            "criterion": ["squared_error", "absolute_error"],
            "splitter": ["best", "random"],
            "min_samples_split": [2, 4, 10, 20, 50, 100, 500, 1000],
            "min_samples_leaf": [2, 4, 10, 20, 50, 100, 500, 1000],
            "max_depth": [None, 5],
            "random_state": [RANDOM_STATE],
            "max_features": [0.1, 0.25, 0.5, 0.75, 1.0],
        }
        model = GridSearchCV(
            estimator=DecisionTreeRegressor(), param_grid=param_grid, cv=10, verbose=3
        ).fit(X_train, np.log(y_train))
        print(model.best_params)
