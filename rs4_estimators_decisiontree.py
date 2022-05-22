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

    # Decision tree max depth
    # https://analyticsindiamag.com/guide-to-hyperparameters-tuning-using-gridsearchcv-and-randomizedsearchcv/
    # https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680

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
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30,30))

        plot_is = iS[: len(R2S)]
        ax1.plot(plot_is, R2S)
        ax1.set_xlabel("max_depth")
        ax1.set_ylabel("R2")

        ax2.plot(plot_is, RMSES)
        ax2.set_xlabel("max_depth")
        ax2.set_ylabel("RMSE")
        plt.savefig("plots/rs4_estimators/Decision_tree_depth.png")
        plt.clf()


    # Decision tree min samples split

    RMSES = []
    R2S = []
    iS = range(100, 10000, 500)
    for min_samples in iS:
        print(f"\rTrying min samples {min_samples} with max {max(iS)}", end=" " * 8)
        model = DecisionTreeRegressor(min_samples_split=min_samples).fit(X_train, np.log(y_train))
        predictions = model.predict(X_test)
        RMSE = sklearn.metrics.mean_squared_error(
            np.log(y_test), predictions, squared=False
        )
        R2 = sklearn.metrics.r2_score(np.log(y_test), predictions)
        RMSES.append(RMSE)
        R2S.append(R2)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30,30))

        plot_is = iS[: len(R2S)]
        ax1.plot(plot_is, R2S)
        ax1.set_xlabel("min_samples_split")
        ax1.set_ylabel("R2")

        ax2.plot(plot_is, RMSES)
        ax2.set_xlabel("min_samples_split")
        ax2.set_ylabel("RMSE")
        plt.savefig("plots/rs4_estimators/Decision_tree_min_samples_split.png")
        plt.clf()
    xxx

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
