from GLOBAL_VARS import BOOTSTRAP_OBS, BOOTSTRAP_B, LOG, RANDOM_STATE

# Normal imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.manifold import Isomap
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
    # - KNN with 20 neigbors 
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
    
    """
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
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(28, 6))

        plot_is = iS[: len(R2S)]
        ax1.plot(plot_is, R2S)
        ax1.set_xlabel("max_depth")
        ax1.set_ylabel("R2")

        ax2.plot(plot_is, RMSES)
        ax2.set_xlabel("max_depth")
        ax2.set_ylabel("RMSE")
        plt.savefig("plots/rs4_estimators/Decision_tree_depth.png")
        plt.clf()
    """

    # sample neighbors
    """
    RMSES = []
    R2S = []    
    iS = range(1, 15)
    for i, n in enumerate(iS):
        print(f"\rTrying {i+1} neighbors, max of {max(iS)}", end=" " * 8)
        model = KNeighborsRegressor(n_neighbors=n).fit(X_train, np.log(y_train))
        predictions = model.predict(X_test)
        RMSE = sklearn.metrics.mean_squared_error(
            np.log(y_test), predictions, squared=False
        )
        R2 = sklearn.metrics.r2_score(np.log(y_test), predictions)
        RMSES.append(RMSE)
        R2S.append(R2)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(28, 6))

        plot_is = iS[:len(R2S)]
        ax1.plot(plot_is, R2S)
        ax1.set_xlabel("n_neighbors")
        ax1.set_ylabel("R2")

        ax2.plot(plot_is, RMSES)
        ax2.set_xlabel("n_neighbors")
        ax2.set_ylabel("RMSE")
        plt.savefig("plots/rs4_estimators/knn.png")
        plt.clf()
    """

    # Decision tree grid search
    param_grid = {
        "criterion": ["squared_error", "absolute_error"],
        "splitter": ["best", "random"],
        "min_samples_split": [],
        "max_depth": [
            None, 5
        ],
        "random_state": [RANDOM_STATE],
    }
