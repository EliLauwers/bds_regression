from GLOBAL_VARS import RANDOM_STATE

# Normal imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import sklearn
from mpld3 import save_html

import pickle
import random
import joblib

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

from matplotlib import pyplot as plt

if __name__ == "__main__":
    compute_knn = False
    input_path = "data/intermediate/track_listens/"

    with open(input_path + "X_train.pk", "rb") as infile:
        X_train = pickle.load(infile)

    with open(input_path + "y_train.pk", "rb") as infile:
        y_train = pickle.load(infile)

    # Scale data with minmax scaler
    minmaxscaler = MinMaxScaler().fit(X_train)
    X_train[:] = minmaxscaler.transform(X_train)

    if compute_knn:
        param_grid = {"n_neighbors": range(1, 31)}
        model = GridSearchCV(
            estimator=KNeighborsRegressor(),
            param_grid=param_grid,
            cv=HYPERCV,
            verbose=3,
            scoring=["neg_root_mean_squared_error", "neg_median_absolute_error", "r2"],
            refit=False,
        ).fit(X_train, np.log(y_train))

        joblib.dump(model.cv_results_, "logs/rs4_estimators/knn.pkl")

    # open results
    results = pd.DataFrame(joblib.load("logs/rs4_estimators/knn.pkl"))

    # start plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 16), sharex=True)

    metric_cols = [
        "test_r2",
        "test_neg_root_mean_squared_error",
        "test_neg_median_absolute_error",
    ]
    metric_names = ["R2", "Neg RMSE", "Neg MAE"]
    collection = zip(axes, metric_cols, metric_names)
    for i, (ax, col, name) in enumerate(collection):

        x_label = "n_neighbors"
        x_param = "param_" + x_label

        ax.set_xlabel("")
        if i == 0:
            ax.set_title("Evaluative Measures for KNN")
        elif i == 2:
            ax.set_xlabel(x_label)            
        
        interval = 2
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
