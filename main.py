# Custom scripts
from statistics import LinearRegression
from giant_steps.create_dataset import create_dataset
from giant_steps.pre_process import (
    pre_process_album_date_released,
    pre_process_track_listens,
)

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
import numpy as np

np.random.seed(RANDOM_STATE)

import pickle
import random

random.seed(RANDOM_STATE)

if __name__ == "__main__":

    run_all = False

    if run_all:
        create_dataset()
        pre_process_track_listens()

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

    estimators = [
        {"name": "Ordinary Least Squares", "func": LinearRegression()},
        {
            "name": "Decision Tree",
            "func": DecisionTreeRegressor(),
        },
        {
            "name": "KNN_n3",
            "func": KNeighborsRegressor(n_neighbors=3),
        },
        {"name": "Random Forest", "func": RandomForestRegressor()},
    ]

    datareds = [None, "ipca", "mutual_information", "isomap"]
    datareds = ["mutual_information", "isomap"]

    params = [
        (d, e, b, s)  # adding a params tuple
        for d in datareds  # data reduction techniques
        for e in estimators  # estimators
        for b in [True, False]  # bootstrap
        for s in [True, False]  # use_smearing
    ]
    params = [
        (d, e, b, s)  # adding a params tuple
        for d in datareds  # data reduction techniques
        for e in estimators  # estimators
        for b in [True]  # bootstrap
        for s in [True]  # use_smearing
    ]
    # Loop over estimators to compare

    for i, row in enumerate(params):
        try:
            # unpack the params row
            datared, estimator, bootstrap, use_smearing = row
            # further unpack the estimator value
            estimator_name, estimator_func = estimator.values()

            X_train_used = X_train
            X_test_used = X_test

            if datared == "ipca":

                reducer = IncrementalPCA(n_components=158).fit(X_train)
                X_train_used = reducer.transform(X_train_used)
                X_test_used = reducer.transform(X_test_used)

            elif datared == "mutual_information":

                mi = mutual_info_regression(
                    X_train_used,
                    np.log(y_train),
                    n_neighbors=5,
                    random_state=RANDOM_STATE,
                )
                mi_logvec = np.where(mi >= 0.005)
                X_train_used = X_train_used.iloc[:, mi_logvec[0]]
                X_test_used = X_test_used.iloc[:, mi_logvec[0]]

            elif datared == "isomap":
                reducer = Isomap(n_neighbors=5).fit(X_train_used)
                X_train_used[:] = reducer.transform(X_train_used)
                X_test_used[:] = reducer.transform(X_test_used)

            if bootstrap:
                estimator = BaggingRegressor(
                    estimator,
                    n_estimators=BOOTSTRAP_B,
                    max_samples=BOOTSTRAP_OBS,
                    random_state=RANDOM_STATE,
                )

            if use_smearing:
                model = estimator_func.fit(X_train, np.log(y_train))
                train_predictions = model.predict(X_train)
                resids = np.log(y_train) - train_predictions
                smearing_raw = np.exp(resids)
                # Some predictions will be infinite due to high exp
                if np.isinf(smearing_raw).any():
                    smearing_raw[np.where(np.isinf(smearing_raw))] = max(y_train)
                y_predictions = model.predict(X_test) * np.mean(smearing_raw)
                # Some predictions will be infinite due to high exp
                if np.isinf(y_predictions).any():
                    y_predictions[np.where(np.isinf(y_predictions))] = max(y_train)
            else:
                model = estimator_func.fit(X_train, y_train)
                y_predictions = model.predict(X_test)

            meta = {
                "data reduction": datared,
                "estimator": estimator_name,
                "bootstrap": bootstrap,
                "smearing": use_smearing,
            }

            LOG.evaluate_predictions(meta, y_predictions, y_test, i + 1, len(params))
        except Exception as e:
            print("\n\nSomething went wrong\n\n")
            print(e)
            print(row)
            print("\n\n")
