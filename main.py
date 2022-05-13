# Custom scripts
from statistics import LinearRegression
from giant_steps.create_dataset import create_dataset
from giant_steps.pre_process import (
    pre_process_album_date_released,
    pre_process_track_listens,
)


from GLOBAL_VARS import BOOTSTRAP_OBS, BOOTSTRAP_B, LOG, RANDOM_STATE

# Normal imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import linearRegression
from sklearn.tree import DecisionTreeRegressor, KNeighborsRegressor

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

    estimators = [
        {"name": "OLS", "func": LinearRegression()},
        {
            "name": "Decision Tree",
            "func": DecisionTreeRegressor(),
        },
        {
            "name": "KNN_n3",
            "func": KNeighborsRegressor(n_neighbors=3),
        },
        {
            "name": "RandomForest",
            "func" : RandomForestRegressor
        }
    ]

    datareds = ["no_datared"]
    scalars = [(StandardScaler, "standard"), (MinMaxScaler, "minmax"), (None, "None")]
    true_false = [True, False]
    params = []
    for datared in datareds:
        for scaler, scaler_name in scalars:
            for estimator in estimators:
                for bootstrap in true_false:
                    for smearing in true_false:
                        params.append(
                            (
                                datared,
                                (scaler, scaler_name),
                                estimator,
                                bootstrap,
                                smearing,
                            )
                        )

    # Loop over estimators to compare

    for row in params:
        datared = row[0]
        scaler, scaler_name = row[1]
        estimator_name, estimator_func = row[2].values()
        bootstrap = row[3]
        use_smearing = row[4]

        with open(
            f"data/intermediate/track_listens/{datared}/y_train.pk", "rb"
        ) as infile:
            y_train = pickle.load(infile).to_numpy()

        with open(
            f"data/intermediate/track_listens/{datared}/X_train.pk", "rb"
        ) as infile:
            X_train = pickle.load(infile)

        with open(
            f"data/intermediate/track_listens/{datared}/y_test.pk", "rb"
        ) as infile:
            y_test = pickle.load(infile).to_numpy()

        with open(
            f"data/intermediate/track_listens/{datared}/X_test.pk", "rb"
        ) as infile:
            X_test = pickle.load(infile)

        if scaler:
            scale_model = scaler().fit(X_train)
            X_train = scale_model.transform(X_train)
            X_test = scale_model.transform(X_test)

        if bootstrap:
            estimator = BaggingRegressor(
                estimator,
                n_estimators=BOOTSTRAP_B,
                max_samples=BOOTSTRAP_OBS,
                random_state=RANDOM_STATE,
            )

        if use_smearing:
            model = estimator_func.fit(X_train, np.log(y_train))
            resids = np.log(y_train) - model.predict(X_train)
            smearing_raw = np.exp(resids)
            # Some predictions will be infinite due to high exp
            smearing_raw[np.where(np.isinf(smearing_raw))] = max(y_train)
            y_predictions = model.predict(X_test) * np.mean(smearing_raw)
            # Some predictions will be infinite due to high exp
            y_predictions[np.where(np.isinf(y_predictions))] = max(y_train)
        else:
            model = estimator_func.fit(X_train, y_train)
            y_predictions = model.predict(X_test)

        meta = {
            "estimator": estimator_name,
            "bootstrap": bootstrap,
            "scaler": scaler_name,
            "smearing": use_smearing,
        }

        LOG.evaluate_predictions(meta, y_predictions, y_test)
