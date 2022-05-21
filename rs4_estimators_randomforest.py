from GLOBAL_VARS import RANDOM_STATE

# Normal imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
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

    param_grid = {
        "n_estimators": [50, 100, 150, 200],
        "criterion": ["squared_error", "absolute_error"],
        "min_samples_split": [2, 4, 10, 20, 50, 100, 500, 1000],
        "min_samples_leaf": [2, 4, 10, 20, 50, 100, 500, 1000],
        "max_depth": [None, 5],
        "random_state": [RANDOM_STATE],
        "max_features": [0.1, 0.25, 0.5, 0.75, 1.0],
    }

    model = GridSearchCV(
        estimator=RandomForestRegressor(), param_grid=param_grid, cv=10, verbose=3
    ).fit(X_train, np.log(y_train))
    print(model.best_params)
