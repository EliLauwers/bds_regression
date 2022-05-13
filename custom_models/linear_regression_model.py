import numpy as np

import GLOBAL_VARS
from GLOBAL_VARS import LOG, BOOTSTRAP_B, BOOTSTRAP_OBS,RANDOM_STATE
from helpers.calculate_optimal_B import calculate_optimal_B
import statsmodels.api as sm
import helpers.evaluative_metrics as eval
import random
import multiprocessing as mp
import pickle

from custom_models import linear_regression_model as linreg


def vanilla_linreg(y_train, X_train, X_test):
    """
    Used as baseline
    """
    i, predictions = fit_and_predict(0, y_train.index, y_train, X_train, X_test)
    return predictions


def bootstrap_linreg(y_train, X_train, X_test):
    """
    This function opens the dataset and will predict by a bootstrapped method. This function is the main function for the linear models
    """
    LOG.process(f"Multiprocess Start: {BOOTSTRAP_B} iterations")

    # https://www.machinelearningplus.com/python/parallel-processing-python/
    pool = mp.Pool(mp.cpu_count() - 2)
    indexes_lst = []
    for i in range(BOOTSTRAP_B):
        random.seed(RANDOM_STATE + i)
        indexes_lst.append(random.choices(y_train.index, k=BOOTSTRAP_OBS))
    result_objects = [
        pool.apply_async(
            fit_and_predict, args=(i, indexes_lst[i], y_train, X_train, X_test)
        )
        for i in range(BOOTSTRAP_B)
    ]
    predictions = np.empty(shape=[BOOTSTRAP_B, len(X_test)])
    for obj in result_objects:
        i, results = obj.get()
        predictions[i] = results
    LOG.process("Closing Pool")
    pool.close()
    pool.join()
    # calculate_optimal_B(BOOTSTRAP_B, predictions, y_true)
    return predictions


def fit_and_predict(i, bootstrap_indexes, y_train, X_train, X_test):
    """
    subfunction: this function generates predictions
    """
    y_train_bootstrap = y_train.loc[bootstrap_indexes]
    X_train_bootstrap = X_train.loc[bootstrap_indexes]
    model = sm.OLS(y_train_bootstrap, X_train_bootstrap).fit()
    # https://stats.stackexchange.com/questions/55692/back-transformation-of-an-mlr-model
    smearing_factor = len(y_train) ** -1 * np.sum(np.exp(model.resid)) # see link
    preds = model.predict(X_test)
    preds = np.array(preds, dtype = np.ulonglong)
    predictions = np.exp(preds) * smearing_factor
    return (i, predictions)
