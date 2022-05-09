import numpy as np
from GLOBAL_VARS import LOG, BOOTSTRAP_B, BOOTSTRAP_OBS
from helpers.calculate_optimal_B import calculate_optimal_B
import statsmodels.api as sm
import helpers.evaluative_metrics as eval
import random
import multiprocessing as mp
import pickle

from custom_models import linear_regression_model as linreg


def vanilla_linreg():
    """
    Used as baseline
    """
    LOG.process("Vanilla Linear Regression")
    with open("data/intermediate/pre_processed/y_train.pk", "rb") as infile:
        y_true = pickle.load(infile)
    with open("data/intermediate/pre_processed/X_train.pk", "rb") as infile:
        X_train = pickle.load(infile)
        X_train = sm.add_constant(X_train)
    i, predictions = fit_and_predict(0, y_true.index, y_true, X_train)
    return predictions


def bootstrap_linreg():
    """
    This function opens the dataset and will predict by a bootstrapped method. This function is the main function for the linear models
    """
    LOG.process("Read Data before predictions")
    with open("data/intermediate/pre_processed/y_train.pk", "rb") as infile:
        y_true = pickle.load(infile)
    with open("data/intermediate/pre_processed/X_train.pk", "rb") as infile:
        X_train = pickle.load(infile)
        X_train = sm.add_constant(X_train)

    LOG.process(f"Multiprocess Start: {BOOTSTRAP_B} iterations")

    # https://www.machinelearningplus.com/python/parallel-processing-python/
    pool = mp.Pool(mp.cpu_count() - 2)
    indexes_lst = [
        random.choices(y_true.index, k=BOOTSTRAP_OBS) for i in range(BOOTSTRAP_B)
    ]
    result_objects = [
        pool.apply_async(fit_and_predict, args=(i, indexes_lst[i], y_true, X_train))
        for i in range(BOOTSTRAP_B)
    ]
    predictions = np.empty(shape=[BOOTSTRAP_B, len(y_true)])
    for obj in result_objects:
        i, results = obj.get()
        predictions[i] = results.to_numpy()
    LOG.process("Closing Pool")
    pool.close()
    pool.join()
    LOG.process("Predictions")
    # calculate_optimal_B(BOOTSTRAP_B, predictions, y_true)
    return predictions


def fit_and_predict(i, bootstrap_indexes, y_true, X_train):
    """
    subfunction: this function generates predictions
    """
    y_true_bootstrap = y_true.loc[bootstrap_indexes]
    X_train_bootstrap = X_train.loc[bootstrap_indexes]
    model = sm.OLS(y_true_bootstrap, X_train_bootstrap).fit()
    predictions = model.predict(X_train)
    return (i, predictions)


def evaluate(model):
    """
    This function can be used to evaluate a linear model. The function is quite badly written
    """
    # Evaluate the model
    y_pred = model.fittedvalues
    y_true = model.model.endog
    model_metrics = {}
    model_metrics["R2"] = model.rsquared
    model_metrics["R2_adj"] = model.rsquared_adj
    model_metrics["AIC"] = model.aic
    model_metrics["BIC"] = model.bic

    N, K = len(y_pred), len(model.params)  # number of instances, predictors
    SST = model.centered_tss
    SSE = model.ssr  # SS residual
    SSR = model.ess  # SS explained => SS regression => SS model
    MSE = SSE / (N - K - 1)
    MSR = SSR / K
    model_metrics["mallows_cp"] = eval.mallows_cp(SSE, MSE, N, K)
    model_metrics["RMSE"] = np.sqrt(MSE)
    model_metrics["RMSEA"] = None
    LOG.model_evaluates({"meta": "Simple linear model", "metrics": model_metrics})
    return model
