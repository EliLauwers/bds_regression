# Custom scripts
from giant_steps.create_dataset import create_dataset
from giant_steps.pre_process import pre_process

# Custom functions
from custom_models import linear_regression_model as linreg
from helpers import evaluative_metrics as eval_metrics
from helpers.plot_linear_regression import plot_linear_regression
from helpers.calculate_optimal_B import calculate_optimal_B
from GLOBAL_VARS import BOOTSTRAP_OBS, BOOTSTRAP_B, LOG, RANDOM_STATE

# Normal imports
import statsmodels.api as sm
import numpy as np

np.random.seed(RANDOM_STATE)

import pickle
import random


if __name__ == "__main__":
    run_full = False
    if run_full:
        create_dataset()
        pre_process()
        linreg.bootstrap_linreg()

    LOG.process("Read Data before predictions")
    with open("data/intermediate/pre_processed/y_train.pk", "rb") as infile:
        y_true = pickle.load(infile)
    with open("data/intermediate/pre_processed/X_train.pk", "rb") as infile:
        X_train = pickle.load(infile)
        X_train = sm.add_constant(X_train)

    models = [
        ("Vanilla OLS", linreg.vanilla_linreg),
        ("Bootstrapped OLS", linreg.bootstrap_linreg),
    ]

    agg_funcs = [("mean", np.mean), ("median", np.median)]

    for model_name, predictions_generator in models:
        preds = predictions_generator()
        print(preds.shape)

        for agg_name, agg_func in agg_funcs:
            agg_predictions = agg_func(preds, axis=0)
            meta = {"model_name": model_name, "agg_name": agg_name}
            LOG.evaluate_predictions(meta, preds, y_true)
