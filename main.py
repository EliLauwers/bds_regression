# Custom scripts
from statistics import LinearRegression
from helpers.create_dataset import create_dataset
from rs1_preprocess import (
    pre_process_album_date_released,
    pre_process_track_listens,
)
from rs2_reduct_datasets import reduct_datasets

from GLOBAL_VARS import BOOTSTRAP_OBS, BOOTSTRAP_B, LOG, NUM_CORES, RANDOM_STATE

# Normal imports
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import os

np.random.seed(RANDOM_STATE)

import pickle
import random

random.seed(RANDOM_STATE)

def group_predictions(preds, tids, conversion):
    preds = pd.DataFrame({"preds":preds,"track_id":tids}).set_index("track_id")
    preds["album_id"] = conversion.loc[preds.index,:]
    album_dates = preds.groupby("album_id")["preds"].apply(lambda albumpreds: np.median(albumpreds)).reset_index()
    preds_joined = preds.merge(album_dates, on = "album_id",how="left")
    return preds_joined["preds_y"]

if __name__ == "__main__":

    run_create_dataset = False
    run_pre_process = False
    run_reduct_datasets = False
    estimate_track_listens = True 
    estimate_album_date_released = True
    estimate_bagged_album = True

    if run_create_dataset:
        create_dataset()
    if run_pre_process:
        pre_process_track_listens()
        pre_process_album_date_released()
    if run_reduct_datasets:         
        reduct_datasets()

    estimators = [
        ("Ordinary Least Squares", LinearRegression()),
        ("Decision Tree", DecisionTreeRegressor(criterion="squared_error",splitter = "best", max_depth = None, min_samples_split=1300, min_samples_leaf = 700)),
        ("KNN", KNeighborsRegressor(n_neighbors=16)),
        ("Random Forest", RandomForestRegressor(criterion="squared_error",
                n_estimators = 50,
                max_depth = 7,
                max_features = .25,
                random_state=RANDOM_STATE)),
    ]

    datareds = ["no_datared","pca99","pca95","elasticnet","lle"]


    params = [
        (d, e, b, s)  # adding a params tuple
        for d in datareds
        for e in estimators  # estimators
        for b in [True, False]
        for s in [True, False]
    ]

    # Estimate track listens
    if estimate_track_listens:
        for i, row in enumerate(params):
            try:
                # unpack the params row
                datared, estimator, bootstrap, use_smearing = row
                # further unpack the estimator value
                estimator_name, estimator_func = estimator

                # Read in the datareducted data
                with open("data/reducted/track_listens/" + datared + "/X_train.pk", "rb") as infile:
                    X_train = pickle.load(infile)
                with open("data/reducted/track_listens/" + datared + "/X_test.pk", "rb") as infile:
                    X_test = pickle.load(infile)
                
                with open("data/intermediate/track_listens/y_train.pk", "rb") as infile:
                    y_train = pickle.load(infile)
                with open("data/intermediate/track_listens/y_test.pk", "rb") as infile:
                    y_test = pickle.load(infile)

                if bootstrap:
                    estimator = BaggingRegressor(
                        estimator,
                        n_estimators=BOOTSTRAP_B,
                        max_samples=BOOTSTRAP_OBS,
                        random_state=RANDOM_STATE,
                        n_jobs = NUM_CORES
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
                    "var": "track_listens",
                    "data reduction": datared,
                    "estimator": estimator_name,
                    "bootstrap": bootstrap,
                    "smearing": use_smearing,
                }

                LOG.evaluate_predictions(meta, y_predictions, y_test, i + 1, len(params))
            except Exception as e:
                LOG.process(e)
                LOG.process("Track listens: Following params did not work")
                LOG.process(row)
    
    if estimate_album_date_released:
        # Estimate album_date_released
        for i, row in enumerate(params):
            try:
                # unpack the params row
                datared, estimator, bootstrap, square = row
                # further unpack the estimator value
                estimator_name, estimator_func = estimator

                # Read in the datareducted data
                with open("data/reducted/album_date_released/" + datared + "/X_train.pk", "rb") as infile:
                    X_train = pickle.load(infile)
                with open("data/reducted/album_date_released/" + datared + "/X_test.pk", "rb") as infile:
                    X_test = pickle.load(infile)
                
                with open("data/intermediate/album_date_released/y_train.pk", "rb") as infile:
                    y_train = pickle.load(infile)
                with open("data/intermediate/album_date_released/y_test.pk", "rb") as infile:
                    y_test = pickle.load(infile)

                if bootstrap:
                    estimator = BaggingRegressor(
                        estimator,
                        n_estimators=BOOTSTRAP_B,
                        max_samples=BOOTSTRAP_OBS,
                        random_state=RANDOM_STATE,
                    )

                if square:
                    model = estimator_func.fit(X_train, np.power(y_train, 2))
                    preds_raw = model.predict(X_test)
                    preds_raw[np.where(preds_raw < 0)] = 0
                    y_predictions = np.sqrt(preds_raw)

                else:
                    model = estimator_func.fit(X_train, y_train)
                    y_predictions = model.predict(X_test)

                meta = {
                    "var": "album date_released",
                    "data reduction": datared,
                    "estimator": estimator_name,
                    "bootstrap": bootstrap,
                    "square": square,
                }

                LOG.evaluate_predictions(meta, y_predictions, y_test, i + 1, len(params))
            except Exception as e:
                LOG.process(e)
                LOG.process("Album dates: Following params did not work")
                LOG.process(row)
    
    if estimate_bagged_album:
        with open("data/intermediate/joined_data.pk", "rb") as infile:
            album_track = pickle.load(infile)[["album_id","track_id"]].set_index("track_id")
        
        # Estimate bagged album
        for i, row in enumerate(params):
            try:
                # unpack the params row
                datared, estimator, bootstrap, square = row
                # further unpack the estimator value
                estimator_name, estimator_func = estimator

                # Read in the datareducted data
                with open("data/reducted/album_date_released/" + datared + "/X_train.pk", "rb") as infile:
                    X_train = pickle.load(infile)
                with open("data/reducted/album_date_released/" + datared + "/X_test.pk", "rb") as infile:
                    X_test = pickle.load(infile)
                
                with open("data/intermediate/album_date_released/y_train.pk", "rb") as infile:
                    y_train = pickle.load(infile)
                with open("data/intermediate/album_date_released/y_test.pk", "rb") as infile:
                    y_test = pickle.load(infile)

                if bootstrap:
                    estimator = BaggingRegressor(
                        estimator,
                        n_estimators=BOOTSTRAP_B,
                        max_samples=BOOTSTRAP_OBS,
                        random_state=RANDOM_STATE,
                    )

                if square:
                    model = estimator_func.fit(X_train, np.power(y_train, 2))
                    preds_raw = model.predict(X_test)
                    preds_raw[np.where(preds_raw < 0)] = 0
                    y_predictions = np.sqrt(preds_raw)
                else:
                    model = estimator_func.fit(X_train, y_train)
                    y_predictions = model.predict(X_test)
                
                y_predictions=group_predictions(y_predictions, y_test.index, album_track)

                meta = {
                    "var": "album date_released bagged",
                    "data reduction": datared,
                    "estimator": estimator_name,
                    "bootstrap": bootstrap,
                    "square": square,
                }

                LOG.evaluate_predictions(meta, y_predictions, y_test, i + 1, len(params))
            except Exception as e:
                LOG.process(e)
                LOG.process("album_date released bag Following params did not work")
                LOG.process(row)
        