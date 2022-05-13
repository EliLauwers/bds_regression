import pickle
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from matplotlib import pyplot as plt
import json
import os
import pandas as pd 

def vanilla_knn():
    LOG.process("Vanilla knn")
    with open("data/intermediate/pre_processed/y_train.pk", "rb") as infile:
        y_true = pickle.load(infile)
    with open("data/intermediate/pre_processed/X_train.pk", "rb") as infile:
        X_train = pickle.load(infile)
        X_train = sm.add_constant(X_train)
    i, predictions = fit_and_predict(0, y_true.index, y_true, X_train)
    return predictions


def fit_and_predict(i, bootstrap_indexes, y_true, X_train):
    """
    subfunction: this function generates predictions
    """
    y_true_bootstrap = y_true.loc[bootstrap_indexes]
    X_train_bootstrap = X_train.loc[bootstrap_indexes]

    model = KNeighborsRegressor(n_neighbors=2).fit()
    predictions = model.predict(X_train)

    return (i, predictions)


def calculate_best_neighbors(y_train, X_train):
    max_neighbors = len(y_train)
    neighs = range(1, max_neighbors)
    bias_lst = []
    SE_lst = []
    R2_lst = []
    RMSEP_lst = []
    path = "logs/track_listens/knn_metrics.json"

    if os.path.exists(path):

        with open(path, "r+") as file:
            cur_file = json.load(file)
        max_in_file = max([el["i"] for el in cur_file]) + 1
    else:
        max_in_file = 1
        with open(path, "w") as file:
            file.write(json.dumps([]))
    for i in range(max_in_file, max_neighbors):
        print(f"trying {i} neighbors")
        model = KNeighborsRegressor(n_neighbors=i).fit(X_train, y_train)
        predictions = model.predict(X_train)

        SSE = np.sum(np.square(y_train - predictions))
        RMSEP = np.sqrt(1 / (len(y_train) - 1) * SSE)
        R2 = np.square(np.corrcoef(predictions, y_train))[0, 1]
        preds_centered = predictions - y_train
        bias = np.mean(preds_centered)
        SE = np.std(preds_centered)

        bias_lst.append(bias)
        SE_lst.append(SE)
        R2_lst.append(R2)
        RMSEP_lst.append(RMSEP)

        eval_dict = {"i": i, "RMSEP": RMSEP, "R2": R2, "bias": bias, "SE": SE}

        with open(path, "r+") as file:
            cur_file = json.load(file)

        with open(path, "w") as file:
            cur_file.append(eval_dict)
            file.write(json.dumps(cur_file))

def calculate_optimal_b_2():
    path = "logs/track_listens/knn_metrics.json"
    df = pd.read_json(path).set_index("i")
    fig, axes = plt.subplots(2, 2, figsize = (30, 12))
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            ind = (i * 2) + j
            colname = df.columns[ind]
            ax.plot(df.index, df[colname], label = colname)
            ax.legend()
            ax.grid(linestyle = "--", linewidth = .3)
            ax.set_xticks(range(0, max(df.index), 5))

    plt.savefig("plots/track_listens/optimalneigh.png")
    plt.clf()
    
    df = pd.read_json(path).set_index("i")
    df = df.iloc[df.index <= 11,:]
    print(df.head())
    fig, axes = plt.subplots(2, 2, figsize = (30, 12))
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            ind = (i * 2) + j
            colname = df.columns[ind]
            ax.plot(df.index, df[colname], label = colname)
            ax.legend()
            ax.grid(linestyle = "--", linewidth = .3)
            ax.set_xticks(range(0, max(df.index), 1))

    plt.savefig("plots/track_listens/optimalneigh_2.png")
    plt.clf()