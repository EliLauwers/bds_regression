from matplotlib import pyplot as plt
import random
import pickle
from GLOBAL_VARS import NUM_CORES, RANDOM_STATE, FHEIGHT, FWIDTH

# Normal imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import sklearn
import joblib
import pandas as pd
from mpld3 import save_html


np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

if __name__ == "__main__":
    # Conclusions
    

    input_path = "data/intermediate/track_listens/"

    with open(input_path + "X_train.pk", "rb") as infile:
        X_train = pickle.load(infile)

    with open(input_path + "y_train.pk", "rb") as infile:
        y_train = pickle.load(infile)

    # Scale data with minmax scaler
    minmaxscaler = MinMaxScaler().fit(X_train)
    X_train[:] = minmaxscaler.transform(X_train)

    compute_random_forest = False

    if compute_random_forest:

        param_grid = {
            "n_estimators": [50, 100, 150, 200],            
        }

        model = GridSearchCV(
            estimator=RandomForestRegressor(
                criterion="squared_error",
                max_depth = None,
                max_features = .5,
                min_samples_split=1300,
                min_samples_leaf=700,
                random_state=RANDOM_STATE
            ),
            param_grid=param_grid,
            cv=10,
            verbose=3,
            n_jobs=NUM_CORES,
            scoring=[
                "neg_root_mean_squared_error",
                "neg_median_absolute_error",
                "r2",
            ],
            refit=False
        ).fit(X_train, np.log(y_train))

        joblib.dump(
            model, "logs/rs4_estimators/rf_full_model.pkl"
        )

    with open("logs/rs4_estimators/rf_full_model.pkl", "rb") as infile:
        results = pd.DataFrame(joblib.load(infile).cv_results_)
    fig, axes = plt.subplots(3, 1, figsize=(
        FHEIGHT * 3, FWIDTH * 2), sharex=False)

    metric_cols = [
        "test_r2",
        "test_neg_root_mean_squared_error",
        "test_neg_median_absolute_error",
    ]
    metric_names = ["R2", "Neg RMSE", "Neg MAE"]
    collection = zip(axes, metric_cols, metric_names)
    for i, (ax, col, name) in enumerate(collection):
        x_label = "n_estimators"
        x_param = "param_" + x_label

        ax.set_xlabel("")
        if i == 0:
            ax.set_title("Evaluative metrics Random Forest")
        elif i == 2:
            ax.set_xlabel(x_label)

        interval = 50
    

        mean_ = results["mean_" + col]
        std_ = results["std_" + col]

        # TODO color every line
        ax.errorbar(
            results[x_param],
            mean_,
            yerr=std_,
            ecolor="r",
            elinewidth=1,
            capsize=2.5,
        )
        ax.set_ylabel(name)
        ax.set_xlim(0, 200 + interval)
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, interval))

    plt.savefig(f"plots/rs4_estimators/rf_full_model.png", dpi=300)
    save_html(fig, f"plots/rs4_estimators/rf_full_model.html")
    plt.clf()
