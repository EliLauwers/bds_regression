import os
import pickle
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import ElasticNetCV
from sklearn.feature_selection import SelectFromModel
from GLOBAL_VARS import RANDOM_STATE, NUM_CORES, LOG
import numpy as np

reductors = [
    (
        "lle",
        LocallyLinearEmbedding(
            n_neighbors = 16,
            eigen_solver="dense",
            method="modified",
            random_state=RANDOM_STATE,
            n_jobs=NUM_CORES,
        ),
    ),
    (
        "elasticnet",
        SelectFromModel(
            ElasticNetCV(
                max_iter=10000,
                random_state=RANDOM_STATE,
                selection="random",
                l1_ratio=1,
            )
        ),
    ),
    ("pca99", PCA(n_components=0.99)),
    ("pca95", PCA(n_components=0.95)),
]


def reduct_datasets():
    LOG.process("Reducing datasets ...")
    base_input = "data/intermediate"
    for var in ["track_listens", "album_date_released"]:
        # Read in cleaned data
        input_path = os.path.join(base_input, var, "X_train.pk")
        with open(input_path, "rb") as infile:
            X_train = pickle.load(infile)
        input_path = os.path.join(base_input, var, "X_test.pk")
        with open(input_path, "rb") as infile:
            X_test = pickle.load(infile)
        input_path = os.path.join(base_input, var, "y_train.pk")
        with open(input_path, "rb") as infile:
            y_train = pickle.load(infile)
        # Scale the data
        scaler = MinMaxScaler().fit(X_train)
        X_train[:] = scaler.transform(X_train)
        X_test[:] = scaler.transform(X_test)
        # Write data without any reduction
        out_path = os.path.join("data", "reducted", var, "no_datared")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with open(out_path + "/" + "X_train.pk", "wb") as outfile:
            pickle.dump(X_train, outfile, protocol=pickle.HIGHEST_PROTOCOL)

        with open(out_path + "/" + "X_test.pk", "wb") as outfile:
            pickle.dump(X_test, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        # Iteratively fit a reduction technique and store results
        for name, reductor in reductors:
            # Create reducted dataset
            if name == "elasticnet" and var == "track_listens":
                red = reductor.fit(X_train, np.log(y_train))
            elif name == "elasticnet" and var == "album_date_released":
                red = reductor.fit(X_train, np.power(y_train, 2))
            elif name == "lle":
                red = reductor.fit(X_train.sample(frac=0.25, random_state=RANDOM_STATE))
            else:
                red = reductor.fit(X_train)

            train_used = red.transform(X_train)
            test_used = red.transform(X_test)

            LOG.process(
                f"no. cols of X_train for {var} after {name}: {train_used.shape[1]}"
            )
            # Store results
            out_path = os.path.join("data", "reducted", var, name)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            with open(out_path + "/" + "X_train.pk", "wb") as outfile:
                pickle.dump(train_used, outfile, protocol=pickle.HIGHEST_PROTOCOL)

            with open(out_path + "/" + "X_test.pk", "wb") as outfile:
                pickle.dump(test_used, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    LOG.process("Reducing datasets done!")


if __name__ == "__main__":
    reduct_datasets()
