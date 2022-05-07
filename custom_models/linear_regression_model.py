import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


# Note: Using https://realpython.com/linear-regression-in-python/
X_train = pd.read_csv("../data/intermediate/pre_processed/X_train.csv", index_col ="track_id").to_numpy()
y_train = pd.read_csv("../data/intermediate/pre_processed/y_train.csv", index_col ="track_id").to_numpy()
model = LinearRegression().fit(X_train, y_train)

y_train_exp = np.exp(y_train)
preds = model.predict(X_train)
preds_exp = np.exp(preds)

scatter_points = {"s" : .1, "alpha" : .3}
fig, axes = plt.subplots(2, 2)
fig.set_size_inches([12,6])
sns.regplot(y_train, preds, ax = axes[0][0], scatter_kws = scatter_points, lowess = True)
axes[0][0].set(xlabel="track_listens_log")
sns.residplot(y_train, preds, ax = axes[0][1], scatter_kws = scatter_points)
axes[0][1].set(xlabel="track_listens_log")
sns.regplot(y_train_exp, preds_exp, ax = axes[1][0], scatter_kws = scatter_points)
axes[1][0].set(xlabel="track_listens")
sns.residplot(y_train_exp, preds_exp, ax = axes[1][1], scatter_kws = scatter_points)
axes[1][1].set(xlabel="track_listens")
for row in axes:
    for ax in row:
        ax.set(ylabel="predictions")
plt.show()