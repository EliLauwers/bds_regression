import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns


# Note: Using https://realpython.com/linear-regression-in-python/
X_train = pd.read_csv("data/intermediate/pre_processed/X_train.csv", index_col = "track_id").to_numpy()
y_train = pd.read_csv("data/intermediate/pre_processed/y_train.csv", index_col = "track_id").to_numpy()
transformer = PolynomialFeatures(degree=3, include_bias=False)
transformer.fit(X_train)

X_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_train)
model = LinearRegression().fit(X_, y_train)
y_train_exp = np.exp(y_train)
preds = model.predict(X_train)
preds_exp = np.exp(preds)

fig, axes = plt.subplots(2, 2)
fig.set_size_inches([12,6])
sns.regplot(y_train, preds, ax = axes[0][0], scatter_kws = {"s" : 1})
axes[0][0].set(xlabel="track_listens_log")
sns.residplot(y_train, preds, ax = axes[0][1], scatter_kws = {"s" : 1})
axes[0][1].set(xlabel="track_listens_log")
sns.regplot(y_train_exp, preds_exp, ax = axes[1][0], scatter_kws = {"s" : 1})
axes[1][0].set(xlabel="track_listens")
sns.residplot(y_train_exp, preds_exp, ax = axes[1][1], scatter_kws = {"s" : 1})
axes[1][1].set(xlabel="track_listens")
for row in axes:
    for ax in row:
        ax.set(ylabel="predictions")
plt.show()

xt = pd.read_csv("data/intermediate/pre_processed/X_train.csv", index_col = "track_id")
xt_corr = xt.corr()
unique, counts = np.unique(xt_corr.to_numpy(), return_counts = True)
counts = counts / 2

df = pd.DataFrame({"element":unique,"count":counts})
plt.hist(abs(df.element), weights = df['count'],bins = 50)
sum(abs(df.element)>=.8) / len(df.element)