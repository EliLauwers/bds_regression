import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import scipy

# some custom scripts to save space
from validate_date_transformation import validate_date_transformation
from detect_outlying_inds_by_iqr import detect_outlying_inds_by_iqr
from calculate_mahalanobis_distance import calculate_mahalanobis_distance


# init a random seed used in algorithms
random_state = 0

# the dataframe joined_data contains:
# (1) music_data.csv: all information
# (2) metadata.csv: track_listens, album_date_released, album_id, track_id
data = pd.read_csv("data/intermediate/joined_data.csv")

# Next, split in relevant stuff
X = data.drop(["track_listens","album_id","album_date_released"], axis = 1)
y = data["track_listens"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Note: I'll start with task 1, the track_listens
# Question: Is there any missing data?
# Answer: No
sum(np.isnan(X_train).all())
# Note: Next, I'll learn some stuff about the outcome variable
# Question: How is the outcome structured
# Answer: large skew to right with very fine tail
# Note: There will be a small number of big outliers
y_train.agg(["count","mean","median","min","max","std"])
scipy.stats.skew(y_train) # Skew > 0, skew to the right
scipy.stats.kurtosis(y_train) # Large kurtosis, fine tail

plt.hist(y_train, bins = 100)
plt.title("Histogram for track_listens in training set")
plt.xlabel("track_listens")
plt.ylabel("Frequency")
plt.savefig("plots/histogram_track_listens_train.png")
plt.clf()

tmp = pd.DataFrame({"track_listens": y_train}).\
    groupby("track_listens").\
    agg(cnt = ("track_listens","count")).\
    sort_values("track_listens").\
    reset_index(level=0)
tmp["cumsum"] = np.cumsum(tmp.cnt)
tmp["rel_cumsum"] = tmp["cumsum"] / tmp["cnt"].sum()
plt.plot(tmp["track_listens"],tmp["rel_cumsum"])
plt.title("Relative cumulative Frequency of track listens")
plt.xlabel("track_listens")
plt.ylabel("Relative Cumulative Frequency")
plt.savefig("plots/cumul_rel_freq_y_train.png")
plt.clf()
# Action: try log transforming
# Question: Are there any 0 values?
# Answer: Yeah there is 1 row
y_train[y_train == 0].index
# Action: Remove that index from y train and X train
empty_index = y_train[y_train == 0].index
y_train = y_train.drop(empty_index, axis = 0)
X_train = X_train.drop(empty_index, axis = 0)
# Action: Proceed with log transformation
y_train_log = np.log(y_train)
y_train_log.agg(["count","mean","median","min","max","std"])
scipy.stats.skew(y_train_log) # Skew > 0, skew to the right
scipy.stats.kurtosis(y_train_log) # Large kurtosis, fine tail

plt.hist(y_train_log, bins = 100)
plt.title("Histogram for log(track_listens) in training set")
plt.xlabel("log(track_listens)")
plt.ylabel("Frequency")
plt.savefig("plots/histogram_log_track_listens_train.png")
plt.clf()

tmp = pd.DataFrame({"track_listens": y_train_log}).\
    groupby("track_listens").\
    agg(cnt = ("track_listens","count")).\
    sort_values("track_listens").\
    reset_index(level=0)
tmp["cumsum"] = np.cumsum(tmp.cnt)
tmp["rel_cumsum"] = tmp["cumsum"] / tmp["cnt"].sum()
plt.plot(tmp["track_listens"],tmp["rel_cumsum"])
plt.title("Relative cumulative Frequency of log track listens")
plt.xlabel("log track_listens")
plt.ylabel("Relative Cumulative Frequency")
plt.savefig("plots/cumul_rel_freq_log_track_listens.png")
plt.clf()
# Question: How many instances must be removed using IQR?
# Answer: Nearly 99%, so that's not happening
IQR_outliers = {}
IQR_outliers["track_listens"] = detect_outlying_inds_by_iqr(y_train_log)
no_cols = len(X_train.columns)
for i, colname in enumerate(X_train.columns):
    print(f"Calculating IQR for {i} of {no_cols}")
    IQR_outliers[colname] = detect_outlying_inds_by_iqr(X_train[colname])

# Check all unique rows
unique_rows, counts =np.unique(np.concatenate([v for v in IQR_outliers.values()]), return_counts = True)
len(unique_rows) / len(y_train_log)
# Question: How many coloms within a row are failed
# Answer: mean=21, std = 20
pd.DataFrame({"x":counts}).describe()
del IQR_outliers, no_cols, i, unique_rows, counts
# Note, let's try Mahalanobis
mahal_distances = []
mahal_distances = np.empty(shape=[len(y_train)])
step = 1000
for left_bound in range(0 , len(y_train), step):
    print(f"Mahal Distance {100 * round(left_bound/len(y_train),2)}% done")
    if left_bound + step > len(y_train):
        right_bound = len(y_train) + 1
    else:
        right_bound = left_bound + step
    mahal_distances[left_bound:right_bound] = calculate_mahalanobis_distance(y=X_train[left_bound:right_bound], data = X_train)
p_values = 1 - scipy.stats.chi2.cdf(mahal_distances, X_train.shape[1] - 1)
len(np.where(p_values<.001)[0])/len(y_train)