import numpy as np


def detect_outlying_inds_by_iqr(series):
    q1, q3 = series.quantile([0.25, 0.75])  # calculate needed quantiles
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr  # calculate upper and lower bound
    upper = q3 + 1.5 * iqr
    # Get the indexes in the series of instances where the series value
    # lays outside the iqr-proof region
    return np.where(np.logical_or(series < lower, series > upper))[
        0
    ].tolist()  # return indexes
