import numpy as np

def calculate_mahalanobis_distance(y=None, data=None, cov=None):
    y_mu = y - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_covmat)
    mahal = np.dot(left, y_mu.T)
    return mahal.diagonal()

def calculate_mahalanobis_distance_iteratevely(y_train, X_train):
    mahal_distances = []
    mahal_distances = np.empty(shape=[len(y_train)])
    step = 1000
    for left_bound in range(0, len(y_train), step):
        print(f"Mahal Distance {100 * round(left_bound / len(y_train), 2)}% done")
        right_bound = len(y_train) + 1 if left_bound + step > len(y_train) else left_bound + step
        # put the mahal_distances in the numpy array
        mahal_distances[left_bound:right_bound] = calculate_mahalanobis_distance(y=X_train[left_bound:right_bound],
                                                                                 data=X_train)
    return mahal_distances

