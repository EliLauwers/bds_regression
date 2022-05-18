import numpy as np



def mahalanobis(x=None, data=None, cov=None):
    x_mu = x - np.mean(data, axis=0)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    return mahal.diagonal()


def mahalanobis_iterative(X_train):
    mahal_distances = []
    mahal_distances = np.empty(shape=[len(X_train)])
    step = 1000
    for left_bound in range(0, len(X_train), step):
        print(f"\rMahal Distance {100 * round(left_bound / len(X_train), 2)}% done {' ' * 15}", end = "")
        right_bound = (
            len(X_train) + 1 if left_bound + step > len(X_train) else left_bound + step
        )
        # put the mahal_distances in the numpy array
        mahal_distances[left_bound:right_bound] = mahalanobis(
            x=X_train[left_bound:right_bound], data=X_train
        )
    return mahal_distances
