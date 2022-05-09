from sklearn.metrics import log_loss
import numpy as np
from GLOBAL_VARS import LOG


def adjusted_rsquare(R2, N, p):
    """
    R^2_a = 1 - (1-R^2)(N-1)/(N-p-1)
    :param R2: The original R2
    :param N: Total Sample Size (no. observations)
    :param p: number of independent variables
    :return: R^2_a
    """
    return 1 - (1 - R2) * (N - 1) / (N - p - 1)


def aik(K, ln):
    """
    see https://www.statology.org/aic-in-python/
    :param K: Number of parameters in the model
    :param ln: log likelihood of the model
    :return: Akaike information criterion
    """
    return 2 * K - 2 * ln


def mallows_cp(SSE, S2, N, P):
    """
    See https://en.wikipedia.org/wiki/Mallows%27s_Cp
    See also https://www.statology.org/mallows-cp/
    :param SSE: SS errors
    :param S2: Use MSE
    :param N: Total sample size
    :param P: Number of regressors
    :return: Mallows's Cp
    """
    return (SSE / S2) - N + 2 * (P + 1)
