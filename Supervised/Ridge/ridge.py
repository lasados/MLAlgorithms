import pandas as pd
import numpy as np
import seaborn as sns
import random

from sklearn.datasets import load_boston
from matplotlib import pyplot as plt

boston_dataset = load_boston()
sns.set()


def train_dev_test_split(X, y, shuffle=True, train_size=0.7, dev_size=0.2):
    """
    Splits given arrays of X and y on three parts.
    Arguments:
        X - np.array of features, X.shape = (N, nx)
        y - np.array of target, y.shape = (N, 1)
        N - number of samples
    Returns:
        [X_train, y_train], [X_dev, y_dev], [X_test, y_test]
    """

    N, nx = X.shape

    assert y.shape[0] == N, "X and y arrays have different number of samples"
    assert train_size + dev_size <= 1.0, "Bad size of sets"

    idx = np.array(range(N))
    if shuffle:
        random.shuffle(idx)

    idx_train = idx[0: int(train_size * N)]
    idx_dev = idx[int(train_size * N): int((train_size + dev_size) * N)]
    idx_test = idx[int((train_size + dev_size) * N):]

    X_train, y_train = X[idx_train], y[idx_train]
    X_dev, y_dev = X[idx_dev], y[idx_dev]
    X_test, y_test = X[idx_test], y[idx_test]

    return [X_train, y_train], [X_dev, y_dev], [X_test, y_test]


def standart_scaler(X_train, X_dev, X_test):
    """
    Scales arrays based on train set with Standart_Scale.
    Arguments:
        X_train, X_dev, X_test - np.arrays
    Returns:
        X_train_scale, X_dev_scale, X_test_scale - np.arrays
    """

    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)

    X_train_scale = (X_train - mu) / sigma
    X_dev_scale = (X_dev - mu) / sigma
    X_test_scale = (X_test - mu) / sigma

    return X_train_scale, X_dev_scale, X_test_scale


def init_weights(nx):
    """
    Initialize weight with random numbers.
    Arguments:
        nx - int, X.shape = (N, nx)
    Returns:
        w - np.array
    """

    w = 100 * np.random.random(nx)
    return w


def ridge_loss(X_train, y_train, w, lamda=1.0):
    N = X_train.shape[0]
    A = np.dot(X_train, w)

    Q = 1 / (2 * N) * np.sum((A - y_train) ** 2) + lamda / 2 * np.sum(w ** 2)

    dw = 1 / N * np.dot((A - y_train), X_train) + lamda * w

    return Q, dw


def update_weights(w, dw, learning_rate=0.01):
    return w - learning_rate*dw