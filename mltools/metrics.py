import numpy as np


def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_true_mean = np.mean(y_true)

    ss_residual = np.sum(np.power((y_true - y_pred), 2))
    ss_total = np.sum(np.power((y_true - y_true_mean), 2))

    r2 = 1 - ss_residual / ss_total
    return r2

