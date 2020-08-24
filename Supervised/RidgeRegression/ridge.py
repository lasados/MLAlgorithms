import numpy as np
import seaborn as sns
import random

from matplotlib import pyplot as plt


class Ridge:
    """
    Ridge regression.
    Parameters:
        alpha :
        solver :
        w_len :
    Methods:
        fit
        predict
    """
    def __init__(self, alpha=1.0, solver='gd', w_len=100, init_w=None):
        self._w = init_w
        self._learning_loss = None
        self.__params = {'alpha': alpha,
                         'solver': solver,
                         'w_len': w_len}

    def __loss(self, X, w, y_true):
        n_samples = X.shape[0]
        y_pred = np.dot(X, w)
        alpha = self.__params['alpha']

        loss = 1 / (2 * n_samples) * np.sum((y_pred - y_true) ** 2) + alpha / 2 * np.sum(w ** 2)
        dw = 1 / n_samples * np.dot((y_pred - y_true), X) + alpha * w

        return {'loss': loss, 'dw': dw}

    def __init_weights(self, nx):
        """
        Initialize weights with random numbers.
        Arguments:
            nx - int, X.shape = (N, nx)
        Returns:
            w - np.array
        """
        w_len = self.__params['w_len']
        w = w_len * np.random.random(nx)
        return w

    def fit(self, X_train, y_train, learning_rate=0.01, n_iter=1000, save_loss=True):
        assert self.__params['solver'] == 'gd', "Solver does not exists yet."

        # Init weights
        nx = X_train.shape[1]
        w = self.__init_weights(nx)

        # Init losses for each iteration
        if save_loss:
            loss_on_iter = {'losses': [],
                            'iterations': list(range(n_iter))}
        else:
            loss_on_iter = None

        # Learning
        for i in range(n_iter):
            cache = self.__loss(X_train, w, y_train)
            loss_current, dw_current = cache['loss'], cache['dw']
            w = w - learning_rate * dw_current

            if save_loss:
                loss_on_iter['losses'].append(loss_current)

        self._w = w
        self._learning_loss = loss_on_iter
        return self

    def predict(self, X):
        w = self._w
        assert w is not None, "Not fitted yet."
        y_pred = np.dot(X, w)
        return y_pred

    def plot_learning_curve(self, *args, **kwargs):
        iters = self._learning_loss['iterations']
        losses = self._learning_loss['losses']

        plt.figure(figsize=(12, 6))
        plt.plot(iters, losses, *args, **kwargs)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Learning Curve')
        plt.show()








