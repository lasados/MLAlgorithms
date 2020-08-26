import numpy as np
import random


class Splitter:
    """Train/develop/test splitter."""

    def __init__(self, shuffle=True, train_size=0.7, dev_size=0.2):
        assert train_size + dev_size <= 1.0, "Bad size of sets"
        self.__shuffle = shuffle
        self.__train_size = train_size
        self.__dev_size = dev_size
        self.sets = None

    def split(self, X, y):
        """
        Splits X and y arrays on three parts.
        Arguments:
            X - np.array, X.shape = (N, nx)
            y - np.array, y.shape = (N, )
        Returns:
            sets - dictionary with train/dev/test sets
        """

        shuffle = self.__shuffle
        train_size = self.__train_size
        dev_size = self.__dev_size

        N, nx = X.shape

        assert y.shape[0] == N, "X and y arrays have different number of samples"

        idx = np.array(range(N))
        if shuffle:
            random.shuffle(idx)

        idx_train = idx[0: int(train_size * N)]
        idx_dev = idx[int(train_size * N): int((train_size + dev_size) * N)]
        idx_test = idx[int((train_size + dev_size) * N):]

        X_train, y_train = X[idx_train], y[idx_train]
        X_dev, y_dev = X[idx_dev], y[idx_dev]
        X_test, y_test = X[idx_test], y[idx_test]

        sets = {"train": {"X": X_train, "y": y_train},
                "dev": {"X": X_dev, "y": y_dev},
                "test": {"X": X_test, "y": y_test}}
        self.sets = sets
        return sets

    def get_train_set(self):
        """Return train set."""

        assert self.sets is not None, "Not splitted yet."
        return self.sets['train']

    def get_dev_set(self):
        """Return develop set."""

        assert self.sets is not None, "Not splitted yet."
        return self.sets['dev']

    def get_test_set(self):
        """Return test set."""

        assert self.sets is not None, "Not splitted yet."
        return self.sets['test']


class StandardScaler:
    """Class for standardization."""

    def __init__(self):
        self._u = None
        self._s = None

    def fit(self, X):
        u = np.mean(X, axis=0)
        s = np.std(X, axis=0)
        u = np.where(s == 0, 0, u)  # not transform if std=0
        s = np.where(s == 0, 1, s)  # replace std=0 on 1

        self._u = u
        self._s = s
        return self

    def transform(self, X):
        assert (self._u is not None) and (self._s is not None), "Not fitted yet."
        X = (X - self._u) / self._s
        return X

    def fit_transform(self, X):
        self.fit(X)
        self.transform(X)
