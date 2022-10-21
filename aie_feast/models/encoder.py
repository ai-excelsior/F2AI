import numpy as np
import pandas as pd


def column_or_1d(y, warn):
    y = np.asarray(y)
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    elif len(shape) == 2 and shape[1] == 1:
        return np.ravel(y)

    raise ValueError("y should be a 1d array, got an array of shape {} instead.".format(shape))


def map_to_integer(values, uniques):
    """Map values based on its position in uniques."""
    table = {val: i for i, val in enumerate(uniques)}
    return np.array([table[v] if v in table else table["UNKNOWN_CAT"] for v in values])


class LabelEncoder:
    def __init__(self, feature_name=None):
        self.feature_name = feature_name

    def fit_self(self, y: pd.Series):
        y = column_or_1d(y, warn=True)
        self._state = ["UNKNOWN_CAT"] + sorted(set(y))
        return self

    def transform_self(self, y: pd.Series) -> pd.Series:
        y = column_or_1d(y, warn=True)
        # transform of empty array is empty array
        if len(y) == 0:
            return np.array([])
        y = map_to_integer(y, self._state)
        return y

    def inverse_transform_self(self, y: pd.Series) -> pd.Series:
        y = column_or_1d(y, warn=True)
        if len(y) == 0:
            return np.array([])
        diff = np.setdiff1d(y, np.arange(len(self._state)))
        if len(diff):
            raise ValueError("y contains previously unseen labels: %s" % str(diff))
        y = np.asarray(y)
        return [self._state[i] for i in y]
