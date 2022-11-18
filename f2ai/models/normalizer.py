import numpy as np
import pandas as pd


class MinMaxNormalizer:
    """scale continous features to [-1,1],"""

    def __init__(self, feature_name=None):
        self.feature_name = feature_name

    def fit_self(self, data: pd.Series):
        min = data.min()
        max = data.max()
        self._state = ((min + max) / 2.0, max - min + 1e-8)
        return self

    def transform_self(self, data: pd.Series) -> pd.Series:
        assert self._state is not None
        center, scale = self._state

        return (data - center) / scale

    def inverse_transform_self(self, data: pd.Series) -> pd.Series:
        assert self._state is not None
        center, scale = self._state

        return data * scale + center


class StandardNormalizer:
    def __init__(self, feature_name=None, center=True):
        self.feature_name = feature_name
        self._center = center
        self._eps = 1e-6

    def fit_self(self, data: pd.Series, source: pd.DataFrame = None, **kwargs):
        if self._center:
            self._state = (data.mean(), data.std() + self._eps)
        else:
            self._state = (0.0, data.mean() + self._eps)

    def transform_self(self, data: pd.Series) -> pd.Series:
        assert self._state is not None
        center, scale = self._state

        return (data - center) / scale

    def inverse_transform_self(self, data: pd.Series) -> pd.Series:
        assert self._state is not None
        center, scale = self._state

        return data * scale + center
