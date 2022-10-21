from multiprocessing.spawn import prepare
from typing import Iterator
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from aie_feast.featurestore import FeatureStore
from aie_feast.common.sampler import GroupFixednbrSampler


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
        super().__init__()
        self.feature_name = feature_name

    def get_norm(self, y: pd.Series, embeddings: dict = {}, **kwargs) -> np.ndarray:
        return np.tile(np.asarray([0, len(self._state) - 1]), (len(y), 1))

    def fit(self, y: pd.Series, embeddings: dict = {}, **kwargs):
        return self.fit_self(y, embeddings, **kwargs)

    def fit_self(self, y: pd.Series):
        y = column_or_1d(y, warn=True)
        self._state = ["UNKNOWN_CAT"] + sorted(set(y))
        return self

    def transform(self, y: pd.Series, embeddings: dict = {}, **kwargs):
        assert self._state is not None
        return self.transform_self(y, **kwargs)

    def transform_self(self, y: pd.Series, **kwargs) -> pd.Series:
        y = column_or_1d(y, warn=True)
        # transform of empty array is empty array
        if len(y) == 0:
            return np.array([])
        y = map_to_integer(y, self._state)
        return y

    def inverse_transform(self, y: pd.Series, embeddings: dict = {}, **kwargs):
        assert self._state is not None
        data_inversed = self.inverse_transform_self(y, **kwargs)
        return data_inversed

    def inverse_transform_self(self, y: pd.Series, **kwargs) -> pd.Series:
        y = column_or_1d(y, warn=True)
        if len(y) == 0:
            return np.array([])
        diff = np.setdiff1d(y, np.arange(len(self._state)))
        if len(diff):
            raise ValueError("y contains previously unseen labels: %s" % str(diff))
        y = np.asarray(y)
        return [self._state[i] for i in y]


class SimpleClassify(nn.Module):
    def __init__(self, cont_nbr, cat_nbr, emd_dim, max_types) -> None:
        super().__init__()
        # num_embeddings not less than type
        self.categorical_embedding = nn.Embedding(num_embeddings=max_types, embedding_dim=emd_dim)
        self.model = nn.Sequential(nn.Linear(cont_nbr + cat_nbr * emd_dim, 1), nn.Sigmoid())

    def forward(self, x):
        cat_vector = []
        for i in range(x["categorical_features"].shape[1]):
            cat = self.categorical_embedding(x["categorical_features"][:, i])
            cat_vector.append(cat)
        cat_vector = torch.cat(cat_vector, dim=-1)
        input_vector = torch.cat([cat_vector, x["continous_features"]], dim=-1)
        return self.model(input_vector)


if __name__ == "__main__":
    fs = FeatureStore("file:///Users/xuyizhou/Desktop/xyz_warehouse/gitlab/f2ai-credit-scoring")
    # test_data = Iterator(  # just example, use your data to replace;
    #     pd.DataFrame(
    #         columns=[
    #             "cont_fea1",
    #             "cont_fea2",
    #             "cont_fea3",
    #             "cont_fea4",
    #             "cont_fea5",
    #             "cat_fea1",
    #             "cat_fea2",
    #             "label",
    #         ]
    #     )
    # )

    def cutomized_collet_fn(datas, cont_scalar={}, cat_coder={}):
        # cutomize your collet_fn to adjust SimpleClassify Model
        # involve pre-process 1. encoder str-like features to numeric;
        #                     2. scale numeric features, the oveall min/max/avg/std can be accessed by `fs.stats` and transported or written in this func;
        #                     3. others if need
        # involve collect method to convert original data format to that used in SimpleClassify.forward and loss calculation

        for data in datas:
            batches = data
        categorical_features = torch.stack([batch[0]["categorical_features"] for batch in batches])
        continous_features = torch.stack([batch[0]["continous_features"] for batch in batches])
        labels = torch.stack([batch[1]["labels"] for batch in batches])
        return (
            dict(
                categorical_features=categorical_features,
                continous_features=continous_features,
            ),
            labels,
        )

    cat_unique = fs.stats(fs.services["credit_scoring_v1"], fn="unique", group_key=[])
    ds = fs.get_dataset(
        service_name="credit_scoring_v1",
        sampler=GroupFixednbrSampler(
            time_bucket="10 days",
            stride=1,
            group_ids=None,
            group_names=None,
            start="2020-08-01",
            end="2021-09-30",
        ),
    )
    i_ds = ds.to_pytorch()

    test_data_loader = DataLoader(  # `batch_siz`e and `drop_last`` do not matter now, `sampler`` set it to be None cause `test_data`` is a Iterator
        i_ds,
        collate_fn=lambda x: cutomized_collet_fn(x, cat_coder={}, cont_scalar={}),
        batch_size=4,
        drop_last=False,
        sampler=None,
    )
    # cont_nbr/cat_nbr means the number of continuous/categorical features delivered to model;
    # max_types means the max different types in all categorical features
    # emd_dim is a parameter do not matter now
    model = SimpleClassify(cont_nbr=5, cat_nbr=2, emd_dim=2, max_types=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # no need to change
    loss_fn = nn.BCELoss()  # loss function to train a classification model

    def prepare_batch():
        pass

    for epoch in range(10):  # assume 10 epoch
        print(f"epoch: {epoch} begin")
        for data in test_data_loader:
            pred_label = model(x)
            true_label = y
            loss = loss_fn(pred_label, true_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch} done, loss: {loss}")
