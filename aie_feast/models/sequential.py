from multiprocessing.spawn import prepare
from typing import Iterator
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from argparse import ArgumentParser

from aie_feast.common.cmd_parser import get_f2ai_parser
from aie_feast.featurestore import FeatureStore
from aie_feast.common.sampler import GroupFixednbrSampler, GroupRandomSampler, UniformNPerGroupSampler


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

    def cutomized_collet_fn(batches):
        # cutomize your collet_fn to adjust SimpleClassify Model
        # involve pre-process 1. encoder str-like features to numeric;
        #                     2. scale numeric features, the oveall min/max/avg/std can be accessed by `fs.stats` and transported or written in this func;
        #                     3. others if need
        # involve collect method to convert original data format to that used in SimpleClassify.forward and loss calculation

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
        i_ds, collate_fn=cutomized_collet_fn, batch_size=4, drop_last=False, sampler=None
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
