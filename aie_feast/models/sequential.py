from multiprocessing.spawn import prepare
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from encoder import LabelEncoder
from normalizer import MinMaxNormalizer
from aie_feast.featurestore import FeatureStore
from aie_feast.common.sampler import GroupFixednbrSampler


class SimpleClassify(nn.Module):
    def __init__(self, cont_nbr, cat_nbr, emd_dim, max_types) -> None:
        super().__init__()
        # num_embeddings not less than type
        self.categorical_embedding = nn.Embedding(num_embeddings=max_types, embedding_dim=emd_dim)
        self.model = nn.Sequential(nn.Linear(cont_nbr + cat_nbr * emd_dim, 1), nn.Sigmoid())

    def forward(self, x):
        cat_vector = []
        for i in range(x["categorical_features"].shape[-1]):
            cat = self.categorical_embedding(x["categorical_features"][..., i])
            cat_vector.append(cat)
        cat_vector = torch.cat(cat_vector, dim=-1)
        input_vector = torch.cat([cat_vector, x["continous_features"]], dim=-1)
        return self.model(input_vector)


if __name__ == "__main__":
    fs = FeatureStore("file:///Users/xuyizhou/Desktop/xyz_warehouse/gitlab/f2ai-credit-scoring")

    def cutomized_collet_fn(datas, cont_scalar={}, cat_coder={}, label=[]):
        batches = []
        # corresspondint to __get_item__ in Dataset
        for data in datas:
            cat_features = torch.stack(
                [
                    torch.tensor(
                        LabelEncoder(cat).fit_self(pd.Series(cat_coder[cat])).transform_self(data[0][cat]),
                        dtype=torch.int,
                    )
                    for cat in cat_coder.keys()
                ],
                dim=-1,
            )
            cont_features = torch.stack(
                [
                    torch.tensor(
                        MinMaxNormalizer(cont)
                        .fit_self(pd.Series(cont_scalar[cont]))
                        .transform_self(data[0][cont]),
                        dtype=torch.float16,
                    )
                    for cont in cont_scalar.keys()
                ],
                dim=-1,
            )
            labels = torch.stack(
                [torch.tensor(data[1][lab], dtype=torch.float) for lab in label],
                dim=-1,
            )
            batch = (dict(categorical_features=cat_features, continous_features=cont_features), labels)
            batches.append(batch)

        # corresspondint to _collect_fn_ in Dataset
        categorical_features = torch.stack([batch[0]["categorical_features"] for batch in batches])
        continous_features = torch.stack([batch[0]["continous_features"] for batch in batches])
        labels = torch.stack([batch[1] for batch in batches])
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
    features_cat = [  # catgorical features
        fea
        for fea in fs._get_available_features(fs.services["credit_scoring_v1"])
        if fea not in fs._get_available_features(fs.services["credit_scoring_v1"], True)
    ]
    cat_unique = fs.stats(
        fs.services["credit_scoring_v1"],
        fn="unique",
        group_key=[],
        start="2020-08-01",
        end="2021-09-30",
        features=features_cat,
    ).to_dict()
    cat_count = {key: len(cat_unique[key]) for key in cat_unique.keys()}
    cont_scalar_max = fs.stats(
        fs.services["credit_scoring_v1"], fn="max", group_key=[], start="2020-08-01", end="2021-09-30"
    ).to_dict()
    cont_scalar_min = fs.stats(
        fs.services["credit_scoring_v1"], fn="min", group_key=[], start="2020-08-01", end="2021-09-30"
    ).to_dict()
    cont_scalar = {key: [cont_scalar_min[key], cont_scalar_max[key]] for key in cont_scalar_min.keys()}

    i_ds = ds.to_pytorch()
    test_data_loader = DataLoader(  # `batch_siz`e and `drop_last`` do not matter now, `sampler`` set it to be None cause `test_data`` is a Iterator
        i_ds,
        collate_fn=lambda x: cutomized_collet_fn(
            x,
            cat_coder=cat_unique,
            cont_scalar=cont_scalar,
            label=fs._get_available_labels(fs.services["credit_scoring_v1"]),
        ),
        batch_size=4,
        drop_last=False,
        sampler=None,
    )

    model = SimpleClassify(
        cont_nbr=len(cont_scalar_max), cat_nbr=len(cat_count), emd_dim=4, max_types=max(cat_count.values())
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # no need to change
    loss_fn = nn.BCELoss()  # loss function to train a classification model

    for epoch in range(10):  # assume 10 epoch
        print(f"epoch: {epoch} begin")
        for x, y in test_data_loader:
            pred_label = model(x)
            true_label = y
            loss = loss_fn(pred_label, true_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch: {epoch} done, loss: {loss}")
