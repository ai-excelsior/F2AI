from aie_feast.common.collecy_fn import classify_collet_fn
import torch
from torch import nn
from torch.utils.data import DataLoader
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
        collate_fn=lambda x: classify_collet_fn(
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
