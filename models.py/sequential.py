from typing import Iterator
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd


class SimpleClassify(nn.Module):
    def __init__(self, cont_nbr=15, cat_nbr=5, emd_dim=2, max_types=6) -> None:
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
    test_data = Iterator(
        pd.DataFrame(
            columns=[
                "cont_fea1",
                "cont_fea2",
                "cont_fea3",
                "cont_fea4",
                "cont_fea5",
                "cat_fea1",
                "cat_fea2",
                "label",
            ]
        )
    )

    def cutomized_collet_fn():
        pass

    test_data_loader = DataLoader(test_data, collate_fn=cutomized_collet_fn)
    # test_data_batch = {  # assume batch_size = 16
    #         "categorical_features": torch.randint(6, [16, 2], dtype=torch.int),
    #         "continous_features": torch.rand([16, 5], dtype=torch.float),
    #         "labels": torch.randint(2, [16, 1], dtype=torch.float),
    #     }
    model = SimpleClassify(cont_nbr=5, cat_nbr=2, emd_dim=2, max_types=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    for epoch in range(10):  # assume 10 epoch
        print(f"epoch: {epoch} begin")
        # each batch_data: batch_size * feature_dim * 1
        pred_label = model(test_data_loader)
        true_label = test_data["labels"]
        loss = loss_fn(pred_label, true_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch: {epoch} done")
