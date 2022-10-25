import torch
from torch import nn
import numpy as np


class SimpleClassify(nn.Module):
    def __init__(self, cont_nbr, cat_nbr, emd_dim, max_types, hidden_size=16, drop_out=0.1) -> None:
        super().__init__()
        # num_embeddings not less than type
        self.categorical_embedding = nn.Embedding(num_embeddings=max_types, embedding_dim=emd_dim)
        hidden_list = 2 ** np.arange(max(np.ceil(np.log2(hidden_size)), 2) + 1)[::-1]
        model_list = [nn.Linear(cont_nbr + cat_nbr * emd_dim, int(hidden_list[0]))]
        for i in range(len(hidden_list) - 1):
            model_list.append(nn.Dropout(drop_out))
            model_list.append(nn.Linear(int(hidden_list[i]), int(hidden_list[i + 1])))

        self.model = nn.Sequential(*model_list, nn.Sigmoid())

    def forward(self, x):
        cat_vector = []
        for i in range(x["categorical_features"].shape[-1]):
            cat = self.categorical_embedding(x["categorical_features"][..., i])
            cat_vector.append(cat)
        cat_vector = torch.cat(cat_vector, dim=-1)
        input_vector = torch.cat([cat_vector, x["continous_features"]], dim=-1)
        return self.model(input_vector)
