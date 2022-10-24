import torch
import pandas as pd
from models.encoder import LabelEncoder
from models.normalizer import MinMaxNormalizer


def classify_collet_fn(datas, cont_scalar={}, cat_coder={}, label=[]):
    """_summary_

    Args:
        datas (_type_): datas to be processed, the length equals to batch_size
        cont_scalar (dict, optional):  normalize continous features, the key is feature name and value is scales corresponding to method using. Defaults to {}.
        cat_coder (dict, optional): encode categorical features, the key is feature name and value is scales corresponding to method using. Defaults to {}.
        label (list, optional): column name of label. Defaults to [].

    Returns:
        tuple: the first element is features and the second is label
    """
    batches = []
    # corresspondint to __get_item__ in Dataset
    for data in datas:  # data[0]:features, data[1]:labels
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
