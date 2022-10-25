# %% [markdown]
# <font size=6>Example: Credit Score Classification</font>
#

# %% [markdown]
# <font size=2>`FeatureStore` is a model-agnostic tool aiming to help data scientists and algorithm engineers get rid of tiring data storing and merging tasks.
# <br>`FeatureStore` not only work on single-dimension data such as classification and prediction, but also work on time-series data.
#  <br>After collecting data, all you need to do is config several straight-forward .yml files, then you can focus on  models/algorithms and leave all exhausting preparation to `FeatureStore`.</font>
#  <font size=4><br><br>Here we present credit scoring mission as a single-dimension data demo, it takes features like wages, loan records to decide whether to grant credit or not.</font>

# %% [markdown]
# <font size=4>Import packages</font>

# %%
import torch
import os
import numpy as np
import zipfile
import tempfile
from torch import nn
from torch.utils.data import DataLoader
from aie_feast.featurestore import FeatureStore
from aie_feast.common.sampler import GroupFixednbrSampler
from aie_feast.common.collect_fn import classify_collet_fn
from aie_feast.common.utils import get_bucket_from_oss_url
from aie_feast.models.earlystop import EarlyStopping
from aie_feast.models.sequential import SimpleClassify

# %% [markdown]
# <font size=4>Download demo project files from `OSS` </font>

# %%
download_from = "oss://aiexcelsior-shanghai-test/xyz_test_data/credit_score.zip"
save_path = "/tmp/"
save_dir = tempfile.mkdtemp(prefix=save_path)
bucket, key = get_bucket_from_oss_url(download_from)
dest_zip_filepath = os.path.join(save_dir, key)
os.makedirs(os.path.dirname(dest_zip_filepath), exist_ok=True)
bucket.get_object_to_file(key, dest_zip_filepath)
zipfile.ZipFile(dest_zip_filepath).extractall(dest_zip_filepath.rsplit("/", 1)[0])
os.remove(dest_zip_filepath)
print(f"Project downloaded and saved in {dest_zip_filepath.rsplit('/',1)[0]}")

# %% [markdown]
# <font size=4>Initialize `FeatureStore`</font>

# %%
TIME_COL = "event_timestamp"
fs = FeatureStore(f"file://{save_dir}/{key.rstrip('.zip')}")

# %%
print(f"All features are: {fs._get_available_features(fs.services['credit_scoring_v1'])}")

# %% [markdown]
# <font size=4>Get the time range of available data</font>

# %%
print(f'Earliest timestamp: {fs.get_latest_entities(fs.services["credit_scoring_v1"])[TIME_COL].min()}')
print(f'Latest timestamp: {fs.get_latest_entities(fs.services["credit_scoring_v1"])[TIME_COL].max()}')

# %% [markdown]
# <font size=4>Split the train / valid / test data at approximately 7/2/1, use `GroupFixednbrSampler` to downsample original data and return a `torch.IterableDataset`</font>

# %%
ds_train = fs.get_dataset(
    service_name="credit_scoring_v1",
    sampler=GroupFixednbrSampler(
        time_bucket="5 days",
        stride=1,
        group_ids=None,
        group_names=None,
        start="2020-08-20",
        end="2021-04-30",
    ),
)
ds_valid = fs.get_dataset(
    service_name="credit_scoring_v1",
    sampler=GroupFixednbrSampler(
        time_bucket="5 days",
        stride=1,
        group_ids=None,
        group_names=None,
        start="2021-04-30",
        end="2021-07-31",
    ),
)
ds_test = fs.get_dataset(
    service_name="credit_scoring_v1",
    sampler=GroupFixednbrSampler(
        time_bucket="1 days",
        stride=1,
        group_ids=None,
        group_names=None,
        start="2021-07-31",
        end="2021-08-31",
    ),
)

# %% [markdown]
# <font size=4>Using `FeatureStore.stats` to obtain `statistical results` for data processing</font>

# %%
# catgorical features
features_cat = [
    fea
    for fea in fs._get_available_features(fs.services["credit_scoring_v1"])
    if fea not in fs._get_available_features(fs.services["credit_scoring_v1"], is_numeric=True)
]
# get unique item number to do labelencoder
cat_unique = fs.stats(
    fs.services["credit_scoring_v1"],
    fn="unique",
    group_key=[],
    start="2020-08-01",
    end="2021-04-30",
    features=features_cat,
).to_dict()
cat_count = {key: len(cat_unique[key]) for key in cat_unique.keys()}
print(f"Number of unique values of categorical features are: {cat_count}")

# %%
# contiouns features
cont_scalar_max = fs.stats(
    fs.services["credit_scoring_v1"], fn="max", group_key=[], start="2020-08-01", end="2021-04-30"
).to_dict()
cont_scalar_min = fs.stats(
    fs.services["credit_scoring_v1"], fn="min", group_key=[], start="2020-08-01", end="2021-04-30"
).to_dict()
cont_scalar = {key: [cont_scalar_min[key], cont_scalar_max[key]] for key in cont_scalar_min.keys()}
print(f"Min-Max boundary of continuous features are: {cont_scalar}")

# %% [markdown]
# <font size=4>Construct `torch.DataLoader` from  `torch.IterableDataset` for modelling</font>
# <font size = 3><br>Here we compose data-preprocess in `collect_fn`, so the time range of `statistical results` used to `.fit()` should be corresponding to `train` data only so as to avoid information leakage. </font>

# %%
batch_size = 16

train_dataloader = DataLoader(
    ds_train.to_pytorch(),
    collate_fn=lambda x: classify_collet_fn(
        x,
        cat_coder=cat_unique,
        cont_scalar=cont_scalar,
        label=fs._get_available_labels(fs.services["credit_scoring_v1"]),
    ),
    batch_size=batch_size,
    drop_last=True,
)

valie_dataloader = DataLoader(
    ds_valid.to_pytorch(),
    collate_fn=lambda x: classify_collet_fn(
        x,
        cat_coder=cat_unique,
        cont_scalar=cont_scalar,
        label=fs._get_available_labels(fs.services["credit_scoring_v1"]),
    ),
    batch_size=batch_size,
    drop_last=False,
)

test_dataloader = DataLoader(
    ds_valid.to_pytorch(),
    collate_fn=lambda x: classify_collet_fn(
        x,
        cat_coder=cat_unique,
        cont_scalar=cont_scalar,
        label=fs._get_available_labels(fs.services["credit_scoring_v1"]),
    ),
    drop_last=False,
)

# %% [markdown]
# <font size=4>Customize `model`, `optimizer` and `loss` function suitable to task</font>

# %%
model = SimpleClassify(
    cont_nbr=len(cont_scalar_max),
    cat_nbr=len(cat_count),
    emd_dim=8,
    max_types=max(cat_count.values()),
    hidden_size=4,
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

# %% [markdown]
# <font size=4>Use `train_dataloader` to train while `valie_dataloader` to guide `earlystop`</font>

# %%
# you can also use any ready-to-use training frame like ignite, pytorch-lightening...
early_stop = EarlyStopping(save_path=f"{save_dir}/{key.rstrip('.zip')}", patience=5, delta=1e-6)
for epoch in range(50):
    train_loss = []
    valid_loss = []

    model.train()
    for x, y in train_dataloader:
        pred_label = model(x)
        true_label = y
        loss = loss_fn(pred_label, true_label)
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    for x, y in valie_dataloader:
        pred_label = model(x)
        true_label = y
        loss = loss_fn(pred_label, true_label)
        valid_loss.append(loss.item())

    print(f"epoch: {epoch} done, train loss: {np.mean(train_loss)}, valid loss: {np.mean(valid_loss)}")
    early_stop(np.mean(valid_loss), model)
    if early_stop.early_stop:
        print(f"Trigger earlystop, stop epoch at {epoch}")
        break

# %% [markdown]
# <font size=4>Get prediction result of `test_dataloader`</font>

# %%
model = torch.load(os.path.join(f"{save_dir}/{key.rstrip('.zip')}", "best_chekpnt.pk"))
model.eval()
preds = []
trues = []
for x, y in test_dataloader:
    pred = model(x)
    pred_label = 1 if pred.cpu().detach().numpy() > 0.5 else 0
    preds.append(pred_label)
    trues.append(y.cpu().detach().numpy())

# %% [markdown]
# <font size =4>Model Evaluation</font>

# %%
# accuracy
acc = [1 if preds[i] == trues[i] else 0 for i in range(len(trues))]
print(f"Accuracy: {np.sum(acc) / len(acc)}")
