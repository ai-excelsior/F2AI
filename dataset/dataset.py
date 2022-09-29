from typing import Dict, List
from torch.utils.data import IterableDataset as ID
from aie_feast.views import FeatureViews, LabelViews


class IterableDataset:

    def __init__(self, dataset: "Dataset", start: str, end: str):
        super().__init__()
        self.dataset = dataset
        self.start = start
        self.end = end

    def __iter__(self):
        pass

    def __len__(self):
        pass


class Dataset:

    def __init__(
        self, 
        features: List[str],
        targets: List[str],
        start: str,
        end: str,
        sampler: callable,
        bucket: int,
        stride: int,
        include: str,
        features_map: Dict[str, FeatureViews] = {},
        labels_map: Dict[str, LabelViews] = {}
    ):
        self.features = features
        self.targets = targets
        self.start = start
        self.end = end
        self.sampler = sampler
        self.bucket = bucket
        self.stride = stride
        self.include = include
        self.features_map = features_map
        self.labels_map = labels_map

    def to_pytorch(self) -> IterableDataset:
        """convert to iterablt pytorch dataset"""

        pass



