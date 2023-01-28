from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..definitions.services import Service
    from ..featurestore import FeatureStore
    from .pytorch_dataset import TorchIterableDataset


class Dataset:
    """
    A dataset is an abstraction, which hold a service and a sampler.
    A service basic tells us, where is the data. A sampler tells us which parts of the data should be retrieved.

    Note: We should not construct Dataset by ourself. using `store.get_dataset()` is recommended.
    """

    def __init__(
        self,
        fs: "FeatureStore",
        service: "Service",
        sampler: callable,
    ):
        self.fs = fs
        self.service = service
        self.sampler = sampler

    def to_pytorch(self, chunk_size: int = 64) -> "TorchIterableDataset":
        """convert to iterable pytorch dataset really hold data"""
        from .pytorch_dataset import TorchIterableDataset

        return TorchIterableDataset(feature_store=self.fs, service=self.service, sampler=self.sampler, chunk_size=chunk_size)
