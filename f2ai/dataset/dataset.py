from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..definitions.services import Service
    from ..featurestore import FeatureStore
    from .pytorch_dataset import TorchIterableDataset


class Dataset:
    def __init__(
        self,
        fs: "FeatureStore",
        service: "Service",
        sampler: callable,
    ):
        self.fs = fs
        self.service = service
        self.sampler = sampler

    def to_pytorch(self, batch: int = None) -> "TorchIterableDataset":
        """convert to iterablt pytorch dataset really hold data"""
        from .pytorch_dataset import TorchIterableDataset

        entity_index = self.sampler()

        return TorchIterableDataset(fs=self.fs, service=self.service, entity_index=entity_index, batch=batch)
