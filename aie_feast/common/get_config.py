import os
import glob
from typing import List, Dict
from aie_feast.offline_stores.offline_file_store import OfflineFileStore
from aie_feast.offline_stores.offline_postgres_store import OfflinePostgresStore
from aie_feast.offline_stores.offline_spark_store import OfflineSparkStore
from aie_feast.definitions import OfflineStoreType, Entity, FeatureView, LabelView, Service

from .source import Source, parse_source_yaml
from .read_file import read_yml
from .utils import remove_prefix


def listdir_with_extensions(path: str, extensions: List[str] = []) -> List[str]:
    path = remove_prefix(path, "file://")
    if os.path.isdir(path):
        files = []
        for extension in extensions:
            files.extend(glob.glob(f"{path}/*.{extension}"))
        return files
    return []


def listdir_yamls(path: str) -> List[str]:
    return listdir_with_extensions(path, extensions=["yml", "yaml"])


def get_offline_store_from_cfg(url: str):
    """connect to pgsql using configs in feature_store.yml

    Args:
        url (str): url of .yml
    """
    cfg = read_yml(url)
    if cfg["offline_store"]["type"] == OfflineStoreType.FILE:
        offline_store = OfflineFileStore()
    elif cfg["offline_store"]["type"] == OfflineStoreType.PGSQL:
        offline_store = OfflinePostgresStore(**cfg["offline_store"]["pgsql_conf"])
    elif cfg["offline_store"]["type"] == OfflineStoreType.SPARK:
        offline_store = OfflineSparkStore(type=cfg["offline_store"]["type"])
    else:
        raise TypeError("offline_store must be one of [file, influxdb, pgsql]")

    return offline_store


def get_service_cfg(url: str) -> Dict[str, Service]:
    """get forecast config like length of look_back and look_forward, features and labels

    Args:
        url (str): url of .yml
    """
    service_cfg = {}
    for filepath in listdir_yamls(url):
        service = Service.from_yaml(read_yml(filepath))
        service_cfg[service.name] = service
    return service_cfg


def get_entity_cfg(url: str) -> Dict[str, Entity]:
    """get entity config for join

    Args:
        url (str): url of .yml
    """
    entities = {}
    for filepath in listdir_yamls(url):
        entity = Entity(**read_yml(filepath))
        entities[entity.name] = entity
    return entities


def get_feature_views(url: str) -> Dict[str, FeatureView]:
    """get Dict(FeatureViews) from /feature_views/*.yml

    Args:
        url (str): rl of .yml
    """
    feature_views = {}
    for filepath in listdir_yamls(url):
        feature_view = FeatureView(**read_yml(filepath))
        feature_views[feature_view.name] = feature_view
    return feature_views


def get_label_views(url: str) -> Dict[str, LabelView]:
    """get Dict(LabelViews) from /label_views/*.yml

    Args:
        url (str): rl of .yml
    """
    label_views = {}
    for filepath in listdir_yamls(url):
        label_view = LabelView(**read_yml(filepath))
        label_views[label_view.name] = label_view
    return label_views


def get_source_cfg(url: str, offline_store_type: OfflineStoreType) -> Dict[str, Source]:
    """get Dict(LabelViews) from /sources/*.yml

    Args:
        url (str): rl of .yml
    """

    source_dict = {}
    for filepath in listdir_yamls(url):
        cfg = read_yml(filepath)
        source = parse_source_yaml(cfg, offline_store_type)
        source_dict.update({source.name: source})
    return source_dict
