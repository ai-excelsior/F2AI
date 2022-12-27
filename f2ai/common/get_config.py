import os
import glob
from typing import List, Dict

from ..definitions import (
    OfflineStoreType,
    Entity,
    FeatureView,
    LabelView,
    Service,
    Source,
    parse_source_yaml,
)
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
