import os
import glob
from typing import List, Dict
from aie_feast.entity import Entity
from aie_feast.service import Service
from aie_feast.views import FeatureView, LabelView
from aie_feast.definations import OfflineStoreType

from .connect import ConnectConfig
from .source import Source, parse_source_yaml
from .read_file import read_yml
from .utils import remove_prefix, schema_to_dict, service_to_dict


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


def get_conn_cfg(url: str):
    """connect to pgsql using configs in feature_store.yml

    Args:
        url (str): url of .yml
    """
    cfg = read_yml(url)
    if cfg["offline_store"]["type"] == OfflineStoreType.FILE:
        conn = ConnectConfig(type=cfg["offline_store"]["type"])
    elif cfg["offline_store"]["type"] == OfflineStoreType.PGSQL:
        conn = ConnectConfig(
            type=cfg["offline_store"]["type"],
            user=cfg["offline_store"]["pgsql_conf"].get("user", "postgres"),
            passwd=cfg["offline_store"]["pgsql_conf"].get("password", "password"),
            host=cfg["offline_store"]["pgsql_conf"]["host"],
            port=cfg["offline_store"]["pgsql_conf"].get("port", "5432"),
            database=cfg["offline_store"]["pgsql_conf"]["database"],
            schema=cfg["offline_store"]["pgsql_conf"].get("schema", "public"),
        )
    elif cfg["offline_store"]["type"] == OfflineStoreType.SPARK:
        conn = ConnectConfig(type=cfg["offline_store"]["type"])  # TODO:will be implemented in future
    else:
        raise TypeError("offline_store must be one of [file, influxdb, pgsql]")
    # TODO:assert
    return conn


def get_service_cfg(url: str):
    """get forecast config like length of look_back and look_forward, features and labels

    Args:
        url (str): url of .yml
    """
    service_cfg = {}
    for filepath in listdir_yamls(url):
        cfg = read_yml(filepath)
        service = Service(
            features=service_to_dict(cfg["features"]),
            labels=service_to_dict(cfg["labels"]),
            materialize_path=cfg.get("materialize", "materialize_table"),
            materialize_type=cfg.get("type", "file"),
            dbt_path=cfg.get("dbt", "dbt_path"),
        )
        service_cfg.update({cfg["name"]: service})
    return service_cfg


def get_entity_cfg(url: str):
    """get entity config for join

    Args:
        url (str): url of .yml
    """
    entities = {}
    for filepath in listdir_yamls(url):
        cfg = read_yml(filepath)
        entity_cfg = Entity(name=cfg.get("join_keys", [cfg["name"]])[0])
        entities.update({cfg["name"]: entity_cfg})
    return entities


def get_feature_views(url: str):
    """get Dict(FeatureViews) from /feature_views/*.yml

    Args:
        url (str): rl of .yml
    """
    feature_views = {}
    for filepath in listdir_yamls(url):
        cfg = read_yml(filepath)
        feature_cfg = FeatureView(
            entity=cfg["entities"],
            features=schema_to_dict(cfg["schema"]),
            batch_source=cfg["batch_source"],
            ttl=cfg.get("ttl", None),
            exogenous=cfg.get("tags", {}).get("exogenous", None),
            request_source=cfg.get("request_source", None),
        )
        feature_views.update({cfg["name"]: feature_cfg})
    return feature_views


def get_label_views(url: str):
    """get Dict(LabelViews) from /label_views/*.yml

    Args:
        url (str): rl of .yml
    """
    label_views = {}
    for filepath in listdir_yamls(url):
        cfg = read_yml(filepath)
        label_cfg = LabelView(
            entity=cfg["entities"],
            labels=schema_to_dict(cfg["schema"]),
            batch_source=cfg["batch_source"],
            ttl=cfg.get("ttl", None),
            request_source=cfg.get("request_source", None),
        )
        label_views.update({cfg["name"]: label_cfg})
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
