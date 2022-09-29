from typing import Type
from aie_feast.entity import Entity
from aie_feast.service import Service
from aie_feast.views import FeatureViews, LabelViews
from .connect import ConnectConfig
from .source import SourceConfig
from .read_file import read_yml
from .utils import remove_prefix, schema_to_dict, service_to_dict
import os


def get_conn_cfg(url: str):
    """connect to pgsql using configs in feature_store.yml

    Args:
        url (str): url of .yml
    """
    cfg = read_yml(url)
    if cfg["offline_store"]["type"] == "file":
        conn = ConnectConfig(type=cfg["offline_store"]["type"])
    elif cfg["offline_store"]["type"] == "influxdb":
        conn = ConnectConfig(cfg)
    elif cfg["offline_store"]["type"] == "pgsql":
        conn = ConnectConfig(
            type=cfg["offline_store"]["type"],
            user=cfg["offline_store"]["pgsql_conf"].get("user", "postgres"),
            passwd=cfg["offline_store"]["pgsql_conf"].get("password", "password"),
            host=cfg["offline_store"]["pgsql_conf"]["host"],
            port=cfg["offline_store"]["pgsql_conf"].get("port", "5432"),
            database=cfg["offline_store"]["pgsql_conf"]["database"],
            schema=cfg["offline_store"]["pgsql_conf"].get("schema", "public"),
        )
    elif cfg["offline_store"]["type"] == "spark":
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
    for cfg in os.listdir(remove_prefix(url, "file://")):
        if cfg.endswith(".yml") or cfg.endswith(".yaml"):
            cfg = read_yml(os.path.join(url, cfg))
            service = Service(
                features=service_to_dict(cfg["features"]),
                labels=service_to_dict(cfg["labels"]),
                materialize_path=cfg.get("materialize", "materialize_table"),
            )
            service_cfg.update({cfg["name"]: service})
    return service_cfg


def get_entity_cfg(url: str):
    """get entity config for join

    Args:
        url (str): url of .yml
    """
    entities = {}
    for cfg in os.listdir(remove_prefix(url, "file://")):
        if cfg.endswith(".yml") or cfg.endswith(".yaml"):
            cfg = read_yml(os.path.join(url, cfg))
            entity_cfg = Entity(entity=cfg.get("join_keys", [cfg["name"]])[0])
            entities.update({cfg["name"]: entity_cfg})
    return entities


def get_feature_views(url: str):
    """get Dict(FeatureViews) from /feature_views/*.yml

    Args:
        url (str): rl of .yml
    """
    feature_views = {}
    for cfg in os.listdir(remove_prefix(url, "file://")):
        if cfg.endswith(".yml") or cfg.endswith(".yaml"):
            cfg = read_yml(os.path.join(url, cfg))
            feature_cfg = FeatureViews(
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
    for cfg in os.listdir(remove_prefix(url, "file://")):
        if cfg.endswith(".yml") or cfg.endswith(".yaml"):
            cfg = read_yml(os.path.join(url, cfg))
            label_cfg = LabelViews(
                entity=cfg["entities"],
                labels=schema_to_dict(cfg["schema"]),
                batch_source=cfg["batch_source"],
                ttl=cfg.get("ttl", None),
                request_source=cfg.get("request_source", None),
            )
            label_views.update({cfg["name"]: label_cfg})
    return label_views


def get_source_cfg(url: str):
    """get Dict(LabelViews) from /sources/*.yml

    Args:
        url (str): rl of .yml
    """
    source_dict = {}
    for filename in os.listdir(remove_prefix(url, "file://")):
        if filename.endswith(".yml") or cfg.endswith(".yaml"):
            cfg = read_yml(os.path.join(url, filename))
            cfg["file_path"] = cfg.pop("path", None)
            cfg["event_time"] = cfg.pop("timestamp_field", None)
            cfg["create_time"] = cfg.pop("created_timestamp_column", None)
            cfg["request_features"] = cfg.pop("schema", None)
            cfg["tags"] = cfg.pop("tags", None)
            source_cfg = SourceConfig(**cfg)
            source_dict.update({cfg["name"]: source_cfg})
    return source_dict
