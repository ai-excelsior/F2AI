from aie_feast.entity import Entity
from aie_feast.service import ForecastService, Service
from aie_feast.views import FeatureViews, LabelViews
from .connect import ConnectConfig
from .source import SourceConfig
from .read_file import read_yml
from .utils import remove_prefix, schema_to_dict
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
        conn = ConnectConfig(cfg)
    else:
        raise TypeError("offline_store must be one of [file, influxdb, pgsql]")
    # TODO:assert
    return conn


def get_service_cfg(url: str):
    """get forecast config like length of look_back and look_forward, features and labels

    Args:
        url (str): url of .yml
    """
    cfg = read_yml(url)
    service_cfg = Service(cfg)
    # if cfg xxxxxx else
    service_cfg = ForecastService(cfg)
    return {cfg["name"]: service_cfg}


def get_entity_cfg(url: str):
    """get entity config for join

    Args:
        url (str): url of .yml
    """
    entities = {}
    for cfg in os.listdir(remove_prefix(url, "file://")):
        cfg = read_yml(os.path.join(url, cfg))
        if cfg.get("join_keys") is not None:
            join_keys = cfg["join_keys"][0]
        else:
            join_keys = cfg["name"]
        entity_cfg = Entity()
        entity_cfg.entity = join_keys
        entities.update({cfg["name"]: entity_cfg})
    return entities


def get_feature_views(url: str):
    """get Dict(FeatureViews) from /feature_views/*.yml

    Args:
        url (str): rl of .yml
    """
    feature_views = {}
    for cfg in os.listdir(remove_prefix(url, "file://")):
        cfg = read_yml(os.path.join(url, cfg))
        feature_cfg = FeatureViews(
            entity=cfg["entities"],
            features=schema_to_dict(cfg["schema"]),
            batch_source=cfg["batch_source"],
            ttl=cfg.get("ttl", None),
            exogenous=cfg.get("exogenous", None),
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
    cfg = read_yml(url)
    source_cfg = SourceConfig(**cfg)
    return cfg["name"], source_cfg
