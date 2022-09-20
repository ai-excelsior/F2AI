from aie_feast.entity import Entity
from aie_feast.service import ForecastService, Service
from aie_feast.views import FeatureViews, LabelViews
from .connect import ConnectConfig
from .source import SourceConfig
from .read_file import read_yml


def get_conn_cfg(url: str):
    """connect to pgsql using configs in feature_store.yml

    Args:
        url (str): url of .yml
    """
    cfg = read_yml(url)
    if cfg["offline_store"]["type"] == "file":
        conn = ConnectConfig(cfg)
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
    cfg = read_yml(url)
    entity_cfg = Entity(cfg)
    return {cfg["name"]: entity_cfg}


def get_feature_views(url: str):
    """get Dict(FeatureViews) from /feature_views/*.yml

    Args:
        url (str): rl of .yml
    """
    cfg = read_yml(url)
    feature_cfg = FeatureViews(cfg)
    return {cfg["name"]: feature_cfg}


def get_label_views(url: str):
    """get Dict(LabelViews) from /label_views/*.yml

    Args:
        url (str): rl of .yml
    """
    cfg = read_yml(url)
    label_cfg = LabelViews(cfg)
    return {cfg["name"]: label_cfg}


def get_source_cfg(url: str):
    """get Dict(LabelViews) from /sources/*.yml

    Args:
        url (str): rl of .yml
    """
    cfg = read_yml(url)
    source_cfg = SourceConfig(cfg)
    return {cfg["name"]: source_cfg}
