from .confg import ForecastConfig, EntityConfig
from .oss_utils import get_bucket_from_oss_url
import yaml


def _read_yml(url: str):
    """read .yml file for following execute

    Args:
        url (str): url of .yml
    """
    file = _read_file(url)
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    if cfg["offline_store"]["type"] == "file":
        pass
    elif cfg["offline_store"]["type"] == "influxdb":
        pass
    elif cfg["offline_store"]["type"] == "pgsql":
        pass
    else:
        raise TypeError("offline_store must be one of [file, influxdb, pgsql]")
    return cfg


def _read_file(url):
    if url.startswith("file://"):
        with open(_remove_prefix(url, "file://"), "r") as file:
            return file.read()
    elif url.startswith("oss://"):  # TODO: may not be correct
        bucket, key = get_bucket_from_oss_url(url)
        return bucket.get_object(key).read()


def _remove_prefix(text: str, prefix: str):
    return text[text.startswith(prefix) and len(prefix) :]


def get_connection(url: str):
    """connect to pgsql using configs in feature_store.yml

    Args:
        url (str): url of .yml
    """
    cfg = _read_yml(url)


def get_forecast_cfg(url: str):
    """get forecast config like length of look_back and look_forward, features and labels

    Args:
        url (str): url of .yml
    """
    cfg = _read_yml(url)
    forecast_cfg = ForecastConfig(cfg)


def get_entity_cfg(url: str):
    """get entity config for join

    Args:
        url (str): url of .yml
    """
    cfg = _read_yml(url)
    entity_cfg = EntityConfig(cfg)
