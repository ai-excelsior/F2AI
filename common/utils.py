from datacfg import ForecastConfig, EntityConfig


def _read_yml(url: str):
    """read .yml file for following execute

    Args:
        url (str): url of .yml
    """


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
