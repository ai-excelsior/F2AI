import yaml
from f2ai.definitions import init_offline_store_from_cfg

offline_file_store_yaml_string = """
type: file
"""


def test_init_file_offline_store():
    from f2ai.offline_stores.offline_file_store import OfflineFileStore

    offline_store = init_offline_store_from_cfg(yaml.safe_load(offline_file_store_yaml_string), 'test')
    assert isinstance(offline_store, OfflineFileStore)
