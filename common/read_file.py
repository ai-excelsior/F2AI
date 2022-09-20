from .oss_utils import get_bucket_from_oss_url
from .utils import remove_prefix
import yaml


def read_yml(url: str):
    """read .yml file for following execute

    Args:
        url (str): url of .yml
    """
    file = _read_file(url)
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    return cfg


def _read_file(url):
    if url.startswith("file://"):
        with open(remove_prefix(url, "file://"), "r") as file:
            return file.read()
    elif url.startswith("oss://"):  # TODO: may not be correct
        bucket, key = get_bucket_from_oss_url(url)
        return bucket.get_object(key).read()
