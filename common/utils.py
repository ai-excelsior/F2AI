from .oss_utils import get_bucket_from_oss_url
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
        with open(_remove_prefix(url, "file://"), "r") as file:
            return file.read()
    elif url.startswith("oss://"):  # TODO: may not be correct
        bucket, key = get_bucket_from_oss_url(url)
        return bucket.get_object(key).read()


def _remove_prefix(text: str, prefix: str):
    return text[text.startswith(prefix) and len(prefix) :]
