import functools
import os
from typing import Tuple, Mapping, Callable, Optional, IO, Any
import oss2
import stat
import tempfile
import zipfile
from ignite.handlers import DiskSaver
from .utils import remove_prefix


@functools.lru_cache(maxsize=64)
def get_bucket(bucket, endpoint=None):
    key_id = os.environ.get("OSS_ACCESS_KEY_ID")
    key_secret = os.environ.get("OSS_ACCESS_KEY_SECRET")
    endpoint = endpoint or os.environ.get("OSS_ENDPOINT")

    return oss2.Bucket(oss2.Auth(key_id, key_secret), endpoint, bucket)


def parse_oss_url(url: str) -> Tuple[str, str, str]:
    """
    url format:  oss://{bucket}/{key}
    """
    url = remove_prefix(url, "oss://")
    components = url.split("/")
    return components[0], "/".join(components[1:])


def get_bucket_from_oss_url(url: str):
    bucket_name, key = parse_oss_url(url)
    return get_bucket(bucket_name), key


@functools.lru_cache(maxsize=1)
def get_pandas_storage_options():
    key_id = os.environ.get("OSS_ACCESS_KEY_ID")
    key_secret = os.environ.get("OSS_ACCESS_KEY_SECRET")
    endpoint = os.environ.get("OSS_ENDPOINT")

    if not endpoint.startswith("https://"):
        endpoint = f"https://{endpoint}"

    return {
        "key": key_id,
        "secret": key_secret,
        "client_kwargs": {
            "endpoint_url": endpoint,
        },
        # "config_kwargs": {"s3": {"addressing_style": "virtual"}},
    }


class DiskAndOssSaverAdd(DiskSaver):
    def __init__(
        self,
        dirname: str,
        ossaddress: str = None,
        create_dir: bool = True,
        require_empty: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            dirname=dirname, atomic=True, create_dir=create_dir, require_empty=require_empty, **kwargs
        )
        self.ossaddress = ossaddress

    def _save_func(self, checkpoint: Mapping, path: str, func: Callable, rank: int = 0) -> None:
        tmp: Optional[IO[bytes]] = None
        if rank == 0:
            tmp = tempfile.NamedTemporaryFile(delete=False, dir=self.dirname)

        try:
            func(checkpoint, tmp.file, **self.kwargs)
        except BaseException:
            if tmp is not None:
                tmp.close()
                os.remove(tmp.name)
                raise

        if tmp is not None:
            tmp.close()
            os.replace(tmp.name, path)
            # append group/others read mode
            os.chmod(path, os.stat(path).st_mode | stat.S_IRGRP | stat.S_IROTH)
            if self.ossaddress:
                bucket, key = get_bucket_from_oss_url(self.ossaddress)
                state_file = f"{path.rsplit('/',maxsplit=1)[0]}{os.sep}state.json"
                file_path = f"{path.rsplit('/',maxsplit=1)[0]}{os.sep}model.zip"
                with zipfile.ZipFile(file_path, "w", compression=zipfile.ZIP_BZIP2) as archive:
                    archive.write(path, os.path.basename(path))
                    archive.write(state_file, os.path.basename(state_file))
                bucket.put_object_from_file(key, file_path)
                os.remove(file_path)
