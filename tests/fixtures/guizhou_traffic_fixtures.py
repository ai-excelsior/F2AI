import pytest
import os

from .git_utils import git_clean, git_clone, git_reset, git_pull
from .constants import TEMP_DIR


GUIZHOU_TRAFFIC_CFG = {
    "repo": "git@code.unianalysis.com:f2ai-examples/guizhou_traffic.git",
    "infra": {
        "file": {
            "cwd": os.path.join(TEMP_DIR, "f2ai-credit-guizhou_traffic_file"),
            "branch": "main",
        },
        "pgsql": {
            "cwd": os.path.join(TEMP_DIR, "f2ai-guizhou_traffic_pgsql"),
            "branch": "ver_pgsql",
        },
    },
}


@pytest.fixture(scope="session")
def make_guizhou_traffic():
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    def get_guizhou_traffic(infra="file"):
        repo = GUIZHOU_TRAFFIC_CFG["repo"]
        infra = GUIZHOU_TRAFFIC_CFG["infra"][infra]
        cwd = infra["cwd"]
        branch = infra["branch"]

        # clone repo to cwd
        if not os.path.isdir(cwd):
            git_clone(cwd, repo, branch)

        # reset, clean and update to latest
        git_reset(cwd, branch)
        git_clean(cwd)
        git_pull(cwd, branch)

        return cwd

    yield get_guizhou_traffic
