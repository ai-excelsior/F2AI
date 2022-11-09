import pytest
import os

from .git_utils import git_clean, git_clone, git_reset, git_pull
from .constants import TEMP_DIR

CREDIT_SCORE_CFG = {
    "repo": "git@code.unianalysis.com:f2ai-examples/f2ai-credit-scoring.git",
    "infra": {
        "file": {
            "cwd": os.path.join(TEMP_DIR, "f2ai-credit-scoring_file"),
            "branch": "main",
        },
        "pgsql": {
            "cwd": os.path.join(TEMP_DIR, "f2ai-credit-scoring_pgsql"),
            "branch": "ver_pgsql",
        },
    },
}


@pytest.fixture(scope="session")
def make_credit_score():
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    def get_credit_score(infra="file"):
        repo = CREDIT_SCORE_CFG["repo"]
        infra = CREDIT_SCORE_CFG["infra"][infra]
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

    yield get_credit_score
