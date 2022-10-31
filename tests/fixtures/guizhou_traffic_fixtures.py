import pytest
import subprocess
import os

CREDIT_SCORE_CFG = {
    "repo": "git@code.unianalysis.com:f2ai-examples/guizhou_traffic.git",
    "infra": {
        "file": {
            "cwd": "/tmp/f2ai-credit-guizhou_traffic_file",
            "branch": "main",
        },
        "pgsql": {
            "cwd": "/tmp/f2ai-guizhou_traffic_pgsql",
            "branch": "ver_pgsql",
        },
    },
}


def git_clone(cwd: str, repo: str, branch: str):
    return subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            branch,
            repo,
            cwd,
        ],
        check=True,
    )


def git_reset(cwd: str, branch: str, mode: str = "hard"):
    return subprocess.run(["git", "reset", f"--{mode}", branch], cwd=cwd, check=True)


def git_clean(cwd: str):
    return subprocess.run(["git", "clean", "-df"], cwd=cwd, check=True)


def git_pull(cwd: str, branch: str):
    return subprocess.run(["git", "pull", "--rebase", "origin", branch], cwd=cwd, check=True)


@pytest.fixture(scope="session")
def make_guizhou_traffic():
    def get_guizhou_traffic(infra="file"):
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

    yield get_guizhou_traffic
