import subprocess


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
        check=False,
    )


def git_reset(cwd: str, branch: str, mode: str = "hard"):
    return subprocess.run(["git", "reset", f"--{mode}", branch], cwd=cwd, check=True)


def git_clean(cwd: str):
    return subprocess.run(["git", "clean", "-df"], cwd=cwd, check=True)


def git_pull(cwd: str, branch: str):
    return subprocess.run(["git", "pull", "--rebase", "origin", branch], cwd=cwd, check=True)
