import sys
from pathlib import Path


def _repos_root():
    return Path(__file__).resolve().parents[1] / "repos"


def ensure_repo_on_path(repo_name):
    repo_path = _repos_root() / repo_name
    if not repo_path.exists():
        raise ImportError(
            "Vendored repository '{}' not found at {}".format(repo_name, repo_path)
        )

    repo_path_str = str(repo_path)
    if repo_path_str not in sys.path:
        sys.path.insert(0, repo_path_str)

    return repo_path
