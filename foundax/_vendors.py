import importlib
import sys
from pathlib import Path


def _repos_root():
    return Path(__file__).resolve().parents[1] / "repos"


def ensure_repo_on_path(repo_name):
    repo_path = _repos_root() / repo_name
    if repo_path.exists():
        repo_path_str = str(repo_path)
        if repo_path_str not in sys.path:
            sys.path.insert(0, repo_path_str)
        return repo_path

    # Wheel / PyPI install: vendored packages are shipped as top-level
    # packages in site-packages, so they should already be importable.
    try:
        importlib.import_module(repo_name)
    except ImportError:
        raise ImportError(
            "Vendored package '{}' is not installed and the local "
            "repository was not found at {}".format(repo_name, repo_path)
        ) from None

    return None
