import importlib

from ._vendors import ensure_repo_on_path


def Ti(*args, **kwargs):
    ensure_repo_on_path("jax_mpp")
    return importlib.import_module("jax_mpp").avit_Ti(*args, **kwargs)


def S(*args, **kwargs):
    ensure_repo_on_path("jax_mpp")
    return importlib.import_module("jax_mpp").avit_S(*args, **kwargs)


def B(*args, **kwargs):
    ensure_repo_on_path("jax_mpp")
    return importlib.import_module("jax_mpp").avit_B(*args, **kwargs)


def L(*args, **kwargs):
    ensure_repo_on_path("jax_mpp")
    return importlib.import_module("jax_mpp").avit_L(*args, **kwargs)


ti = Ti
s = S
b = B
l = L


__all__ = ["Ti", "S", "B", "L", "ti", "s", "b", "l"]
