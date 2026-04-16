import importlib

from ._vendors import ensure_repo_on_path


def T(*args, **kwargs):
    ensure_repo_on_path("jax_poseidon")
    return importlib.import_module("jax_poseidon").poseidonT(*args, **kwargs)


def B(*args, **kwargs):
    ensure_repo_on_path("jax_poseidon")
    return importlib.import_module("jax_poseidon").poseidonB(*args, **kwargs)


def L(*args, **kwargs):
    ensure_repo_on_path("jax_poseidon")
    return importlib.import_module("jax_poseidon").poseidonL(*args, **kwargs)


t = T
b = B
l = L


__all__ = ["T", "B", "L", "t", "b", "l"]
