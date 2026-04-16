import importlib

from ._vendors import ensure_repo_on_path


def base(*args, **kwargs):
    ensure_repo_on_path("jax_walrus")
    return importlib.import_module("jax_walrus").IsotropicModel(*args, **kwargs)


def v1(*args, **kwargs):
    return base(*args, **kwargs)


default = base


__all__ = ["base", "v1", "default"]
