import importlib

from ._vendors import ensure_repo_on_path


def Ti(*args, **kwargs):
    ensure_repo_on_path("jax_morph")
    return importlib.import_module("jax_morph").morph_Ti(*args, **kwargs)


def S(*args, **kwargs):
    ensure_repo_on_path("jax_morph")
    return importlib.import_module("jax_morph").morph_S(*args, **kwargs)


def M(*args, **kwargs):
    ensure_repo_on_path("jax_morph")
    return importlib.import_module("jax_morph").morph_M(*args, **kwargs)


def L(*args, **kwargs):
    ensure_repo_on_path("jax_morph")
    return importlib.import_module("jax_morph").morph_L(*args, **kwargs)


ti = Ti
s = S
m = M
l = L


__all__ = ["Ti", "S", "M", "L", "ti", "s", "m", "l"]
