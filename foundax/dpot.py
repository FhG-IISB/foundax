import importlib

from ._vendors import ensure_repo_on_path


def Ti(*args, **kwargs):
    ensure_repo_on_path("jax_dpot")
    return importlib.import_module("jax_dpot").dpot_ti(*args, **kwargs)


def S(*args, **kwargs):
    ensure_repo_on_path("jax_dpot")
    return importlib.import_module("jax_dpot").dpot_s(*args, **kwargs)


def M(*args, **kwargs):
    ensure_repo_on_path("jax_dpot")
    return importlib.import_module("jax_dpot").dpot_m(*args, **kwargs)


def L(*args, **kwargs):
    ensure_repo_on_path("jax_dpot")
    return importlib.import_module("jax_dpot").dpot_l(*args, **kwargs)


def H(*args, **kwargs):
    ensure_repo_on_path("jax_dpot")
    return importlib.import_module("jax_dpot").dpot_h(*args, **kwargs)


ti = Ti
s = S
m = M
l = L
h = H


__all__ = ["Ti", "S", "M", "L", "H", "ti", "s", "m", "l", "h"]
