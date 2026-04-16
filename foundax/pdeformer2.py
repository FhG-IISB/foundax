import importlib

from ._vendors import ensure_repo_on_path


def small(*args, **kwargs):
    ensure_repo_on_path("jax_pdeformer2")
    mod = importlib.import_module("jax_pdeformer2")
    return mod.create_pdeformer_from_config(
        {"model": mod.PDEFORMER_SMALL_CONFIG}, *args, **kwargs
    )


def base(*args, **kwargs):
    ensure_repo_on_path("jax_pdeformer2")
    mod = importlib.import_module("jax_pdeformer2")
    return mod.create_pdeformer_from_config(
        {"model": mod.PDEFORMER_BASE_CONFIG}, *args, **kwargs
    )


def fast(*args, **kwargs):
    ensure_repo_on_path("jax_pdeformer2")
    mod = importlib.import_module("jax_pdeformer2")
    return mod.create_pdeformer_from_config(
        {"model": mod.PDEFORMER_FAST_CONFIG}, *args, **kwargs
    )


__all__ = ["small", "base", "fast"]
