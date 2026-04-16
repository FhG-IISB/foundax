import importlib

from ._vendors import ensure_repo_on_path


def fd_1to1(*args, **kwargs):
    ensure_repo_on_path("jax_prose")
    return importlib.import_module("jax_prose").prose_fd_1to1(*args, **kwargs)


def fd_2to1(*args, **kwargs):
    ensure_repo_on_path("jax_prose")
    return importlib.import_module("jax_prose").prose_fd_2to1(*args, **kwargs)


def ode_2to1(*args, **kwargs):
    ensure_repo_on_path("jax_prose")
    return importlib.import_module("jax_prose").prose_ode_2to1(*args, **kwargs)


def pde_2to1(*args, **kwargs):
    ensure_repo_on_path("jax_prose")
    return importlib.import_module("jax_prose").prose_pde_2to1(*args, **kwargs)


__all__ = ["fd_1to1", "fd_2to1", "ode_2to1", "pde_2to1"]
