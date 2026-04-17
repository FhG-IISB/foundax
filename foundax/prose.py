"""PROSE -- JAX translation of the PROSE model family.

**Paper:** Lample & Charton, *"PROSE"* (2024)

Architecture: Transformer-based sequence-to-sequence models for
operator learning from finite-difference, ODE, and PDE data.

All factories return an ``equinox.Module`` directly.

Usage::

    model = foundax.prose.fd_1to1(x_num=64)
    model = foundax.prose.pde_2to1(n_words=100, pad_index=0)
"""

import importlib

from ._vendors import ensure_repo_on_path


def fd_1to1(
    x_num=128,
    max_output_dim=4,
    output_len=10,
    *,
    key=None,
):
    """PROSE finite-difference 1-to-1 model (Equinox).

    Returns an ``eqx.Module``.
    """
    import jax
    ensure_repo_on_path("jax_prose")
    mod = importlib.import_module("jax_prose.model_eqx")
    if key is None:
        key = jax.random.PRNGKey(0)
    return mod.PROSE1to1(x_num=x_num, max_output_dim=max_output_dim, output_len=output_len, key=key)


def fd_2to1(
    n_words,
    x_num=128,
    max_output_dim=4,
    *,
    key=None,
):
    """PROSE finite-difference 2-to-1 model (Equinox).

    Returns an ``eqx.Module``.
    """
    import jax
    ensure_repo_on_path("jax_prose")
    mod = importlib.import_module("jax_prose.model_eqx")
    if key is None:
        key = jax.random.PRNGKey(0)
    return mod.PROSE2to1(n_words=n_words, x_num=x_num, max_output_dim=max_output_dim, key=key)


def ode_2to1(
    n_words,
    pad_index,
    max_output_dimension=3,
    *,
    key=None,
):
    """PROSE ODE 2-to-1 model (Equinox).

    Returns an ``eqx.Module``.
    """
    import jax
    ensure_repo_on_path("jax_prose")
    mod = importlib.import_module("jax_prose.model_eqx")
    if key is None:
        key = jax.random.PRNGKey(0)
    return mod.PROSEODE2to1(n_words=n_words, pad_index=pad_index, max_output_dimension=max_output_dimension, key=key)


def pde_2to1(
    n_words,
    pad_index,
    max_output_dimension=1,
    x_patch_size=1,
    x_grid_size=128,
    *,
    key=None,
):
    """PROSE PDE 2-to-1 model (Equinox).

    Returns an ``eqx.Module``.
    """
    import jax
    ensure_repo_on_path("jax_prose")
    mod = importlib.import_module("jax_prose.model_eqx")
    if key is None:
        key = jax.random.PRNGKey(0)
    return mod.PROSEPDE2to1(
        n_words=n_words,
        pad_index=pad_index,
        max_output_dimension=max_output_dimension,
        x_patch_size=x_patch_size,
        x_grid_size=x_grid_size,
        key=key,
    )


__all__ = ["fd_1to1", "fd_2to1", "ode_2to1", "pde_2to1"]
