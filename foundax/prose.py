"""PROSE -- JAX/Flax translation of the PROSE-FD model family.

**Paper:** Lample & Charton, *"PROSE"* (2024)

Architecture: Transformer-based sequence-to-sequence models for
operator learning from finite-difference, ODE, and PDE data.

Unlike most other foundax models these factories **return
``(model, params)``** -- i.e. they already initialise parameters.

Usage::

    model, params = foundax.prose.fd_1to1(x_num=64)
    model, params = foundax.prose.pde_2to1(n_words=100, pad_index=0)
"""

import importlib

from ._vendors import ensure_repo_on_path


def fd_1to1(
    config=None,
    x_num=128,
    max_output_dim=4,
    input_len=10,
    output_len=10,
):
    """PROSE finite-difference 1-to-1 model.

    Returns ``(model, params)``.
    """
    ensure_repo_on_path("jax_prose")
    mod = importlib.import_module("jax_prose")
    return mod.prose_fd_1to1(**{k: v for k, v in locals().items() if k != "mod"})


def fd_2to1(
    n_words,
    x_num=128,
    max_output_dim=4,
    input_len=10,
    output_len=10,
    symbol_len=48,
):
    """PROSE finite-difference 2-to-1 model.

    Returns ``(model, params)``.
    """
    ensure_repo_on_path("jax_prose")
    mod = importlib.import_module("jax_prose")
    return mod.prose_fd_2to1(**{k: v for k, v in locals().items() if k != "mod"})


def ode_2to1(
    n_words,
    pad_index,
    max_output_dimension=3,
    input_len=50,
    output_len=50,
    text_len=48,
    cfg=None,
):
    """PROSE ODE 2-to-1 model.

    Returns ``(model, params)``.
    """
    ensure_repo_on_path("jax_prose")
    mod = importlib.import_module("jax_prose")
    return mod.prose_ode_2to1(**{k: v for k, v in locals().items() if k != "mod"})


def pde_2to1(
    n_words,
    pad_index,
    max_output_dimension=1,
    x_patch_size=1,
    x_grid_size=128,
    input_len=10,
    output_len=10,
    text_len=48,
    cfg=None,
):
    """PROSE PDE 2-to-1 model.

    Returns ``(model, params)``.
    """
    ensure_repo_on_path("jax_prose")
    mod = importlib.import_module("jax_prose")
    return mod.prose_pde_2to1(**{k: v for k, v in locals().items() if k != "mod"})


__all__ = ["fd_1to1", "fd_2to1", "ode_2to1", "pde_2to1"]
