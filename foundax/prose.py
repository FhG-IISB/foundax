"""PROSE — JAX/Flax translation of the PROSE-FD model family.

Lazy-loading factories for the vendored ``jax_prose`` package.

Architecture: Transformer-based sequence-to-sequence models for
operator learning from finite-difference, ODE, and PDE data.

Unlike most other foundax models these factories **return
``(model, params)``** — i.e. they already initialise parameters.
"""

import importlib
from typing import Any, Optional, Tuple

from ._vendors import ensure_repo_on_path


def fd_1to1(
    config: Optional[Any] = None,
    x_num: int = 128,
    max_output_dim: int = 4,
    input_len: int = 10,
    output_len: int = 10,
) -> Any:
    """Create PROSE finite-difference 1-to-1 model.

    Delegates to ``jax_prose.prose_fd_1to1``.

    Parameters
    ----------
    config : PROSE1to1Config, optional
        Full model config.  Uses internal defaults when ``None``.
    x_num : int, default 128
        Spatial grid size.
    max_output_dim : int, default 4
        Maximum output dimensionality.
    input_len : int, default 10
        Input sequence length (time-steps).
    output_len : int, default 10
        Output sequence length (time-steps).

    Returns
    -------
    tuple[PROSE1to1, dict]
        ``(model, params)`` — the model **and** its initialised
        parameters.
    """
    ensure_repo_on_path("jax_prose")
    return importlib.import_module("jax_prose").prose_fd_1to1(
        config=config,
        x_num=x_num,
        max_output_dim=max_output_dim,
        input_len=input_len,
        output_len=output_len,
    )


def fd_2to1(
    n_words: int,
    x_num: int = 128,
    max_output_dim: int = 4,
    input_len: int = 10,
    output_len: int = 10,
    symbol_len: int = 48,
) -> Any:
    """Create PROSE finite-difference 2-to-1 model.

    Delegates to ``jax_prose.prose_fd_2to1``.

    Parameters
    ----------
    n_words : int
        Vocabulary size (**required**).
    x_num : int, default 128
        Spatial grid size.
    max_output_dim : int, default 4
        Maximum output dimensionality.
    input_len : int, default 10
        Input sequence length.
    output_len : int, default 10
        Output sequence length.
    symbol_len : int, default 48
        Symbol / equation token sequence length.

    Returns
    -------
    tuple[PROSE2to1, dict]
        ``(model, params)``.
    """
    ensure_repo_on_path("jax_prose")
    return importlib.import_module("jax_prose").prose_fd_2to1(
        n_words=n_words,
        x_num=x_num,
        max_output_dim=max_output_dim,
        input_len=input_len,
        output_len=output_len,
        symbol_len=symbol_len,
    )


def ode_2to1(
    n_words: int,
    pad_index: int,
    max_output_dimension: int = 3,
    input_len: int = 50,
    output_len: int = 50,
    text_len: int = 48,
    cfg: Optional[Any] = None,
) -> Any:
    """Create PROSE ODE 2-to-1 model.

    Delegates to ``jax_prose.prose_ode_2to1``.

    Parameters
    ----------
    n_words : int
        Vocabulary size (**required**).
    pad_index : int
        Padding token index (**required**).
    max_output_dimension : int, default 3
        Maximum output dimension.
    input_len : int, default 50
        Input sequence length.
    output_len : int, default 50
        Output sequence length.
    text_len : int, default 48
        Text / symbol sequence length.
    cfg : ProseTextData2to1Config, optional
        Full model config.

    Returns
    -------
    tuple[PROSEODE2to1, dict]
        ``(model, params)``.
    """
    ensure_repo_on_path("jax_prose")
    return importlib.import_module("jax_prose").prose_ode_2to1(
        n_words=n_words,
        pad_index=pad_index,
        max_output_dimension=max_output_dimension,
        input_len=input_len,
        output_len=output_len,
        text_len=text_len,
        cfg=cfg,
    )


def pde_2to1(
    n_words: int,
    pad_index: int,
    max_output_dimension: int = 1,
    x_patch_size: int = 1,
    x_grid_size: int = 128,
    input_len: int = 10,
    output_len: int = 10,
    text_len: int = 48,
    cfg: Optional[Any] = None,
) -> Any:
    """Create PROSE PDE 2-to-1 model.

    Delegates to ``jax_prose.prose_pde_2to1``.

    Parameters
    ----------
    n_words : int
        Vocabulary size (**required**).
    pad_index : int
        Padding token index (**required**).
    max_output_dimension : int, default 1
        Maximum output dimension.
    x_patch_size : int, default 1
        Spatial patch size.
    x_grid_size : int, default 128
        Spatial grid size.
    input_len : int, default 10
        Input sequence length.
    output_len : int, default 10
        Output sequence length.
    text_len : int, default 48
        Text / symbol sequence length.
    cfg : ProseTextData2to1Config, optional
        Full model config.

    Returns
    -------
    tuple[PROSEPDE2to1, dict]
        ``(model, params)``.
    """
    ensure_repo_on_path("jax_prose")
    return importlib.import_module("jax_prose").prose_pde_2to1(
        n_words=n_words,
        pad_index=pad_index,
        max_output_dimension=max_output_dimension,
        x_patch_size=x_patch_size,
        x_grid_size=x_grid_size,
        input_len=input_len,
        output_len=output_len,
        text_len=text_len,
        cfg=cfg,
    )


__all__ = ["fd_1to1", "fd_2to1", "ode_2to1", "pde_2to1"]
