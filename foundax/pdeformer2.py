"""PDEformer-2 — Foundation Model for Two-Dimensional PDEs.

Lazy-loading factories for the vendored ``jax_pdeformer2`` package.

**Paper:** Shi et al., *"PDEformer-2: A Foundation Model for
Two-Dimensional PDEs"* (2025)
https://arxiv.org/abs/2502.14844

**Code:** https://github.com/functoreality/pdeformer-2

Architecture: Graphormer encoder + Implicit Neural Representation (INR)
decoder with a hyper-network bridge.  Three predefined configs are
provided (Small, Base, Fast) that differ mainly in Graphormer width
and INR hidden dimension.

Key config dimensions:

=========  ===========  ===========  =========  ===========  =======
Variant    graph_layers graph_embed  graph_ffn  inr_hidden   ~Params
=========  ===========  ===========  =========  ===========  =======
Small      9            512          1024       128          ~27.7 M
Base       12           768          1536       768          larger
Fast       12           768          1536       256          larger
=========  ===========  ===========  =========  ===========  =======

Base and Fast share the same Graphormer size; they differ in
``inr.dim_hidden`` (768 vs 256), trading accuracy for speed.
"""

import importlib
from typing import Any

from ._vendors import ensure_repo_on_path


def small(dtype: Any = None) -> Any:
    """Create PDEformer-2 with the predefined **Small** config (~27.7 M params).

    Delegates to ``jax_pdeformer2.create_pdeformer_from_config`` with
    ``PDEFORMER_SMALL_CONFIG``.

    Small config highlights::

        graphormer:  layers=9, embed=512, ffn=1024, heads=32
        inr:         type=poly_inr, layers=12, hidden=128, act=sin
        hypernet:    hidden=512, layers=2
        function_encoder: cnn2dv3, 4 branches, resolution=128

    Parameters
    ----------
    dtype : jnp.dtype, optional
        JAX data type for model computations. Defaults to ``jnp.float32``.

    Returns
    -------
    PDEformer
        Uninitialised Flax ``nn.Module``.

    References
    ----------
    Shi et al., "PDEformer-2", 2025.  https://arxiv.org/abs/2502.14844
    """
    ensure_repo_on_path("jax_pdeformer2")
    mod = importlib.import_module("jax_pdeformer2")
    kw = {}
    if dtype is not None:
        kw["dtype"] = dtype
    return mod.create_pdeformer_from_config(
        {"model": mod.PDEFORMER_SMALL_CONFIG}, **kw
    )


def base(dtype: Any = None) -> Any:
    """Create PDEformer-2 with the predefined **Base** config.

    Delegates to ``jax_pdeformer2.create_pdeformer_from_config`` with
    ``PDEFORMER_BASE_CONFIG``.

    Base config highlights (overrides Small defaults)::

        graphormer:  layers=12, embed=768, ffn=1536, heads=32
        inr:         hidden=768, layers=12
        hypernet:    hidden=512

    Parameters
    ----------
    dtype : jnp.dtype, optional
        JAX data type for model computations. Defaults to ``jnp.float32``.

    Returns
    -------
    PDEformer
        Uninitialised Flax ``nn.Module``.

    References
    ----------
    Shi et al., "PDEformer-2", 2025.  https://arxiv.org/abs/2502.14844
    """
    ensure_repo_on_path("jax_pdeformer2")
    mod = importlib.import_module("jax_pdeformer2")
    kw = {}
    if dtype is not None:
        kw["dtype"] = dtype
    return mod.create_pdeformer_from_config(
        {"model": mod.PDEFORMER_BASE_CONFIG}, **kw
    )


def fast(dtype: Any = None) -> Any:
    """Create PDEformer-2 with the predefined **Fast** config.

    Delegates to ``jax_pdeformer2.create_pdeformer_from_config`` with
    ``PDEFORMER_FAST_CONFIG``.

    Fast config highlights (overrides Small defaults)::

        graphormer:  layers=12, embed=768, ffn=1536, heads=32
        inr:         hidden=256, layers=12   (smaller than Base → faster)
        hypernet:    hidden=512

    Parameters
    ----------
    dtype : jnp.dtype, optional
        JAX data type for model computations. Defaults to ``jnp.float32``.

    Returns
    -------
    PDEformer
        Uninitialised Flax ``nn.Module``.

    References
    ----------
    Shi et al., "PDEformer-2", 2025.  https://arxiv.org/abs/2502.14844
    """
    ensure_repo_on_path("jax_pdeformer2")
    mod = importlib.import_module("jax_pdeformer2")
    kw = {}
    if dtype is not None:
        kw["dtype"] = dtype
    return mod.create_pdeformer_from_config(
        {"model": mod.PDEFORMER_FAST_CONFIG}, **kw
    )


__all__ = ["small", "base", "fast"]
