"""MPP -- Multiple Physics Pretraining for Physical Surrogate Models.

**Paper:** McCabe et al., *"Multiple Physics Pretraining"* (NeurIPS 2024)
https://openreview.net/forum?id=DKSI3bULiZ

Architecture: ``AViT`` (Adaptive Vision Transformer) with variable-
resolution patching for multi-physics operator learning.

Usage::

    model = foundax.mpp.Ti(n_states=6)
    model = foundax.mpp(embed_dim=192, num_heads=3, processor_blocks=12)
"""

import importlib

from ._vendors import ensure_repo_on_path
from . import _callable_module

_AdaptedAViT = None


def _adapted_cls():
    """Return a cached AViT subclass with (B, T, H, W, C) convention."""
    global _AdaptedAViT
    if _AdaptedAViT is not None:
        return _AdaptedAViT
    import jax.numpy as jnp

    ensure_repo_on_path("jax_mpp")
    _mod = importlib.import_module("jax_mpp.avit_eqx")

    class _Adapted(_mod.AViT):
        def __call__(self, x, state_labels, bcs, deterministic=True):
            # (B, T, H, W, C) -> (T, B, C, H, W)
            x = jnp.transpose(x, (1, 0, 4, 2, 3))
            out = super().__call__(x, state_labels, bcs, deterministic)
            # (B, C, H, W) -> (B, H, W, C)
            return jnp.transpose(out, (0, 2, 3, 1))

    _AdaptedAViT = _Adapted
    return _AdaptedAViT


def _build(
    patch_size=(16, 16),
    embed_dim=768,
    processor_blocks=8,
    n_states=6,
    drop_path=0.2,
    bias_type="rel",
    num_heads=12,
    *,
    key=None,
):
    import jax

    cls = _adapted_cls()
    if key is None:
        key = jax.random.PRNGKey(0)
    kw = {k: v for k, v in locals().items() if k not in ("cls", "jax")}
    return cls(**kw)


def Ti(
    patch_size=(16, 16),
    embed_dim=192,
    processor_blocks=12,
    n_states=12,
    drop_path=0.2,
    bias_type="rel",
    num_heads=3,
):
    """AViT-Tiny ~5.5 M params.

    Adaptive Vision Transformer with variable-resolution patching for
    multi-physics operator learning (MPP).

    Reference:
        McCabe et al., *Multiple Physics Pretraining for Physical
        Surrogate Models* (NeurIPS 2024).
        https://openreview.net/forum?id=DKSI3bULiZ

    Example::

        model = foundax.mpp.Ti(n_states=6)

    Shape:
        - Input: ``(B, T, H, W, C)`` + state_labels ``(C,)``
        - Output: ``(B, H, W, C)``

    See Also:
        :func:`S`, :func:`B`, :func:`L`

    Args:
        patch_size: Spatial patch dimensions ``(H, W)``.
        embed_dim: Token embedding dimension.
        processor_blocks: Number of space-time transformer blocks.
        n_states: Number of active physical state variables (channels).
        drop_path: Stochastic depth rate.
        bias_type: Attention bias type (``"rel"`` = relative position
            bias).
        num_heads: Number of attention heads.

    Returns:
        An ``equinox.Module`` (AViT).
    """
    return _build(**{k: v for k, v in locals().items()})


def S(
    patch_size=(16, 16),
    embed_dim=384,
    processor_blocks=12,
    n_states=12,
    drop_path=0.2,
    bias_type="rel",
    num_heads=6,
):
    """AViT-Small ~21 M params.

    Adaptive Vision Transformer with variable-resolution patching for
    multi-physics operator learning (MPP).

    Reference:
        McCabe et al., *Multiple Physics Pretraining for Physical
        Surrogate Models* (NeurIPS 2024).
        https://openreview.net/forum?id=DKSI3bULiZ

    Example::

        model = foundax.mpp.S(n_states=6)

    Shape:
        - Input: ``(B, T, H, W, C)`` + state_labels ``(C,)``
        - Output: ``(B, H, W, C)``

    See Also:
        :func:`Ti`, :func:`B`, :func:`L`

    Args:
        patch_size: Spatial patch dimensions ``(H, W)``.
        embed_dim: Token embedding dimension.
        processor_blocks: Number of space-time transformer blocks.
        n_states: Number of active physical state variables (channels).
        drop_path: Stochastic depth rate.
        bias_type: Attention bias type (``"rel"`` = relative position
            bias).
        num_heads: Number of attention heads.

    Returns:
        An ``equinox.Module`` (AViT).
    """
    return _build(**{k: v for k, v in locals().items()})


def B(
    patch_size=(16, 16),
    embed_dim=768,
    processor_blocks=12,
    n_states=12,
    drop_path=0.2,
    bias_type="rel",
    num_heads=12,
):
    """AViT-Base ~83 M params.

    Adaptive Vision Transformer with variable-resolution patching for
    multi-physics operator learning (MPP).

    Reference:
        McCabe et al., *Multiple Physics Pretraining for Physical
        Surrogate Models* (NeurIPS 2024).
        https://openreview.net/forum?id=DKSI3bULiZ

    Example::

        model = foundax.mpp.B(n_states=6)

    Shape:
        - Input: ``(B, T, H, W, C)`` + state_labels ``(C,)``
        - Output: ``(B, H, W, C)``

    See Also:
        :func:`Ti`, :func:`S`, :func:`L`

    Args:
        patch_size: Spatial patch dimensions ``(H, W)``.
        embed_dim: Token embedding dimension.
        processor_blocks: Number of space-time transformer blocks.
        n_states: Number of active physical state variables (channels).
        drop_path: Stochastic depth rate.
        bias_type: Attention bias type (``"rel"`` = relative position
            bias).
        num_heads: Number of attention heads.

    Returns:
        An ``equinox.Module`` (AViT).
    """
    return _build(**{k: v for k, v in locals().items()})


def L(
    patch_size=(16, 16),
    embed_dim=1024,
    processor_blocks=24,
    n_states=12,
    drop_path=0.2,
    bias_type="rel",
    num_heads=16,
):
    """AViT-Large ~300 M params.

    Adaptive Vision Transformer with variable-resolution patching for
    multi-physics operator learning (MPP).

    Reference:
        McCabe et al., *Multiple Physics Pretraining for Physical
        Surrogate Models* (NeurIPS 2024).
        https://openreview.net/forum?id=DKSI3bULiZ

    Example::

        model = foundax.mpp.L(n_states=6)

    Shape:
        - Input: ``(B, T, H, W, C)`` + state_labels ``(C,)``
        - Output: ``(B, H, W, C)``

    See Also:
        :func:`Ti`, :func:`S`, :func:`B`

    Args:
        patch_size: Spatial patch dimensions ``(H, W)``.
        embed_dim: Token embedding dimension.
        processor_blocks: Number of space-time transformer blocks.
        n_states: Number of active physical state variables (channels).
        drop_path: Stochastic depth rate.
        bias_type: Attention bias type (``"rel"`` = relative position
            bias).
        num_heads: Number of attention heads.

    Returns:
        An ``equinox.Module`` (AViT).
    """
    return _build(**{k: v for k, v in locals().items()})


ti = Ti
s = S
b = B
l = L

__all__ = ["Ti", "S", "B", "L", "ti", "s", "b", "l"]

_callable_module.install(__name__, _build)
