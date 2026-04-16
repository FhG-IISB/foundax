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


def _build(
    patch_size=(16, 16),
    embed_dim=768,
    processor_blocks=8,
    n_states=6,
    drop_path=0.2,
    bias_type="rel",
    num_heads=12,
):
    ensure_repo_on_path("jax_mpp")
    mod = importlib.import_module("jax_mpp")
    return mod.AViT(**{k: v for k, v in locals().items() if k != "mod"})


def Ti(
    patch_size=(16, 16),
    embed_dim=192,
    processor_blocks=12,
    n_states=12,
    drop_path=0.2,
    bias_type="rel",
    num_heads=3,
):
    """AViT-Tiny ~5.5 M params. embed=192, heads=3, blocks=12."""
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
    """AViT-Small ~21 M params. embed=384, heads=6, blocks=12."""
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
    """AViT-Base ~83 M params. embed=768, heads=12, blocks=12."""
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
    """AViT-Large ~300 M params. embed=1024, heads=16, blocks=24."""
    return _build(**{k: v for k, v in locals().items()})


ti = Ti
s = S
b = B
l = L

__all__ = ["Ti", "S", "B", "L", "ti", "s", "b", "l"]

_callable_module.install(__name__, _build)
