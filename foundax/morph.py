"""MORPH -- PDE Foundation Models with Arbitrary Data Modality.

**Paper:** Rautela et al., *"MORPH"* (2025)
https://arxiv.org/abs/2509.21670

Architecture: ``ViT3DRegression`` -- a 3-D Vision Transformer for
regression over PDE data.

Usage::

    model = foundax.morph.Ti(dropout=0.1)
    model = foundax.morph(dim=256, depth=4, heads=4, mlp_dim=1024)
"""

import importlib

from ._vendors import ensure_repo_on_path
from . import _callable_module


def _build(
    patch_size=8,
    dim=256,
    depth=4,
    heads=4,
    heads_xa=32,
    mlp_dim=1024,
    max_components=3,
    conv_filter=8,
    max_ar=1,
    max_patches=4096,
    max_fields=3,
    dropout=0.1,
    emb_dropout=0.1,
    lora_r_attn=0,
    lora_r_mlp=0,
    lora_alpha=None,
    lora_p=0.0,
    model_size="Ti",
):
    ensure_repo_on_path("jax_morph")
    mod = importlib.import_module("jax_morph")
    return mod.ViT3DRegression(**{k: v for k, v in locals().items() if k != "mod"})


def Ti(
    patch_size=8,
    dim=256,
    depth=4,
    heads=4,
    heads_xa=32,
    mlp_dim=1024,
    max_components=3,
    conv_filter=8,
    max_ar=1,
    max_patches=4096,
    max_fields=3,
    dropout=0.0,
    emb_dropout=0.0,
    lora_r_attn=0,
    lora_r_mlp=0,
    lora_alpha=None,
    lora_p=0.0,
    model_size="Ti",
):
    """MORPH-Ti (Tiny) ~9.9 M params. dim=256, depth=4, heads=4."""
    return _build(**{k: v for k, v in locals().items()})


def S(
    patch_size=8,
    dim=512,
    depth=4,
    heads=8,
    heads_xa=32,
    mlp_dim=2048,
    max_components=3,
    conv_filter=8,
    max_ar=1,
    max_patches=4096,
    max_fields=3,
    dropout=0.0,
    emb_dropout=0.0,
    lora_r_attn=0,
    lora_r_mlp=0,
    lora_alpha=None,
    lora_p=0.0,
    model_size="S",
):
    """MORPH-S (Small) ~32.8 M params. dim=512, depth=4, heads=8."""
    return _build(**{k: v for k, v in locals().items()})


def M(
    patch_size=8,
    dim=768,
    depth=8,
    heads=12,
    heads_xa=32,
    mlp_dim=3072,
    max_components=3,
    conv_filter=8,
    max_ar=1,
    max_patches=4096,
    max_fields=3,
    dropout=0.0,
    emb_dropout=0.0,
    lora_r_attn=0,
    lora_r_mlp=0,
    lora_alpha=None,
    lora_p=0.0,
    model_size="M",
):
    """MORPH-M (Medium) ~125.6 M params. dim=768, depth=8, heads=12."""
    return _build(**{k: v for k, v in locals().items()})


def L(
    patch_size=8,
    dim=1024,
    depth=16,
    heads=16,
    heads_xa=32,
    mlp_dim=4096,
    max_components=3,
    conv_filter=8,
    max_ar=16,
    max_patches=4096,
    max_fields=3,
    dropout=0.0,
    emb_dropout=0.0,
    lora_r_attn=0,
    lora_r_mlp=0,
    lora_alpha=None,
    lora_p=0.0,
    model_size="L",
):
    """MORPH-L (Large) ~483.3 M params. dim=1024, depth=16, heads=16, max_ar=16."""
    return _build(**{k: v for k, v in locals().items()})


ti = Ti
s = S
m = M
l = L

__all__ = ["Ti", "S", "M", "L", "ti", "s", "m", "l"]

_callable_module.install(__name__, _build)
