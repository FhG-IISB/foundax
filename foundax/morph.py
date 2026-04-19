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

_AdaptedViT3D = None


def _adapted_cls():
    """Return a cached ViT3DRegression subclass with (B, T, [D,] H, W, C) convention."""
    global _AdaptedViT3D
    if _AdaptedViT3D is not None:
        return _AdaptedViT3D
    import jax.numpy as jnp

    ensure_repo_on_path("jax_morph")
    _mod = importlib.import_module("jax_morph.model_eqx")

    class _Adapted(_mod.ViT3DRegression):
        def __call__(self, vol):
            ndim = vol.ndim
            if ndim == 5:
                # 2D: (B, T, H, W, C) -> (B, T, 1, C, 1, H, W)
                vol = jnp.transpose(vol, (0, 1, 4, 2, 3))  # (B, T, C, H, W)
                vol = vol[:, :, None, :, None, :, :]
            elif ndim == 6:
                # 3D: (B, T, D, H, W, C) -> (B, T, 1, C, D, H, W)
                vol = jnp.transpose(vol, (0, 1, 5, 2, 3, 4))  # (B, T, C, D, H, W)
                vol = vol[:, :, None, :, :, :, :]
            else:
                raise ValueError(
                    f"Expected 5-D (B,T,H,W,C) or 6-D (B,T,D,H,W,C), got {ndim}-D"
                )
            enc, z, pred = super().__call__(vol)
            # pred: (B, F=1, C, D, H, W)
            pred = pred[:, 0]  # (B, C, D, H, W)
            if ndim == 5:
                # squeeze D=1: (B, C, 1, H, W) -> (B, H, W, C)
                pred = pred[:, :, 0, :, :]  # (B, C, H, W)
                pred = jnp.transpose(pred, (0, 2, 3, 1))
            else:
                # (B, C, D, H, W) -> (B, D, H, W, C)
                pred = jnp.transpose(pred, (0, 2, 3, 4, 1))
            return enc, z, pred

    _AdaptedViT3D = _Adapted
    return _AdaptedViT3D


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
    """MORPH-Ti (Tiny) ~9.9 M params.

    3-D Vision Transformer (ViT3DRegression) for regression over
    arbitrary-modality PDE data with LoRA fine-tuning support.

    Reference:
        Rautela et al., *MORPH: PDE Foundation Models with Arbitrary
        Data Modality* (2025). https://arxiv.org/abs/2509.21670

    Example::

        model = foundax.morph.Ti(dropout=0.1)

    Shape:
        - Input: ``(B, T, H, W, C)`` for 2-D or ``(B, T, D, H, W, C)`` for 3-D.
        - Output: ``(enc, z, pred)`` where ``pred`` is
          ``(B, H, W, C)`` (2-D) or ``(B, D, H, W, C)`` (3-D).
        - Spatial dims must be divisible by ``patch_size``.

    See Also:
        :func:`S`, :func:`M`, :func:`L`

    Args:
        patch_size: Spatial size of 3-D patches.
        dim: Model embedding dimension.
        depth: Number of self-attention transformer blocks.
        heads: Number of self-attention heads.
        heads_xa: Number of cross-attention heads.
        mlp_dim: MLP hidden dimension.
        max_components: Maximum number of decomposed field components.
        conv_filter: Convolutional filter width in the patch embedding.
        max_ar: Maximum autoregressive rollout order.
        max_patches: Maximum number of patches the model can handle.
        max_fields: Maximum number of output fields.
        dropout: General dropout rate.
        emb_dropout: Dropout applied to the patch embeddings.
        lora_r_attn: LoRA rank for attention layers (``0`` = disabled).
        lora_r_mlp: LoRA rank for MLP layers (``0`` = disabled).
        lora_alpha: LoRA scaling factor (``None`` = ``lora_r``).
        lora_p: LoRA dropout probability.
        model_size: Variant label used for architecture selection.

    Returns:
        An ``equinox.Module`` (ViT3DRegression).
    """
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
    """MORPH-S (Small) ~32.8 M params.

    3-D Vision Transformer (ViT3DRegression) for regression over
    arbitrary-modality PDE data with LoRA fine-tuning support.

    Reference:
        Rautela et al., *MORPH: PDE Foundation Models with Arbitrary
        Data Modality* (2025). https://arxiv.org/abs/2509.21670

    Example::

        model = foundax.morph.S(dropout=0.1)

    Shape:
        - Input: ``(B, T, H, W, C)`` for 2-D or ``(B, T, D, H, W, C)`` for 3-D.
        - Output: ``(enc, z, pred)`` where ``pred`` is
          ``(B, H, W, C)`` (2-D) or ``(B, D, H, W, C)`` (3-D).
        - Spatial dims must be divisible by ``patch_size``.

    See Also:
        :func:`Ti`, :func:`M`, :func:`L`

    Args:
        patch_size: Spatial size of 3-D patches.
        dim: Model embedding dimension.
        depth: Number of self-attention transformer blocks.
        heads: Number of self-attention heads.
        heads_xa: Number of cross-attention heads.
        mlp_dim: MLP hidden dimension.
        max_components: Maximum number of decomposed field components.
        conv_filter: Convolutional filter width in the patch embedding.
        max_ar: Maximum autoregressive rollout order.
        max_patches: Maximum number of patches the model can handle.
        max_fields: Maximum number of output fields.
        dropout: General dropout rate.
        emb_dropout: Dropout applied to the patch embeddings.
        lora_r_attn: LoRA rank for attention layers (``0`` = disabled).
        lora_r_mlp: LoRA rank for MLP layers (``0`` = disabled).
        lora_alpha: LoRA scaling factor (``None`` = ``lora_r``).
        lora_p: LoRA dropout probability.
        model_size: Variant label used for architecture selection.

    Returns:
        An ``equinox.Module`` (ViT3DRegression).
    """
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
    """MORPH-M (Medium) ~125.6 M params.

    3-D Vision Transformer (ViT3DRegression) for regression over
    arbitrary-modality PDE data with LoRA fine-tuning support.

    Reference:
        Rautela et al., *MORPH: PDE Foundation Models with Arbitrary
        Data Modality* (2025). https://arxiv.org/abs/2509.21670

    Example::

        model = foundax.morph.M(dropout=0.1)

    Shape:
        - Input: ``(B, T, H, W, C)`` for 2-D or ``(B, T, D, H, W, C)`` for 3-D.
        - Output: ``(enc, z, pred)`` where ``pred`` is
          ``(B, H, W, C)`` (2-D) or ``(B, D, H, W, C)`` (3-D).
        - Spatial dims must be divisible by ``patch_size``.

    See Also:
        :func:`Ti`, :func:`S`, :func:`L`

    Args:
        patch_size: Spatial size of 3-D patches.
        dim: Model embedding dimension.
        depth: Number of self-attention transformer blocks.
        heads: Number of self-attention heads.
        heads_xa: Number of cross-attention heads.
        mlp_dim: MLP hidden dimension.
        max_components: Maximum number of decomposed field components.
        conv_filter: Convolutional filter width in the patch embedding.
        max_ar: Maximum autoregressive rollout order.
        max_patches: Maximum number of patches the model can handle.
        max_fields: Maximum number of output fields.
        dropout: General dropout rate.
        emb_dropout: Dropout applied to the patch embeddings.
        lora_r_attn: LoRA rank for attention layers (``0`` = disabled).
        lora_r_mlp: LoRA rank for MLP layers (``0`` = disabled).
        lora_alpha: LoRA scaling factor (``None`` = ``lora_r``).
        lora_p: LoRA dropout probability.
        model_size: Variant label used for architecture selection.

    Returns:
        An ``equinox.Module`` (ViT3DRegression).
    """
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
    """MORPH-L (Large) ~483.3 M params.

    3-D Vision Transformer (ViT3DRegression) for regression over
    arbitrary-modality PDE data with LoRA fine-tuning support.

    Reference:
        Rautela et al., *MORPH: PDE Foundation Models with Arbitrary
        Data Modality* (2025). https://arxiv.org/abs/2509.21670

    Example::

        model = foundax.morph.L(dropout=0.1)

    Shape:
        - Input: ``(B, T, H, W, C)`` for 2-D or ``(B, T, D, H, W, C)`` for 3-D.
        - Output: ``(enc, z, pred)`` where ``pred`` is
          ``(B, H, W, C)`` (2-D) or ``(B, D, H, W, C)`` (3-D).
        - Spatial dims must be divisible by ``patch_size``.

    See Also:
        :func:`Ti`, :func:`S`, :func:`M`

    Args:
        patch_size: Spatial size of 3-D patches.
        dim: Model embedding dimension.
        depth: Number of self-attention transformer blocks.
        heads: Number of self-attention heads.
        heads_xa: Number of cross-attention heads.
        mlp_dim: MLP hidden dimension.
        max_components: Maximum number of decomposed field components.
        conv_filter: Convolutional filter width in the patch embedding.
        max_ar: Maximum autoregressive rollout order.
        max_patches: Maximum number of patches the model can handle.
        max_fields: Maximum number of output fields.
        dropout: General dropout rate.
        emb_dropout: Dropout applied to the patch embeddings.
        lora_r_attn: LoRA rank for attention layers (``0`` = disabled).
        lora_r_mlp: LoRA rank for MLP layers (``0`` = disabled).
        lora_alpha: LoRA scaling factor (``None`` = ``lora_r``).
        lora_p: LoRA dropout probability.
        model_size: Variant label used for architecture selection.

    Returns:
        An ``equinox.Module`` (ViT3DRegression).
    """
    return _build(**{k: v for k, v in locals().items()})


ti = Ti
s = S
m = M
l = L

__all__ = ["Ti", "S", "M", "L", "ti", "s", "m", "l"]

_callable_module.install(__name__, _build)
