"""DPOT -- DPOTNet (2-D) model wrappers.

**Paper:** Hao et al., *"DPOT: Auto-Regressive Denoising Operator Transformer
for Large-Scale PDE Pre-Training"* (ICML 2024)
https://arxiv.org/abs/2403.03542

Architecture: ``DPOTNet`` -- a patched Fourier / AFNO-based neural
operator with temporal aggregation for 2-D PDE surrogate modelling.

Usage::

    model = foundax.dpot.Ti(in_channels=1, out_channels=1)
    model = foundax.dpot(embed_dim=512, depth=4, in_channels=1, out_channels=1)
"""

import importlib
from typing import Callable

import jax

from ._vendors import ensure_repo_on_path
from . import _callable_module

_AdaptedDPOTNet = None


def _adapted_cls():
    """Return a cached DPOTNet subclass with (B, T, H, W, C) convention."""
    global _AdaptedDPOTNet
    if _AdaptedDPOTNet is not None:
        return _AdaptedDPOTNet
    import jax.numpy as jnp

    ensure_repo_on_path("jax_dpot")
    _mod = importlib.import_module("jax_dpot.model_eqx")

    class _Adapted(_mod.DPOTNet):
        def __call__(self, x):
            # (B, T, H, W, C) -> (B, H, W, T, C)
            x = jnp.transpose(x, (0, 2, 3, 1, 4))
            out, cls = super().__call__(x)
            # (B, H, W, T_out, C) -> (B, T_out, H, W, C)
            return jnp.transpose(out, (0, 3, 1, 2, 4)), cls

    _AdaptedDPOTNet = _Adapted
    return _AdaptedDPOTNet


def _build(
    img_size=224,
    patch_size=16,
    mixing_type="afno",
    in_channels=1,
    out_channels=4,
    in_timesteps=1,
    out_timesteps=1,
    n_blocks=4,
    embed_dim=768,
    out_layer_dim=32,
    depth=12,
    modes=32,
    mlp_ratio=1.0,
    n_cls=12,
    normalize=False,
    act: Callable = jax.nn.gelu,
    time_agg="exp_mlp",
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
    img_size=128,
    patch_size=8,
    mixing_type="afno",
    in_channels=4,
    out_channels=4,
    in_timesteps=10,
    out_timesteps=1,
    n_blocks=4,
    embed_dim=512,
    out_layer_dim=32,
    depth=4,
    modes=32,
    mlp_ratio=1.0,
    n_cls=12,
    normalize=False,
    act: Callable = jax.nn.gelu,
    time_agg="exp_mlp",
):
    """DPOTNet-Ti (Tiny).

    Patched Fourier / AFNO-based neural operator with temporal
    aggregation for 2-D PDE surrogate modelling.

    Reference:
        Hao et al., *DPOT: Auto-Regressive Denoising Operator
        Transformer for Large-Scale PDE Pre-Training* (ICML 2024).
        https://arxiv.org/abs/2403.03542

    Example::

        model = foundax.dpot.Ti(in_channels=1, out_channels=1)

    Shape:
        - Input: ``(B, in_timesteps, H, W, in_channels)``
        - Output: ``(B, out_timesteps, H, W, out_channels)``
        - ``img_size`` must be divisible by ``patch_size``.

    See Also:
        :func:`S`, :func:`M`, :func:`L`, :func:`H`

    Args:
        img_size: Spatial resolution of the input image.
        patch_size: Patch size for input tokenisation.
        mixing_type: Spatial mixing mechanism (``"afno"`` = Adaptive
            Fourier Neural Operator).
        in_channels: Number of input physical channels.
        out_channels: Number of output physical channels.
        in_timesteps: Number of input time steps.
        out_timesteps: Number of output time steps.
        n_blocks: Number of AFNO mixing blocks per layer.
        embed_dim: Token embedding dimension.
        out_layer_dim: Hidden dimension of the output projection head.
        depth: Number of transformer layers.
        modes: Number of Fourier modes retained per dimension.
        mlp_ratio: Ratio of MLP hidden dim to ``embed_dim``.
        n_cls: Number of classification / task-embedding tokens.
        normalize: Enable feature normalisation.
        act: Activation callable (e.g. ``jax.nn.gelu``,
            ``jax.nn.relu``).
        time_agg: Temporal aggregation method (``"exp_mlp"``).

    Returns:
        An ``equinox.Module`` (DPOTNet).
    """
    return _build(**{k: v for k, v in locals().items()})


def S(
    img_size=128,
    patch_size=8,
    mixing_type="afno",
    in_channels=4,
    out_channels=4,
    in_timesteps=10,
    out_timesteps=1,
    n_blocks=8,
    embed_dim=1024,
    out_layer_dim=32,
    depth=6,
    modes=32,
    mlp_ratio=1.0,
    n_cls=12,
    normalize=False,
    act: Callable = jax.nn.gelu,
    time_agg="exp_mlp",
):
    """DPOTNet-S (Small).

    Patched Fourier / AFNO-based neural operator with temporal
    aggregation for 2-D PDE surrogate modelling.

    Reference:
        Hao et al., *DPOT: Auto-Regressive Denoising Operator
        Transformer for Large-Scale PDE Pre-Training* (ICML 2024).
        https://arxiv.org/abs/2403.03542

    Example::

        model = foundax.dpot.S(in_channels=1, out_channels=1)

    Shape:
        - Input: ``(B, in_timesteps, H, W, in_channels)``
        - Output: ``(B, out_timesteps, H, W, out_channels)``
        - ``img_size`` must be divisible by ``patch_size``.

    See Also:
        :func:`Ti`, :func:`M`, :func:`L`, :func:`H`

    Args:
        img_size: Spatial resolution of the input image.
        patch_size: Patch size for input tokenisation.
        mixing_type: Spatial mixing mechanism (``"afno"`` = Adaptive
            Fourier Neural Operator).
        in_channels: Number of input physical channels.
        out_channels: Number of output physical channels.
        in_timesteps: Number of input time steps.
        out_timesteps: Number of output time steps.
        n_blocks: Number of AFNO mixing blocks per layer.
        embed_dim: Token embedding dimension.
        out_layer_dim: Hidden dimension of the output projection head.
        depth: Number of transformer layers.
        modes: Number of Fourier modes retained per dimension.
        mlp_ratio: Ratio of MLP hidden dim to ``embed_dim``.
        n_cls: Number of classification / task-embedding tokens.
        normalize: Enable feature normalisation.
        act: Activation callable (e.g. ``jax.nn.gelu``,
            ``jax.nn.relu``).
        time_agg: Temporal aggregation method (``"exp_mlp"``).

    Returns:
        An ``equinox.Module`` (DPOTNet).
    """
    return _build(**{k: v for k, v in locals().items()})


def M(
    img_size=128,
    patch_size=8,
    mixing_type="afno",
    in_channels=4,
    out_channels=4,
    in_timesteps=10,
    out_timesteps=1,
    n_blocks=8,
    embed_dim=1024,
    out_layer_dim=32,
    depth=12,
    modes=32,
    mlp_ratio=4.0,
    n_cls=12,
    normalize=False,
    act: Callable = jax.nn.gelu,
    time_agg="exp_mlp",
):
    """DPOTNet-M (Medium).

    Patched Fourier / AFNO-based neural operator with temporal
    aggregation for 2-D PDE surrogate modelling.

    Reference:
        Hao et al., *DPOT: Auto-Regressive Denoising Operator
        Transformer for Large-Scale PDE Pre-Training* (ICML 2024).
        https://arxiv.org/abs/2403.03542

    Example::

        model = foundax.dpot.M(in_channels=1, out_channels=1)

    Shape:
        - Input: ``(B, in_timesteps, H, W, in_channels)``
        - Output: ``(B, out_timesteps, H, W, out_channels)``
        - ``img_size`` must be divisible by ``patch_size``.

    See Also:
        :func:`Ti`, :func:`S`, :func:`L`, :func:`H`

    Args:
        img_size: Spatial resolution of the input image.
        patch_size: Patch size for input tokenisation.
        mixing_type: Spatial mixing mechanism (``"afno"`` = Adaptive
            Fourier Neural Operator).
        in_channels: Number of input physical channels.
        out_channels: Number of output physical channels.
        in_timesteps: Number of input time steps.
        out_timesteps: Number of output time steps.
        n_blocks: Number of AFNO mixing blocks per layer.
        embed_dim: Token embedding dimension.
        out_layer_dim: Hidden dimension of the output projection head.
        depth: Number of transformer layers.
        modes: Number of Fourier modes retained per dimension.
        mlp_ratio: Ratio of MLP hidden dim to ``embed_dim``.
        n_cls: Number of classification / task-embedding tokens.
        normalize: Enable feature normalisation.
        act: Activation callable (e.g. ``jax.nn.gelu``,
            ``jax.nn.relu``).
        time_agg: Temporal aggregation method (``"exp_mlp"``).

    Returns:
        An ``equinox.Module`` (DPOTNet).
    """
    return _build(**{k: v for k, v in locals().items()})


def L(
    img_size=128,
    patch_size=8,
    mixing_type="afno",
    in_channels=4,
    out_channels=4,
    in_timesteps=10,
    out_timesteps=1,
    n_blocks=16,
    embed_dim=1536,
    out_layer_dim=128,
    depth=24,
    modes=32,
    mlp_ratio=4.0,
    n_cls=12,
    normalize=False,
    act: Callable = jax.nn.gelu,
    time_agg="exp_mlp",
):
    """DPOTNet-L (Large).

    Patched Fourier / AFNO-based neural operator with temporal
    aggregation for 2-D PDE surrogate modelling.

    Reference:
        Hao et al., *DPOT: Auto-Regressive Denoising Operator
        Transformer for Large-Scale PDE Pre-Training* (ICML 2024).
        https://arxiv.org/abs/2403.03542

    Example::

        model = foundax.dpot.L(in_channels=1, out_channels=1)

    Shape:
        - Input: ``(B, in_timesteps, H, W, in_channels)``
        - Output: ``(B, out_timesteps, H, W, out_channels)``
        - ``img_size`` must be divisible by ``patch_size``.

    See Also:
        :func:`Ti`, :func:`S`, :func:`M`, :func:`H`

    Args:
        img_size: Spatial resolution of the input image.
        patch_size: Patch size for input tokenisation.
        mixing_type: Spatial mixing mechanism (``"afno"`` = Adaptive
            Fourier Neural Operator).
        in_channels: Number of input physical channels.
        out_channels: Number of output physical channels.
        in_timesteps: Number of input time steps.
        out_timesteps: Number of output time steps.
        n_blocks: Number of AFNO mixing blocks per layer.
        embed_dim: Token embedding dimension.
        out_layer_dim: Hidden dimension of the output projection head.
        depth: Number of transformer layers.
        modes: Number of Fourier modes retained per dimension.
        mlp_ratio: Ratio of MLP hidden dim to ``embed_dim``.
        n_cls: Number of classification / task-embedding tokens.
        normalize: Enable feature normalisation.
        act: Activation callable (e.g. ``jax.nn.gelu``,
            ``jax.nn.relu``).
        time_agg: Temporal aggregation method (``"exp_mlp"``).

    Returns:
        An ``equinox.Module`` (DPOTNet).
    """
    return _build(**{k: v for k, v in locals().items()})


def H(
    img_size=128,
    patch_size=8,
    mixing_type="afno",
    in_channels=4,
    out_channels=4,
    in_timesteps=10,
    out_timesteps=1,
    n_blocks=8,
    embed_dim=2048,
    out_layer_dim=128,
    depth=27,
    modes=32,
    mlp_ratio=4.0,
    n_cls=12,
    normalize=False,
    act: Callable = jax.nn.gelu,
    time_agg="exp_mlp",
):
    """DPOTNet-H (Huge).

    Patched Fourier / AFNO-based neural operator with temporal
    aggregation for 2-D PDE surrogate modelling.

    Reference:
        Hao et al., *DPOT: Auto-Regressive Denoising Operator
        Transformer for Large-Scale PDE Pre-Training* (ICML 2024).
        https://arxiv.org/abs/2403.03542

    Example::

        model = foundax.dpot.H(in_channels=1, out_channels=1)

    Shape:
        - Input: ``(B, in_timesteps, H, W, in_channels)``
        - Output: ``(B, out_timesteps, H, W, out_channels)``
        - ``img_size`` must be divisible by ``patch_size``.

    See Also:
        :func:`Ti`, :func:`S`, :func:`M`, :func:`L`

    Args:
        img_size: Spatial resolution of the input image.
        patch_size: Patch size for input tokenisation.
        mixing_type: Spatial mixing mechanism (``"afno"`` = Adaptive
            Fourier Neural Operator).
        in_channels: Number of input physical channels.
        out_channels: Number of output physical channels.
        in_timesteps: Number of input time steps.
        out_timesteps: Number of output time steps.
        n_blocks: Number of AFNO mixing blocks per layer.
        embed_dim: Token embedding dimension.
        out_layer_dim: Hidden dimension of the output projection head.
        depth: Number of transformer layers.
        modes: Number of Fourier modes retained per dimension.
        mlp_ratio: Ratio of MLP hidden dim to ``embed_dim``.
        n_cls: Number of classification / task-embedding tokens.
        normalize: Enable feature normalisation.
        act: Activation callable (e.g. ``jax.nn.gelu``,
            ``jax.nn.relu``).
        time_agg: Temporal aggregation method (``"exp_mlp"``).

    Returns:
        An ``equinox.Module`` (DPOTNet).
    """
    return _build(**{k: v for k, v in locals().items()})


ti = Ti
s = S
m = M
l = L
h = H

__all__ = ["Ti", "S", "M", "L", "H", "ti", "s", "m", "l", "h"]

_callable_module.install(__name__, _build)
