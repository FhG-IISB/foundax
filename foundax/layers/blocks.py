"""Composable operator blocks with consistent (spatial…, C) channel-last convention.

All blocks map unbatched tensors; batching is the caller's responsibility via
``jax.vmap``.  Each block exposes ``in_channels`` and ``out_channels`` as
static fields so :func:`foundax.block` can detect channel mismatches at
pipe-construction time.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import equinox as eqx

from foundax.architectures.fno import SpectralConv1d, SpectralConv2d, SpectralConv3d
from foundax.architectures.linear import Linear
from foundax.architectures.common import BatchNorm


def _make_norm(norm: Optional[str], channels: int) -> Optional[eqx.Module]:
    if norm == "layer":
        return eqx.nn.LayerNorm(channels)
    if norm == "batch":
        return BatchNorm(channels)
    return None


class SpectralBlock1d(eqx.Module):
    """Single FNO-style spectral layer for 1-D fields: ``(W, C) -> (W, C')``.

    Computes ``spectral_conv(x) + linear_skip(x)``, applies optional
    normalisation pointwise over W, then the activation.
    """

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    n_modes: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    norm: Optional[str] = eqx.field(static=True)

    spectral_conv: SpectralConv1d
    linear_skip: Linear
    norm_layer: Optional[eqx.Module]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int,
        activation: Callable = jax.nn.gelu,
        norm: Optional[str] = None,
        linear_conv: bool = True,
        *,
        key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.activation = activation
        self.norm = norm

        k1, k2 = jax.random.split(key)
        self.spectral_conv = SpectralConv1d(
            in_channels, out_channels, n_modes, linear_conv, key=k1
        )
        self.linear_skip = Linear(in_channels, out_channels, key=k2)
        self.norm_layer = _make_norm(norm, out_channels)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        x1 = self.spectral_conv(x)
        x2 = self.linear_skip(x)
        x = x1 + x2
        if self.norm_layer is not None:
            x = jax.vmap(self.norm_layer)(x)
        return self.activation(x)


class SpectralBlock2d(eqx.Module):
    """Single FNO-style spectral layer for 2-D fields: ``(H, W, C) -> (H, W, C')``.

    Computes ``spectral_conv(x) + linear_skip(x)``, applies optional
    normalisation pointwise over H×W, then the activation.

    Uses the same ``n_modes`` for both spatial axes; for asymmetric modes use
    ``SpectralConv2d`` directly.
    """

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    n_modes: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    norm: Optional[str] = eqx.field(static=True)

    spectral_conv: SpectralConv2d
    linear_skip: Linear
    norm_layer: Optional[eqx.Module]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int,
        activation: Callable = jax.nn.gelu,
        norm: Optional[str] = None,
        linear_conv: bool = True,
        *,
        key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.activation = activation
        self.norm = norm

        k1, k2 = jax.random.split(key)
        self.spectral_conv = SpectralConv2d(
            in_channels, out_channels, n_modes, n_modes, linear_conv, key=k1
        )
        self.linear_skip = Linear(in_channels, out_channels, key=k2)
        self.norm_layer = _make_norm(norm, out_channels)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        x1 = self.spectral_conv(x)
        x2 = self.linear_skip(x)
        x = x1 + x2
        if self.norm_layer is not None:
            x = jax.vmap(jax.vmap(self.norm_layer))(x)
        return self.activation(x)


class SpectralBlock3d(eqx.Module):
    """Single FNO-style spectral layer for 3-D fields: ``(D, H, W, C) -> (D, H, W, C')``.

    Uses the same ``n_modes`` for all three spatial axes; for asymmetric
    modes use ``SpectralConv3d`` directly.
    """

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    n_modes: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    norm: Optional[str] = eqx.field(static=True)

    spectral_conv: SpectralConv3d
    linear_skip: Linear
    norm_layer: Optional[eqx.Module]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int,
        activation: Callable = jax.nn.gelu,
        norm: Optional[str] = None,
        linear_conv: bool = True,
        *,
        key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.activation = activation
        self.norm = norm

        k1, k2 = jax.random.split(key)
        self.spectral_conv = SpectralConv3d(
            in_channels, out_channels, n_modes, n_modes, n_modes, linear_conv, key=k1
        )
        self.linear_skip = Linear(in_channels, out_channels, key=k2)
        self.norm_layer = _make_norm(norm, out_channels)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        x1 = self.spectral_conv(x)
        x2 = self.linear_skip(x)
        x = x1 + x2
        if self.norm_layer is not None:
            x = jax.vmap(jax.vmap(jax.vmap(self.norm_layer)))(x)
        return self.activation(x)
