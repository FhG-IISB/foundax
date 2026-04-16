"""Flax linen UNet (2-D) for neural operator applications.

Port of jNO's Equinox UNet2D to Flax ``nn.Module``.
All spatial data is channels-last (H, W, C).
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn


# ---------------------------------------------------------------------------
# Activation helper
# ---------------------------------------------------------------------------

_ACTIVATION_MAP = {
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "tanh": jnp.tanh,
    "silu": jax.nn.silu,
    "celu": jax.nn.celu,
    "elu": jax.nn.elu,
}


def _get_activation(name: str) -> Callable:
    return _ACTIVATION_MAP[name.lower()]


# ---------------------------------------------------------------------------
# Padding & pooling helpers
# ---------------------------------------------------------------------------


def _pad_2d(x: jnp.ndarray, pad: int, mode: str = "circular") -> jnp.ndarray:
    pad_mode = "wrap" if mode == "circular" else "reflect"
    return jnp.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode=pad_mode)


def _avg_pool_2d(x: jnp.ndarray, stride: int = 2) -> jnp.ndarray:
    H, W, C = x.shape
    nH, nW = H // stride, W // stride
    return (
        x[: nH * stride, : nW * stride, :]
        .reshape(nH, stride, nW, stride, C)
        .mean(axis=(1, 3))
    )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class UNetConv2d(nn.Module):
    """Single conv + optional norm + optional activation."""

    features: int
    kernel_size: int = 3
    norm: Optional[str] = "batch"
    act: Optional[str] = None
    padding_mode: str = "circular"

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        pad = self.kernel_size // 2
        x = _pad_2d(x, pad, self.padding_mode)
        x = nn.Conv(
            self.features,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding="VALID",
            name="conv",
        )(x)
        if self.norm == "batch":
            x = nn.BatchNorm(use_running_average=deterministic, name="bn")(x)
        elif self.norm == "layer":
            x = nn.LayerNorm(name="ln")(x)
        elif self.norm == "group":
            x = nn.GroupNorm(num_groups=min(32, self.features), name="gn")(x)
        if self.act is not None:
            x = _get_activation(self.act)(x)
        return x


class UNetConvBlock2d(nn.Module):
    """Two convolutions (conv → norm → act) + (conv → norm)."""

    features: tuple  # (ch1, ch2)
    kernel_size: int = 3
    norm: Optional[str] = "batch"
    act: str = "celu"
    padding_mode: str = "circular"

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = UNetConv2d(
            self.features[0],
            self.kernel_size,
            self.norm,
            self.act,
            self.padding_mode,
            name="c1",
        )(x, deterministic)
        x = UNetConv2d(
            self.features[1],
            self.kernel_size,
            self.norm,
            None,
            self.padding_mode,
            name="c2",
        )(x, deterministic)
        return x


class UNetDownBlock2d(nn.Module):
    """Conv block → avg pool (returns skip + downsampled)."""

    features: tuple
    kernel_size: int = 3
    norm: Optional[str] = "batch"
    act: str = "celu"
    padding_mode: str = "circular"

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True):
        skip = UNetConvBlock2d(
            self.features,
            self.kernel_size,
            self.norm,
            self.act,
            self.padding_mode,
            name="block",
        )(x, deterministic)
        x_down = _avg_pool_2d(skip)
        return x_down, skip


class UNetUpBlock2d(nn.Module):
    """Upsample → concat skip → conv block."""

    features: tuple
    up_mode: str = "upconv"
    kernel_size: int = 3
    norm: Optional[str] = "batch"
    act: str = "celu"
    padding_mode: str = "circular"

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, skip: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        if self.up_mode == "upconv":
            x = nn.ConvTranspose(
                x.shape[-1], kernel_size=(2, 2), strides=(2, 2), name="up"
            )(x)
        else:
            H, W = x.shape[-3], x.shape[-2]
            x = jax.image.resize(
                x, shape=(H * 2, W * 2, x.shape[-1]), method="bilinear"
            )
            x = nn.Conv(x.shape[-1], kernel_size=(1, 1), name="up")(x)

        # Handle size mismatch
        tH, tW = skip.shape[0], skip.shape[1]
        x = x[:tH, :tW, :]
        x = jnp.concatenate([x, skip], axis=-1)
        return UNetConvBlock2d(
            self.features,
            self.kernel_size,
            self.norm,
            self.act,
            self.padding_mode,
            name="block",
        )(x, deterministic)


class UNet2D(nn.Module):
    """2-D UNet encoder-decoder with skip connections.

    Input:  ``(H, W, C_in)``
    Output: ``(H, W, C_out)``
    """

    in_channels: int = 1
    out_channels: int = 1
    depth: int = 4
    wf: int = 6
    norm: Optional[str] = "batch"
    up_mode: str = "upconv"
    activation: str = "celu"
    padding_mode: str = "circular"

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        skips = []

        # Encoder
        for i in range(self.depth):
            ch = (2**self.wf) * (2**i)
            x, skip = UNetDownBlock2d(
                features=(ch, ch),
                norm=self.norm,
                act=self.activation,
                padding_mode=self.padding_mode,
                name=f"enc_{i}",
            )(x, deterministic)
            skips.append(skip)

        # Bottleneck
        bneck_ch = (2**self.wf) * (2 ** (self.depth - 1))
        x = UNetConvBlock2d(
            features=(bneck_ch, bneck_ch),
            norm=self.norm,
            act=self.activation,
            padding_mode=self.padding_mode,
            name="bottleneck",
        )(x, deterministic)

        # Decoder
        for i in range(self.depth):
            didx = self.depth - 1 - i
            ch_in = (2**self.wf) * (2**didx)
            ch_out = (2**self.wf) * (2 ** max(0, didx - 1))
            x = UNetUpBlock2d(
                features=(ch_in, ch_out),
                up_mode=self.up_mode,
                norm=self.norm,
                act=self.activation,
                padding_mode=self.padding_mode,
                name=f"dec_{i}",
            )(x, skips[-(i + 1)], deterministic)

        # Final 1×1 conv
        x = nn.Conv(
            self.out_channels, kernel_size=(1, 1), use_bias=False, name="final"
        )(x)
        return x
