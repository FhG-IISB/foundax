"""Flax linen Fourier Neural Operators (1-D, 2-D, 3-D).

Port of jNO's Equinox FNO to Flax ``nn.Module``.
All spatial data is channels-last (NHWC / NWC / NDHWC).
"""

from typing import Callable, Optional, Sequence, Tuple

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
    "swish": jax.nn.silu,
    "celu": jax.nn.celu,
    "elu": jax.nn.elu,
}


def _get_activation(name: str) -> Callable:
    return _ACTIVATION_MAP[name.lower()]


# ---------------------------------------------------------------------------
# 2-D Spectral Conv
# ---------------------------------------------------------------------------


class SpectralConv2d(nn.Module):
    """2-D spectral convolution in the Fourier domain."""

    out_channels: int
    n_modes1: int
    n_modes2: int
    linear_conv: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (H, W, C_in)
        H, W, C_in = x.shape
        C_out = self.out_channels
        n1, n2 = self.n_modes1, self.n_modes2
        scale = 1.0 / (C_in * C_out)

        w1_r = self.param(
            "w1_real", nn.initializers.uniform(scale), (C_in, C_out, n1, n2)
        )
        w1_i = self.param(
            "w1_imag", nn.initializers.uniform(scale), (C_in, C_out, n1, n2)
        )
        w2_r = self.param(
            "w2_real", nn.initializers.uniform(scale), (C_in, C_out, n1, n2)
        )
        w2_i = self.param(
            "w2_imag", nn.initializers.uniform(scale), (C_in, C_out, n1, n2)
        )

        fft_h = H * 2 - 1 if self.linear_conv else H
        fft_w = W * 2 - 1 if self.linear_conv else W

        X = jnp.fft.rfft2(x, s=(fft_h, fft_w), axes=(0, 1), norm="ortho")
        freq_h, freq_w = X.shape[0], X.shape[1]

        nm1 = min(n1, (freq_h + 1) // 2)
        nm2 = min(n2, freq_w)

        w1 = (w1_r + 1j * w1_i)[:, :, :nm1, :nm2]
        w2 = (w2_r + 1j * w2_i)[:, :, :nm1, :nm2]

        out_upper = jnp.einsum("hwi,iohw->hwo", X[:nm1, :nm2, :], w1)
        out_lower = jnp.einsum("hwi,iohw->hwo", X[-nm1:, :nm2, :], w2)

        out_ft = jnp.zeros((freq_h, freq_w, C_out), dtype=jnp.complex64)
        out_ft = out_ft.at[:nm1, :nm2, :].set(out_upper)
        out_ft = out_ft.at[-nm1:, :nm2, :].set(out_lower)

        return jnp.fft.irfft2(out_ft, s=(fft_h, fft_w), axes=(0, 1), norm="ortho")[
            :H, :W, :
        ]


class SpectralLayers2d(nn.Module):
    """Stack of 2-D spectral convolution layers."""

    n_channels: int
    n_modes1: int
    n_modes2: int
    n_layers: int = 4
    activation: str = "gelu"
    norm: Optional[str] = "layer"
    linear_conv: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        act = _get_activation(self.activation)
        for i in range(self.n_layers):
            x1 = SpectralConv2d(
                out_channels=self.n_channels,
                n_modes1=self.n_modes1,
                n_modes2=self.n_modes2,
                linear_conv=self.linear_conv,
                name=f"spec_{i}",
            )(x)
            # 1x1 conv (channels-last)
            x2 = nn.Conv(self.n_channels, kernel_size=(1, 1), name=f"w_{i}")(x)
            x = x1 + x2
            if self.norm == "layer":
                x = nn.LayerNorm(name=f"ln_{i}")(x)
            elif self.norm == "batch":
                x = nn.BatchNorm(use_running_average=deterministic, name=f"bn_{i}")(x)
            x = act(x)
        return x


class FNO2D(nn.Module):
    """2-D Fourier Neural Operator.

    Input:  ``(H, W, C_in)`` or ``(T, H, W, C_in)``
    Output: ``(n_steps, H, W, d_vars)`` or squeezed variants.
    """

    hidden_channels: int = 32
    n_modes: int = 16
    d_vars: int = 1
    n_layers: int = 4
    n_steps: int = 1
    activation: str = "gelu"
    norm: Optional[str] = "layer"
    use_positions: bool = False
    linear_conv: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        act = _get_activation(self.activation)

        # Normalize to (T, H, W, C)
        ndim = x.ndim
        if ndim == 2:
            x = x[:, :, jnp.newaxis][jnp.newaxis, :, :, :]
        elif ndim == 3:
            x = x[jnp.newaxis, :, :, :]

        T, H, W, C = x.shape
        # Merge time into channels: (H, W, T*C)
        x = x.transpose(1, 2, 0, 3).reshape(H, W, T * C)

        if self.use_positions:
            gy = jnp.linspace(0, 1, H)
            gx = jnp.linspace(0, 1, W)
            yy, xx = jnp.meshgrid(gy, gx, indexing="ij")
            grid = jnp.stack([yy, xx], axis=-1)
            x = jnp.concatenate([x, grid], axis=-1)

        # Lift: pointwise Dense over (H, W)
        P = nn.Dense(self.hidden_channels, name="lift")
        x = jax.vmap(jax.vmap(P))(x)

        # Spectral layers
        x = SpectralLayers2d(
            n_channels=self.hidden_channels,
            n_modes1=self.n_modes,
            n_modes2=self.n_modes,
            n_layers=self.n_layers,
            activation=self.activation,
            norm=self.norm,
            linear_conv=self.linear_conv,
            name="spectral",
        )(x, deterministic=deterministic)

        # Project: pointwise MLP over (H, W)
        Q1 = nn.Dense(128, name="proj1")
        Q2 = nn.Dense(self.d_vars * self.n_steps, name="proj2")
        x = jax.vmap(jax.vmap(Q1))(x)
        x = act(x)
        x = jax.vmap(jax.vmap(Q2))(x)

        # Reshape to (n_steps, H, W, d_vars)
        x = x.reshape(H, W, self.n_steps, self.d_vars)
        x = x.transpose(2, 0, 1, 3)

        # De-normalize
        if ndim == 2 and self.n_steps == 1 and self.d_vars == 1:
            return x[0, :, :, 0]
        if ndim == 2 and self.n_steps == 1:
            return x[0]
        if ndim == 3 and self.n_steps == 1:
            return x[0]
        return x
