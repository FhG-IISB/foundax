# Factorized Fourier Neural Operator backbone.
#
# Paper: "Factorized Fourier Neural Operators"
#        Tran et al. (2023) — https://arxiv.org/abs/2111.13802
# Code:  https://github.com/alasdairtran/fourierflow  (PyTorch reference)
#
# Key idea: replace the O(m^d · C^2) full-d-D spectral convolution with d
# independent 1-D spectral convolutions summed together, reducing cost to
# O(d · m · C^2) while retaining most of the expressivity.
#
# All tensors follow the channel-last, unbatched foundax convention:
#   2-D: (H, W, C)
#   3-D: (D, H, W, C)

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import equinox as eqx

from .linear import Linear
from .time_embed import FiLMLayer


# ── factorized spectral convolutions ─────────────────────────────────────────


class FactorizedSpectralConv2d(eqx.Module):
    """Factorized 2-D spectral convolution via two 1-D spectral ops.

    For input (H, W, C_in) computes:

        out = IFFT_W(W_x · FFT_W(x)) + IFFT_H(W_y · FFT_H(x))

    Weight shapes:
        weight_x_real / _imag : (in_channels, out_channels, n_modes_x)
        weight_y_real / _imag : (in_channels, out_channels, n_modes_y)
    """

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    n_modes_x: int = eqx.field(static=True)
    n_modes_y: int = eqx.field(static=True)

    weight_x_real: jnp.ndarray
    weight_x_imag: jnp.ndarray
    weight_y_real: jnp.ndarray
    weight_y_imag: jnp.ndarray

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes_x: int,
        n_modes_y: int,
        *,
        key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes_x = n_modes_x
        self.n_modes_y = n_modes_y

        scale = 1.0 / (in_channels * out_channels)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        shape_x = (in_channels, out_channels, n_modes_x)
        shape_y = (in_channels, out_channels, n_modes_y)
        self.weight_x_real = jax.random.uniform(
            k1, shape_x, minval=-scale, maxval=scale
        )
        self.weight_x_imag = jax.random.uniform(
            k2, shape_x, minval=-scale, maxval=scale
        )
        self.weight_y_real = jax.random.uniform(
            k3, shape_y, minval=-scale, maxval=scale
        )
        self.weight_y_imag = jax.random.uniform(
            k4, shape_y, minval=-scale, maxval=scale
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        H, W, _ = x.shape
        W_x = self.weight_x_real + 1j * self.weight_x_imag
        W_y = self.weight_y_real + 1j * self.weight_y_imag

        # Branch 1: 1-D FFT along W (axis=1)
        X_w = jnp.fft.rfft(x, axis=1, norm="ortho")  # (H, W//2+1, C_in)
        nx = min(self.n_modes_x, W // 2 + 1)
        out1 = jnp.einsum(
            "hmi,iom->hmo", X_w[:, :nx, :], W_x[:, :, :nx]
        )  # (H, nx, C_out)
        out1_pad = jnp.zeros((H, W // 2 + 1, self.out_channels), dtype=jnp.complex64)
        out1_pad = out1_pad.at[:, :nx, :].set(out1)
        branch1 = jnp.fft.irfft(out1_pad, n=W, axis=1, norm="ortho")  # (H, W, C_out)

        # Branch 2: 1-D FFT along H (axis=0)
        X_h = jnp.fft.rfft(x, axis=0, norm="ortho")  # (H//2+1, W, C_in)
        ny = min(self.n_modes_y, H // 2 + 1)
        out2 = jnp.einsum(
            "mwi,iom->mwo", X_h[:ny, :, :], W_y[:, :, :ny]
        )  # (ny, W, C_out)
        out2_pad = jnp.zeros((H // 2 + 1, W, self.out_channels), dtype=jnp.complex64)
        out2_pad = out2_pad.at[:ny, :, :].set(out2)
        branch2 = jnp.fft.irfft(out2_pad, n=H, axis=0, norm="ortho")  # (H, W, C_out)

        return branch1 + branch2


class FactorizedSpectralConv3d(eqx.Module):
    """Factorized 3-D spectral convolution via three 1-D spectral ops.

    Weight shapes:
        weight_{d,h,w}_real / _imag : (in_channels, out_channels, n_modes_{axis})
    """

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    n_modes_d: int = eqx.field(static=True)
    n_modes_h: int = eqx.field(static=True)
    n_modes_w: int = eqx.field(static=True)

    weight_d_real: jnp.ndarray
    weight_d_imag: jnp.ndarray
    weight_h_real: jnp.ndarray
    weight_h_imag: jnp.ndarray
    weight_w_real: jnp.ndarray
    weight_w_imag: jnp.ndarray

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes_d: int,
        n_modes_h: int,
        n_modes_w: int,
        *,
        key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes_d = n_modes_d
        self.n_modes_h = n_modes_h
        self.n_modes_w = n_modes_w

        scale = 1.0 / (in_channels * out_channels)
        keys = jax.random.split(key, 6)
        self.weight_d_real = jax.random.uniform(
            keys[0], (in_channels, out_channels, n_modes_d), minval=-scale, maxval=scale
        )
        self.weight_d_imag = jax.random.uniform(
            keys[1], (in_channels, out_channels, n_modes_d), minval=-scale, maxval=scale
        )
        self.weight_h_real = jax.random.uniform(
            keys[2], (in_channels, out_channels, n_modes_h), minval=-scale, maxval=scale
        )
        self.weight_h_imag = jax.random.uniform(
            keys[3], (in_channels, out_channels, n_modes_h), minval=-scale, maxval=scale
        )
        self.weight_w_real = jax.random.uniform(
            keys[4], (in_channels, out_channels, n_modes_w), minval=-scale, maxval=scale
        )
        self.weight_w_imag = jax.random.uniform(
            keys[5], (in_channels, out_channels, n_modes_w), minval=-scale, maxval=scale
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        D, H, W, _ = x.shape
        W_d = self.weight_d_real + 1j * self.weight_d_imag
        W_h = self.weight_h_real + 1j * self.weight_h_imag
        W_w = self.weight_w_real + 1j * self.weight_w_imag

        # Branch D: FFT along axis=0
        X_d = jnp.fft.rfft(x, axis=0, norm="ortho")  # (D//2+1, H, W, C)
        nd = min(self.n_modes_d, D // 2 + 1)
        o_d = jnp.einsum("mhwi,iom->mhwo", X_d[:nd], W_d[:, :, :nd])
        p_d = jnp.zeros((D // 2 + 1, H, W, self.out_channels), dtype=jnp.complex64)
        p_d = p_d.at[:nd].set(o_d)
        br_d = jnp.fft.irfft(p_d, n=D, axis=0, norm="ortho")  # (D, H, W, C_out)

        # Branch H: FFT along axis=1
        X_h = jnp.fft.rfft(x, axis=1, norm="ortho")  # (D, H//2+1, W, C)
        nh = min(self.n_modes_h, H // 2 + 1)
        o_h = jnp.einsum("dmwi,iom->dmwo", X_h[:, :nh], W_h[:, :, :nh])
        p_h = jnp.zeros((D, H // 2 + 1, W, self.out_channels), dtype=jnp.complex64)
        p_h = p_h.at[:, :nh].set(o_h)
        br_h = jnp.fft.irfft(p_h, n=H, axis=1, norm="ortho")  # (D, H, W, C_out)

        # Branch W: FFT along axis=2
        X_w = jnp.fft.rfft(x, axis=2, norm="ortho")  # (D, H, W//2+1, C)
        nw = min(self.n_modes_w, W // 2 + 1)
        o_w = jnp.einsum("dhmi,iom->dhmo", X_w[:, :, :nw], W_w[:, :, :nw])
        p_w = jnp.zeros((D, H, W // 2 + 1, self.out_channels), dtype=jnp.complex64)
        p_w = p_w.at[:, :, :nw].set(o_w)
        br_w = jnp.fft.irfft(p_w, n=W, axis=2, norm="ortho")  # (D, H, W, C_out)

        return br_d + br_h + br_w


# ── composable layer blocks (pipe-compatible) ─────────────────────────────────


class FactorizedSpectralBlock2d(eqx.Module):
    """Composable F-FNO layer for 2-D fields: (H, W, C) → (H, W, C').

    ``spectral_conv(x) + linear_skip(x)`` followed by optional FiLM
    conditioning from a time embedding.  Exposes ``in_channels`` and
    ``out_channels`` as static fields so :func:`foundax.block` can detect
    channel mismatches at pipe-construction time.

    Accepts ``**kwargs`` in ``__call__`` so it can be freely mixed with
    other block types in heterogeneous pipes (unknown kwargs are ignored).
    """

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    n_modes: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    use_film: bool = eqx.field(static=True)

    spectral_conv: FactorizedSpectralConv2d
    linear_skip: Linear
    film: Optional[FiLMLayer]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int,
        activation: Callable = jax.nn.gelu,
        use_film: bool = False,
        emb_dim: int = 64,
        *,
        key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.activation = activation
        self.use_film = use_film

        k1, k2, k3 = jax.random.split(key, 3)
        self.spectral_conv = FactorizedSpectralConv2d(
            in_channels, out_channels, n_modes, n_modes, key=k1
        )
        self.linear_skip = Linear(in_channels, out_channels, key=k2)
        self.film = FiLMLayer(emb_dim, out_channels, key=k3) if use_film else None

    def __call__(self, x: jnp.ndarray, t_emb=None, **kwargs) -> jnp.ndarray:
        out = self.activation(self.spectral_conv(x) + self.linear_skip(x))
        if self.film is not None and t_emb is not None:
            out = self.film(out, t_emb)
        return out


class FactorizedSpectralBlock3d(eqx.Module):
    """Composable F-FNO layer for 3-D fields: (D, H, W, C) → (D, H, W, C')."""

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    n_modes: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    use_film: bool = eqx.field(static=True)

    spectral_conv: FactorizedSpectralConv3d
    linear_skip: Linear
    film: Optional[FiLMLayer]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int,
        activation: Callable = jax.nn.gelu,
        use_film: bool = False,
        emb_dim: int = 64,
        *,
        key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.activation = activation
        self.use_film = use_film

        k1, k2, k3 = jax.random.split(key, 3)
        self.spectral_conv = FactorizedSpectralConv3d(
            in_channels, out_channels, n_modes, n_modes, n_modes, key=k1
        )
        self.linear_skip = Linear(in_channels, out_channels, key=k2)
        self.film = FiLMLayer(emb_dim, out_channels, key=k3) if use_film else None

    def __call__(self, x: jnp.ndarray, t_emb=None, **kwargs) -> jnp.ndarray:
        out = self.activation(self.spectral_conv(x) + self.linear_skip(x))
        if self.film is not None and t_emb is not None:
            out = self.film(out, t_emb)
        return out


# ── full models ───────────────────────────────────────────────────────────────


class FFNO2d(eqx.Module):
    """Factorized FNO for 2-D PDE fields.

    Architecture: lift → n_layers × FactorizedSpectralBlock2d → project
    """

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    hidden_channels: int = eqx.field(static=True)

    lift: Linear
    blocks: list
    proj: Linear

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_modes: int,
        n_layers: int = 4,
        use_film: bool = False,
        emb_dim: int = 64,
        *,
        key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        keys = jax.random.split(key, n_layers + 2)
        self.lift = Linear(in_channels, hidden_channels, key=keys[0])
        self.blocks = [
            FactorizedSpectralBlock2d(
                hidden_channels,
                hidden_channels,
                n_modes,
                use_film=use_film,
                emb_dim=emb_dim,
                key=keys[1 + i],
            )
            for i in range(n_layers)
        ]
        self.proj = Linear(hidden_channels, out_channels, key=keys[-1])

    def __call__(self, x: jnp.ndarray, t_emb=None, **kwargs) -> jnp.ndarray:
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x, t_emb=t_emb)
        return self.proj(x)


class FFNO3d(eqx.Module):
    """Factorized FNO for 3-D volumetric PDE fields."""

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    hidden_channels: int = eqx.field(static=True)

    lift: Linear
    blocks: list
    proj: Linear

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_modes: int,
        n_layers: int = 4,
        use_film: bool = False,
        emb_dim: int = 64,
        *,
        key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        keys = jax.random.split(key, n_layers + 2)
        self.lift = Linear(in_channels, hidden_channels, key=keys[0])
        self.blocks = [
            FactorizedSpectralBlock3d(
                hidden_channels,
                hidden_channels,
                n_modes,
                use_film=use_film,
                emb_dim=emb_dim,
                key=keys[1 + i],
            )
            for i in range(n_layers)
        ]
        self.proj = Linear(hidden_channels, out_channels, key=keys[-1])

    def __call__(self, x: jnp.ndarray, t_emb=None, **kwargs) -> jnp.ndarray:
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x, t_emb=t_emb)
        return self.proj(x)
