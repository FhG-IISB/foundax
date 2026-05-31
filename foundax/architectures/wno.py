# Wavelet Neural Operator backbone.
#
# Paper: "Wavelet Neural Operator for solving parametric partial differential
#         equations in computational mechanics problems"
#         Tripura & Chakraborty (2022) — https://arxiv.org/abs/2205.02191
# Code:  https://github.com/tapas-tripura/Wavelet-Neural-Operator  (PyTorch reference)
#
# This implementation uses pure JAX: Daubechies-8 (16-tap) low-pass filter
# is hardcoded; the high-pass is derived via the quadrature mirror.
# A multi-scale DWT decomposes the input, a learned linear mixes channels
# in the wavelet domain, and jax.image.resize restores spatial resolution.
#
# All tensors follow the channel-last, unbatched foundax convention:
#   1-D: (W, C)
#   2-D: (H, W, C)
#   3-D: (D, H, W, C)
#
# Spatial dimensions must be divisible by 2^n_scales.

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx

from .linear import Linear


# ── Daubechies-8 filter coefficients ─────────────────────────────────────────
# Low-pass (scaling) filter — 16 taps.  Values from the standard db8 table
# (identical to PyWavelets pywt.Wavelet('db8').dec_lo).

_DB8_LO = jnp.array(
    [
        -1.1747678400228192e-4,
        6.754494059985568e-4,
        -3.917403729959771e-4,
        -4.870352993010670e-3,
        8.746094047405776e-3,
        1.398102791701552e-2,
        -4.408825393106472e-2,
        -1.736930100202211e-2,
        1.287474266201860e-1,
        4.724845739979725e-4,
        -2.840155429624281e-1,
        -1.582910525602389e-2,
        5.853546836548691e-1,
        6.756307362980128e-1,
        3.128715909144659e-1,
        5.441584224308161e-2,
    ],
    dtype=jnp.float32,
)

# High-pass (wavelet) filter via quadrature mirror: h_hi[n] = (-1)^n * h_lo[L-1-n]
_DB8_HI = jnp.array(
    [(-1) ** n * float(_DB8_LO[15 - n]) for n in range(16)], dtype=jnp.float32
)


# ── 1-D wavelet helpers ───────────────────────────────────────────────────────


def _dwt_axis(
    x: jnp.ndarray, h_lo: jnp.ndarray, h_hi: jnp.ndarray, axis: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One level of 1-D DWT along *axis*.

    Returns ``(approx, detail)`` each with the target axis halved.
    Uses zero-boundary convolution (``mode='same'``).
    Input size along *axis* should be ≥ 16 and even.
    """
    x_moved = jnp.moveaxis(x, axis, 0)  # move target to front
    shape = x_moved.shape  # (Waxis, ...)
    x_2d = x_moved.reshape(shape[0], -1)  # (Waxis, rest)

    def _conv_down(col: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        # mode='full' gives length N+M-1; slice center N elements → N//2 after stride 2.
        # This is correct for any N, including N < M (unlike mode='same' which gives max(N,M)).
        full = jnp.convolve(col, h, mode="full")
        N = col.shape[0]
        start = (h.shape[0] - 1) // 2  # = 7 for db8
        return full[start : start + N : 2]

    approx_2d = jax.vmap(lambda col: _conv_down(col, h_lo), in_axes=1, out_axes=1)(x_2d)
    detail_2d = jax.vmap(lambda col: _conv_down(col, h_hi), in_axes=1, out_axes=1)(x_2d)

    approx = jnp.moveaxis(approx_2d.reshape(shape[0] // 2, *shape[1:]), 0, axis)
    detail = jnp.moveaxis(detail_2d.reshape(shape[0] // 2, *shape[1:]), 0, axis)
    return approx, detail


# ── multi-scale DWT for each dimensionality ───────────────────────────────────


def _multi_dwt1d(
    x: jnp.ndarray, h_lo: jnp.ndarray, h_hi: jnp.ndarray, n_scales: int
) -> tuple[jnp.ndarray, list]:
    """Multi-level 1-D DWT.  Returns (approx, [detail_1, ..., detail_n]).

    approx: (W//2^n, C).  detail_k: (W//2^k, C).
    """
    details = []
    cur = x
    for _ in range(n_scales):
        approx, detail = _dwt_axis(cur, h_lo, h_hi, axis=0)
        details.append(detail)
        cur = approx
    return cur, details


def _multi_dwt2d(
    x: jnp.ndarray, h_lo: jnp.ndarray, h_hi: jnp.ndarray, n_scales: int
) -> tuple[jnp.ndarray, list]:
    """Multi-level separable 2-D DWT.

    Returns (LL_n, [LH_1,HL_1,HH_1, LH_2,HL_2,HH_2, ...]).
    LL_n: (H//2^n, W//2^n, C).
    """
    all_details = []
    cur = x
    for _ in range(n_scales):
        lo_w, hi_w = _dwt_axis(cur, h_lo, h_hi, axis=1)  # along W
        LL, LH = _dwt_axis(lo_w, h_lo, h_hi, axis=0)  # LL, LH
        HL, HH = _dwt_axis(hi_w, h_lo, h_hi, axis=0)  # HL, HH
        all_details.extend([LH, HL, HH])
        cur = LL
    return cur, all_details


def _multi_dwt3d(
    x: jnp.ndarray, h_lo: jnp.ndarray, h_hi: jnp.ndarray, n_scales: int
) -> tuple[jnp.ndarray, list]:
    """Multi-level separable 3-D DWT.

    Returns (LLL_n, 7 detail subbands per scale).
    """
    all_details = []
    cur = x
    for _ in range(n_scales):
        lo_d, hi_d = _dwt_axis(cur, h_lo, h_hi, axis=0)
        lo_dh, hi_dh = _dwt_axis(lo_d, h_lo, h_hi, axis=1)
        LLL, LLH = _dwt_axis(lo_dh, h_lo, h_hi, axis=2)
        LHL, LHH = _dwt_axis(hi_dh, h_lo, h_hi, axis=2)
        lo_Dh, hi_Dh = _dwt_axis(hi_d, h_lo, h_hi, axis=1)
        HLL, HLH = _dwt_axis(lo_Dh, h_lo, h_hi, axis=2)
        HHL, HHH = _dwt_axis(hi_Dh, h_lo, h_hi, axis=2)
        all_details.extend([LLH, LHL, LHH, HLL, HLH, HHL, HHH])
        cur = LLL
    return cur, all_details


# ── pooling helpers ───────────────────────────────────────────────────────────


def _pool1d(x: jnp.ndarray, target_W: int) -> jnp.ndarray:
    W, C = x.shape
    f = W // target_W
    return x.reshape(target_W, f, C).mean(axis=1)


def _pool2d(x: jnp.ndarray, target_H: int, target_W: int) -> jnp.ndarray:
    H, W, C = x.shape
    fH, fW = H // target_H, W // target_W
    return x.reshape(target_H, fH, target_W, fW, C).mean(axis=(1, 3))


def _pool3d(x: jnp.ndarray, target_D: int, target_H: int, target_W: int) -> jnp.ndarray:
    D, H, W, C = x.shape
    fD, fH, fW = D // target_D, H // target_H, W // target_W
    return x.reshape(target_D, fD, target_H, fH, target_W, fW, C).mean(axis=(1, 3, 5))


# ── composable wavelet blocks (pipe-compatible) ───────────────────────────────


class WaveletBlock1d(eqx.Module):
    """Wavelet-domain operator block for 1-D fields: (W, C) → (W, C').

    Multi-level DWT decomposes x into approx + detail coefficients;
    these are pooled to the coarsest scale, mixed by a learned linear,
    then upsampled back to the original resolution.
    A linear skip connection is added before the activation.

    Accepts ``**kwargs`` for heterogeneous pipe compatibility.
    """

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    n_scales: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)

    wavelet_linear: Linear  # in_ch*(n_scales+1) → out_ch
    skip: Linear  # in_ch → out_ch

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_scales: int = 2,
        activation: Callable = jax.nn.gelu,
        *,
        key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_scales = n_scales
        self.activation = activation
        k1, k2 = jax.random.split(key)
        self.wavelet_linear = Linear(in_channels * (n_scales + 1), out_channels, key=k1)
        self.skip = Linear(in_channels, out_channels, key=k2)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        W = x.shape[0]
        approx, details = _multi_dwt1d(x, _DB8_LO, _DB8_HI, self.n_scales)
        target_W = approx.shape[0]
        pooled = [approx] + [_pool1d(d, target_W) for d in details]
        x_wave = jnp.concatenate(pooled, axis=-1)  # (target_W, in_ch*(n+1))
        x_wave = self.wavelet_linear(x_wave)  # (target_W, out_ch)
        x_wave = jax.image.resize(x_wave, (W, self.out_channels), method="linear")
        return self.activation(x_wave + self.skip(x))


class WaveletBlock2d(eqx.Module):
    """Wavelet-domain operator block for 2-D fields: (H, W, C) → (H, W, C').

    Uses a separable 2-D DWT (3 detail subbands per scale).
    Accepts ``**kwargs`` for heterogeneous pipe compatibility.
    """

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    n_scales: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)

    wavelet_linear: Linear  # in_ch*(3*n_scales+1) → out_ch
    skip: Linear

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_scales: int = 2,
        activation: Callable = jax.nn.gelu,
        *,
        key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_scales = n_scales
        self.activation = activation
        k1, k2 = jax.random.split(key)
        self.wavelet_linear = Linear(
            in_channels * (3 * n_scales + 1), out_channels, key=k1
        )
        self.skip = Linear(in_channels, out_channels, key=k2)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        H, W = x.shape[0], x.shape[1]
        approx, details = _multi_dwt2d(x, _DB8_LO, _DB8_HI, self.n_scales)
        tH, tW = approx.shape[0], approx.shape[1]
        pooled = [approx] + [_pool2d(d, tH, tW) for d in details]
        x_wave = jnp.concatenate(pooled, axis=-1)
        x_wave = self.wavelet_linear(x_wave)  # (tH, tW, out_ch)
        x_wave = jax.image.resize(x_wave, (H, W, self.out_channels), method="linear")
        return self.activation(x_wave + self.skip(x))


class WaveletBlock3d(eqx.Module):
    """Wavelet-domain operator block for 3-D fields: (D, H, W, C) → (D, H, W, C').

    Uses a separable 3-D DWT (7 detail subbands per scale).
    Accepts ``**kwargs`` for heterogeneous pipe compatibility.
    """

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    n_scales: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)

    wavelet_linear: Linear  # in_ch*(7*n_scales+1) → out_ch
    skip: Linear

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_scales: int = 2,
        activation: Callable = jax.nn.gelu,
        *,
        key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_scales = n_scales
        self.activation = activation
        k1, k2 = jax.random.split(key)
        self.wavelet_linear = Linear(
            in_channels * (7 * n_scales + 1), out_channels, key=k1
        )
        self.skip = Linear(in_channels, out_channels, key=k2)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        D, H, W = x.shape[0], x.shape[1], x.shape[2]
        approx, details = _multi_dwt3d(x, _DB8_LO, _DB8_HI, self.n_scales)
        tD, tH, tW = approx.shape[0], approx.shape[1], approx.shape[2]
        pooled = [approx] + [_pool3d(d, tD, tH, tW) for d in details]
        x_wave = jnp.concatenate(pooled, axis=-1)
        x_wave = self.wavelet_linear(x_wave)
        x_wave = jax.image.resize(x_wave, (D, H, W, self.out_channels), method="linear")
        return self.activation(x_wave + self.skip(x))


# ── full models ───────────────────────────────────────────────────────────────


class WNO1d(eqx.Module):
    """Wavelet Neural Operator for 1-D PDE fields.

    Architecture: lift → depth × WaveletBlock1d → project
    """

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    lift: Linear
    blocks: list
    proj: Linear

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_scales: int = 2,
        depth: int = 4,
        *,
        key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        keys = jax.random.split(key, depth + 2)
        self.lift = Linear(in_channels, hidden_channels, key=keys[0])
        self.blocks = [
            WaveletBlock1d(hidden_channels, hidden_channels, n_scales, key=keys[1 + i])
            for i in range(depth)
        ]
        self.proj = Linear(hidden_channels, out_channels, key=keys[-1])

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        return self.proj(x)


class WNO2d(eqx.Module):
    """Wavelet Neural Operator for 2-D PDE fields."""

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    lift: Linear
    blocks: list
    proj: Linear

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_scales: int = 2,
        depth: int = 4,
        *,
        key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        keys = jax.random.split(key, depth + 2)
        self.lift = Linear(in_channels, hidden_channels, key=keys[0])
        self.blocks = [
            WaveletBlock2d(hidden_channels, hidden_channels, n_scales, key=keys[1 + i])
            for i in range(depth)
        ]
        self.proj = Linear(hidden_channels, out_channels, key=keys[-1])

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        return self.proj(x)


class WNO3d(eqx.Module):
    """Wavelet Neural Operator for 3-D volumetric PDE fields."""

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    lift: Linear
    blocks: list
    proj: Linear

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_scales: int = 2,
        depth: int = 4,
        *,
        key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        keys = jax.random.split(key, depth + 2)
        self.lift = Linear(in_channels, hidden_channels, key=keys[0])
        self.blocks = [
            WaveletBlock3d(hidden_channels, hidden_channels, n_scales, key=keys[1 + i])
            for i in range(depth)
        ]
        self.proj = Linear(hidden_channels, out_channels, key=keys[-1])

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        return self.proj(x)
