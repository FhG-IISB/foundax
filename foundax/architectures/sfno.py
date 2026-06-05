"""
SFNO — Spherical Fourier Neural Operator (Bonev et al. 2023, ICML)
==================================================================

Paper: https://arxiv.org/abs/2306.03838
Code:  https://github.com/NVIDIA/torch-harmonics (PyTorch reference)

SFNO replaces FNO's FFT-based spectral convolution with a spherical
harmonic transform (SHT), giving rotational equivariance on the sphere
and long-horizon stability for global weather / climate fields where
FFT-based FNO is unstable past ~3 days of rollout.

Implementation notes
--------------------
Rather than depending on an external SHT library (``torch-harmonics`` is
PyTorch, and ``s2fft`` currently does not provide wheels for Python
3.14), this module ships a small pure-JAX real-valued SHT built on a
precomputed associated-Legendre table. The table is computed once at
construction via ``scipy.special.lpmn`` and stored as a static JAX array.

Two latitude grids are supported:
- ``"legendre-gauss"`` (default): ``nlat`` Gauss–Legendre nodes in
  ``cos θ``, weights are exact for polynomials up to degree ``2 nlat − 1``.
  Best numerical accuracy for SFNO.
- ``"equiangular"``: equispaced colatitudes ``θ_i = (i + ½) π / nlat``
  with naïve ``sin θ Δθ`` weights. More convenient for ERA5-style grids
  but quadrature is approximate.

All tensors follow the channel-last, unbatched foundax convention:
``(nlat, nlon, C)`` for fields, ``(L, L, C)`` complex for spectra.
Batch with ``jax.vmap`` externally.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional

import numpy as np
import scipy.special as sp

import jax
import jax.numpy as jnp
import equinox as eqx

from .common import get_activation
from .linear import Linear


# ── associated Legendre table + quadrature weights ──────────────────────────


def _legendre_table_and_weights(
    L: int, nlat: int, grid: str
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute the orthonormal associated-Legendre table.

    Returns:
        legendre: ``(L, L, nlat)`` float64 — ``N_lm · P_lm(cos θ_i)``.
            Entries with ``m > l`` are zero.
        weights:  ``(nlat,)`` float64 — quadrature weights such that
            ``∫ f(cos θ) dx ≈ Σ_i weights_i · f(cos θ_i)``.
    """
    if grid == "legendre-gauss":
        x, w = np.polynomial.legendre.leggauss(nlat)
        order = np.argsort(-x)  # decreasing cos θ ⇒ increasing θ
        cos_theta = x[order]
        weights = w[order]
    elif grid == "equiangular":
        theta = (np.arange(nlat) + 0.5) * np.pi / nlat
        cos_theta = np.cos(theta)
        weights = np.sin(theta) * (np.pi / nlat)
    else:
        raise ValueError(
            f"unknown grid {grid!r}; expected 'legendre-gauss' or 'equiangular'"
        )

    # Build the orthonormal-normalisation table once.
    # ``scipy.special.lpmv`` broadcasts over (m, l, x), so we compute the
    # full (L, L, nlat) Legendre table in a single call. Entries with
    # ``m > l`` are returned as 0 already.
    ls = np.arange(L)[:, None, None]
    ms = np.arange(L)[None, :, None]
    xs = cos_theta[None, None, :]
    Plm = sp.lpmv(ms, ls, xs)  # (L, L, nlat) — unnormalised

    # Orthonormal normalisation via gammaln to avoid overflow at high L.
    log_norm = 0.5 * (
        np.log(2 * ls + 1.0)
        - np.log(4 * np.pi)
        + sp.gammaln(ls - ms + 1)
        - sp.gammaln(ls + ms + 1)
    )
    # For m > l the gammaln of a negative integer is +inf → norm is 0
    # by construction, which correctly zeroes invalid entries.
    norm = np.where(ms <= ls, np.exp(log_norm), 0.0)

    legendre = (norm * Plm).astype(np.float64)
    return legendre, weights


# ── real-valued spherical harmonic transform ────────────────────────────────


class RealSHT2d(eqx.Module):
    """Real-valued spherical harmonic transform on a 2-D lat/lon grid.

    ``forward``: ``(nlat, nlon, C)`` real → ``(L, L, C)`` complex (m ≥ 0).
    ``inverse``: ``(L, L, C)`` complex → ``(nlat, nlon, C)`` real.

    Entries of the spectral coefficient array with ``m > l`` are
    semantically zero; the corresponding rows of the Legendre table are
    zero so those weights do not contribute.
    """

    legendre: jnp.ndarray  # (L, L, nlat) static (treated as buffer)
    weights: jnp.ndarray  # (nlat,)      static (treated as buffer)
    L: int = eqx.field(static=True)
    nlat: int = eqx.field(static=True)
    nlon: int = eqx.field(static=True)
    grid: str = eqx.field(static=True)

    def __init__(
        self,
        L: int,
        nlat: int,
        nlon: int,
        grid: Literal["legendre-gauss", "equiangular"] = "legendre-gauss",
    ):
        if L > nlat:
            raise ValueError(f"L ({L}) must be ≤ nlat ({nlat}) for stable SHT.")
        if L > nlon // 2 + 1:
            raise ValueError(
                f"L ({L}) must be ≤ nlon//2+1 ({nlon // 2 + 1}) for stable SHT."
            )
        legendre, weights = _legendre_table_and_weights(L, nlat, grid)
        self.legendre = jnp.asarray(legendre)
        self.weights = jnp.asarray(weights)
        self.L = L
        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """``(nlat, nlon, C) real → (L, L, C) complex``."""
        # FFT over longitude.
        X = jnp.fft.rfft(x, axis=1)  # (nlat, nlon//2 + 1, C)
        X = X[:, : self.L, :]
        scale = 2 * jnp.pi / self.nlon
        # ``legendre`` and ``weights`` encode the fixed sphere geometry;
        # they must not drift under training.
        legendre = jax.lax.stop_gradient(self.legendre)
        weights = jax.lax.stop_gradient(self.weights)
        return scale * jnp.einsum("lmi,i,imc->lmc", legendre, weights, X)

    def inverse(self, f_lm: jnp.ndarray) -> jnp.ndarray:
        """``(L, L, C) complex → (nlat, nlon, C) real``."""
        legendre = jax.lax.stop_gradient(self.legendre)
        F = jnp.einsum("lmi,lmc->imc", legendre, f_lm)  # (nlat, L, C) complex
        Lphi = self.nlon // 2 + 1
        F_full = jnp.zeros((self.nlat, Lphi, F.shape[-1]), dtype=F.dtype)
        F_full = F_full.at[:, : self.L, :].set(F)
        # irfft divides by nlon internally; multiply by nlon to undo and
        # leave the result on the same scale as the input grid would have
        # had under forward∘inverse for a single mode (so block weights
        # don't need to compensate).
        return jnp.fft.irfft(F_full, n=self.nlon, axis=1) * self.nlon


# ── spectral conv + block ───────────────────────────────────────────────────


class SphericalConv2d(eqx.Module):
    """Spherical spectral convolution: SHT → complex weights → inverse SHT.

    Input / output shape ``(nlat, nlon, C)`` real.
    """

    sht: RealSHT2d
    weight_real: jnp.ndarray  # (L, L, in_channels, out_channels)
    weight_imag: jnp.ndarray
    L: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        L: int,
        nlat: int,
        nlon: int,
        grid: Literal["legendre-gauss", "equiangular"] = "legendre-gauss",
        *,
        key,
    ):
        self.sht = RealSHT2d(L, nlat, nlon, grid=grid)
        self.L = L
        self.in_channels = in_channels
        self.out_channels = out_channels
        scale = 1.0 / (in_channels * out_channels)
        k1, k2 = jax.random.split(key)
        shape = (L, L, in_channels, out_channels)
        self.weight_real = scale * jax.random.uniform(
            k1, shape, minval=-1.0, maxval=1.0
        )
        self.weight_imag = scale * jax.random.uniform(
            k2, shape, minval=-1.0, maxval=1.0
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        f_lm = self.sht.forward(x)
        weight = self.weight_real + 1j * self.weight_imag
        f_lm_out = jnp.einsum("lmi,lmio->lmo", f_lm, weight)
        return self.sht.inverse(f_lm_out)


class SphericalBlock2d(eqx.Module):
    """One SFNO block: ``act(spherical_conv(x) + W x)``.

    Matches the FNO block pattern: a spectral path plus a pointwise
    linear skip, followed by a nonlinearity.

    Input / output shape ``(nlat, nlon, C)``. If ``in_channels !=
    out_channels`` the block changes channel width.
    """

    spectral: SphericalConv2d
    linear: Linear
    activation: Optional[Callable] = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        L: int,
        nlat: int,
        nlon: int,
        grid: Literal["legendre-gauss", "equiangular"] = "legendre-gauss",
        activation: str = "gelu",
        *,
        key,
    ):
        k1, k2 = jax.random.split(key)
        self.spectral = SphericalConv2d(
            in_channels, out_channels, L, nlat, nlon, grid=grid, key=k1
        )
        self.linear = Linear(in_channels, out_channels, key=k2)
        self.activation = get_activation(activation)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = self.spectral(x) + self.linear(x)
        return self.activation(y) if self.activation is not None else y


# ── top-level SFNO model ────────────────────────────────────────────────────


class SFNO2d(eqx.Module):
    """Spherical Fourier Neural Operator on a 2-D lat/lon grid.

    Standard lift → ``n_layers`` spherical spectral blocks → project
    pipeline. Inputs are ``(nlat, nlon, in_channels)``, outputs are
    ``(nlat, nlon, out_channels)``.

    Reference:
        Bonev et al. *Spherical Fourier Neural Operators: Learning
        Stable Dynamics on the Sphere* (ICML 2023).
        https://arxiv.org/abs/2306.03838
    """

    lift: Linear
    blocks: list
    project: Linear
    n_layers: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        L: int,
        nlat: int,
        nlon: int,
        n_layers: int = 4,
        grid: Literal["legendre-gauss", "equiangular"] = "legendre-gauss",
        activation: str = "gelu",
        *,
        key,
    ):
        keys = jax.random.split(key, n_layers + 2)
        self.lift = Linear(in_channels, hidden_channels, key=keys[0])
        self.blocks = [
            SphericalBlock2d(
                hidden_channels,
                hidden_channels,
                L,
                nlat,
                nlon,
                grid=grid,
                activation=activation,
                key=keys[1 + i],
            )
            for i in range(n_layers)
        ]
        self.project = Linear(hidden_channels, out_channels, key=keys[-1])
        self.n_layers = n_layers

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.project(x)


__all__ = [
    "RealSHT2d",
    "SphericalConv2d",
    "SphericalBlock2d",
    "SFNO2d",
]
