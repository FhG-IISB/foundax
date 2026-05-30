"""Kolmogorov-Arnold Network layers and networks.

KAN replaces a Linear layer's scalar weights with learnable univariate functions
on each edge. Different variants choose different bases for those univariate
functions; this module implements the major ones. Each layer cites the
original publication (or canonical reference implementation) in its docstring.

Variants (with primary reference):

- ``KANLayer``         — B-spline + SiLU residual; Liu et al. 2024 (arXiv:2404.19756)
- ``EfficientKANLayer``— Reformulated B-spline; Blealtan/efficient-kan (GitHub, 2024)
- ``FastKANLayer``     — Gaussian RBF; Li 2024 (arXiv:2405.06721)
- ``FourierKANLayer``  — Fourier series; GistNoesis/FourierKAN (GitHub, 2024)
- ``ChebyshevKANLayer``— Chebyshev T_n; SS 2024 (arXiv:2405.07200)
- ``JacobiKANLayer``   — Jacobi P_n^(a,b); Aghaei 2024 (fKAN, arXiv:2406.07456)
- ``LegendreKANLayer`` — Legendre P_n (Jacobi α=β=0); Seydi 2024 (arXiv:2406.02583)
- ``WaveletKANLayer``  — Wavelet (Wav-KAN); Bozorgasl & Chen 2024 (arXiv:2405.12832)
- ``TaylorKANLayer``   — Taylor / power series; SeydiOptimisation/TaylorKAN (GitHub, 2024)
- ``HermiteKANLayer``  — Hermite He_n; Seydi 2024 (arXiv:2406.02583, OrthogPolyKAN family)
- ``LaguerreKANLayer`` — Laguerre L_n; Seydi 2024 (arXiv:2406.02583)
- ``BernsteinKANLayer``— Bernstein basis; Seydi 2024 (arXiv:2406.07456 et al.)
- ``ReLUKANLayer``     — (ReLU·ReLU)^k basis; Qiu et al. 2024 (arXiv:2406.02075)
- ``RationalKANLayer`` — Rational basis (rKAN family); Aghaei 2024 (arXiv:2406.14495)
- ``SincKANLayer``     — Sinc basis; Yang et al. 2024 (arXiv:2410.04096)
- ``GramKANLayer``     — Orthonormal-Legendre / Gram basis; OrthogPolyKAN
- ``BSRBFKANLayer``    — B-spline + RBF hybrid; Ta 2024 (arXiv:2406.11173)

Structural blocks:

- ``KANConv1d/2d/3d``   — KAN-convolution (Bodner et al. 2024, arXiv:2406.13155)
- ``KANSpectralBlock*d``— FNO spectral block with KAN channel mixer; foundax design,
                          inspired by Liu et al. 2024 + Li et al. FNO (arXiv:2010.08895)
- ``KANResBlock``       — Residual KAN block; common pattern in deep-KAN follow-ups
- ``KANAttentionBlock`` — Transformer block with KAN FFN; Yang & Wang 2024
                          ("Kolmogorov-Arnold Transformer", arXiv:2409.10594)

All layers map ``(..., in_features) -> (..., out_features)`` over arbitrary
leading dimensions, with ``in_features`` and ``out_features`` exposed as
static fields so :func:`foundax.block` sniffs them for the ``|`` pipe API.

Each variant also has a corresponding multi-layer network class
(``KAN``, ``FastKAN``, …) with the same constructor surface as
:class:`foundax.architectures.MLP`.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import equinox as eqx

from .linear import Linear


# =====================================================================
# Bases
#
# Each basis is a callable that takes an input array of shape (..., I) and
# returns an array of shape (..., I, G) where G is the basis size.
# The corresponding KAN layer then contracts the (I, G) axes with a weight
# of shape (O, I, G).
# =====================================================================


def _bspline_basis(x: jnp.ndarray, grid: jnp.ndarray, k: int) -> jnp.ndarray:
    """Cox-de Boor recursion for B-splines of order ``k`` on knot vector ``grid``.

    Args:
        x:    Input of shape (..., I).
        grid: Knot vector of shape (I, G + 2k + 1) — one knot row per input dim,
              already padded by ``k`` knots on each side.
        k:    Spline order (degree).

    Returns:
        B-spline basis of shape (..., I, G + k). With ``G`` interior grid
        intervals and order ``k``, there are ``G + k`` basis functions.
    """
    # x: (..., I), grid: (I, T) where T = G + 2k + 1.
    # Bring x to (..., I, 1) so it broadcasts against the knot axis.
    x_ = x[..., None]
    # Order-0 bases: indicator on [t_j, t_{j+1}).
    bases = ((x_ >= grid[..., :-1]) & (x_ < grid[..., 1:])).astype(x.dtype)
    # Recurse up to order k.
    for p in range(1, k + 1):
        left_num = x_ - grid[..., : -(p + 1)]
        left_den = grid[..., p:-1] - grid[..., : -(p + 1)]
        right_num = grid[..., p + 1 :] - x_
        right_den = grid[..., p + 1 :] - grid[..., 1:-p]
        # Avoid division by zero on degenerate knots.
        left = jnp.where(left_den > 0, left_num / jnp.where(left_den > 0, left_den, 1.0), 0.0)
        right = jnp.where(right_den > 0, right_num / jnp.where(right_den > 0, right_den, 1.0), 0.0)
        bases = left * bases[..., :-1] + right * bases[..., 1:]
    return bases


def _make_bspline_grid(in_features: int, grid_size: int, k: int,
                      grid_range: tuple[float, float]) -> jnp.ndarray:
    """Build a (in_features, grid_size + 2k + 1) extended uniform knot vector."""
    lo, hi = grid_range
    h = (hi - lo) / grid_size
    # Indices from -k to grid_size + k inclusive give grid_size + 2k + 1 knots.
    js = jnp.arange(-k, grid_size + k + 1, dtype=jnp.float32)
    knots = lo + js * h  # (grid_size + 2k + 1,)
    return jnp.broadcast_to(knots, (in_features, knots.shape[0]))


# ---- Basis modules used by Network classes for introspection / config ----


class BSplineBasis(eqx.Module):
    """B-spline basis evaluated on a fixed uniform grid."""

    grid: jnp.ndarray
    in_features: int = eqx.field(static=True)
    grid_size: int = eqx.field(static=True)
    spline_order: int = eqx.field(static=True)

    def __init__(self, in_features: int, grid_size: int = 5, spline_order: int = 3,
                 grid_range: tuple[float, float] = (-1.0, 1.0)):
        self.in_features = in_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid = _make_bspline_grid(in_features, grid_size, spline_order, grid_range)

    @property
    def size(self) -> int:
        return self.grid_size + self.spline_order

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return _bspline_basis(x, self.grid, self.spline_order)


class RBFBasis(eqx.Module):
    """Gaussian radial basis on a fixed uniform grid (FastKAN)."""

    centers: jnp.ndarray  # (G,)
    inv_h: jnp.ndarray  # scalar
    grid_size: int = eqx.field(static=True)

    def __init__(self, grid_size: int = 8, grid_range: tuple[float, float] = (-2.0, 2.0)):
        self.grid_size = grid_size
        self.centers = jnp.linspace(grid_range[0], grid_range[1], grid_size)
        h = (grid_range[1] - grid_range[0]) / (grid_size - 1)
        self.inv_h = jnp.asarray(1.0 / h)

    @property
    def size(self) -> int:
        return self.grid_size

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # (..., I, G)
        z = (x[..., None] - self.centers) * self.inv_h
        return jnp.exp(-(z ** 2))


class FourierBasis(eqx.Module):
    """sin/cos Fourier series basis."""

    num_frequencies: int = eqx.field(static=True)

    def __init__(self, num_frequencies: int = 8):
        self.num_frequencies = num_frequencies

    @property
    def size(self) -> int:
        return 2 * self.num_frequencies

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        ks = jnp.arange(1, self.num_frequencies + 1, dtype=x.dtype)
        ang = jnp.pi * x[..., None] * ks  # (..., I, K)
        return jnp.concatenate([jnp.cos(ang), jnp.sin(ang)], axis=-1)


class ChebyshevBasis(eqx.Module):
    """Chebyshev polynomials of the first kind T_n, evaluated via recurrence.

    Inputs are squashed through ``tanh`` to the stable interval ``[-1, 1]``.
    """

    degree: int = eqx.field(static=True)

    def __init__(self, degree: int = 5):
        self.degree = degree

    @property
    def size(self) -> int:
        return self.degree + 1

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.tanh(x)
        out = [jnp.ones_like(x)]
        if self.degree >= 1:
            out.append(x)
        for _ in range(2, self.degree + 1):
            out.append(2 * x * out[-1] - out[-2])
        return jnp.stack(out, axis=-1)


class JacobiBasis(eqx.Module):
    """Jacobi polynomials P_n^(a,b) via three-term recurrence; inputs squashed by tanh."""

    degree: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)

    def __init__(self, degree: int = 5, alpha: float = 1.0, beta: float = 1.0):
        self.degree = degree
        self.alpha = alpha
        self.beta = beta

    @property
    def size(self) -> int:
        return self.degree + 1

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        a, b = self.alpha, self.beta
        x = jnp.tanh(x)
        p = [jnp.ones_like(x)]
        if self.degree >= 1:
            p.append(0.5 * ((a - b) + (a + b + 2.0) * x))
        for n in range(1, self.degree):
            n_f = float(n)
            c1 = 2 * (n_f + 1) * (n_f + a + b + 1) * (2 * n_f + a + b)
            c2 = (2 * n_f + a + b + 1) * (a * a - b * b)
            c3 = (2 * n_f + a + b) * (2 * n_f + a + b + 1) * (2 * n_f + a + b + 2)
            c4 = 2 * (n_f + a) * (n_f + b) * (2 * n_f + a + b + 2)
            p.append(((c2 + c3 * x) * p[-1] - c4 * p[-2]) / c1)
        return jnp.stack(p, axis=-1)


class LegendreBasis(JacobiBasis):
    """Legendre polynomials P_n — Jacobi with alpha = beta = 0."""

    def __init__(self, degree: int = 5):
        super().__init__(degree=degree, alpha=0.0, beta=0.0)


class WaveletBasis(eqx.Module):
    """Wavelet basis with selectable mother wavelet, applied at scales 2**(-j).

    Supported wavelets: ``mexican_hat``, ``morlet``, ``shannon``, ``dog``.
    Produces one basis function per scale (so ``size == num_scales``).
    """

    num_scales: int = eqx.field(static=True)
    wavelet_type: str = eqx.field(static=True)

    def __init__(self, num_scales: int = 6, wavelet_type: str = "mexican_hat"):
        if wavelet_type not in ("mexican_hat", "morlet", "shannon", "dog"):
            raise ValueError(f"Unknown wavelet_type '{wavelet_type}'")
        self.num_scales = num_scales
        self.wavelet_type = wavelet_type

    @property
    def size(self) -> int:
        return self.num_scales

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scales = 2.0 ** (-jnp.arange(self.num_scales, dtype=x.dtype))
        z = x[..., None] * scales  # (..., I, S)
        if self.wavelet_type == "mexican_hat":
            return (1.0 - z * z) * jnp.exp(-0.5 * z * z)
        if self.wavelet_type == "morlet":
            return jnp.cos(5.0 * z) * jnp.exp(-0.5 * z * z)
        if self.wavelet_type == "shannon":
            # Sinc-like Shannon wavelet; pi*z handles z==0 via jnp.sinc convention.
            return jnp.sinc(z) - jnp.sinc(z / 2.0)
        # "dog" — derivative of Gaussian
        return -z * jnp.exp(-0.5 * z * z)


class TaylorBasis(eqx.Module):
    """Truncated Taylor / power basis {1, x, x^2, ...}; inputs squashed by tanh."""

    degree: int = eqx.field(static=True)

    def __init__(self, degree: int = 4):
        self.degree = degree

    @property
    def size(self) -> int:
        return self.degree + 1

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.tanh(x)
        out = [jnp.ones_like(x)]
        for _ in range(self.degree):
            out.append(out[-1] * x)
        return jnp.stack(out, axis=-1)


class HermiteBasis(eqx.Module):
    """Probabilist Hermite polynomials He_n via recurrence He_{n+1} = x He_n - n He_{n-1}.

    Inputs squashed by ``tanh`` to keep magnitudes bounded.
    """

    degree: int = eqx.field(static=True)

    def __init__(self, degree: int = 5):
        self.degree = degree

    @property
    def size(self) -> int:
        return self.degree + 1

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.tanh(x)
        h = [jnp.ones_like(x), x]
        for n in range(1, self.degree):
            h.append(x * h[-1] - n * h[-2])
        return jnp.stack(h[: self.degree + 1], axis=-1)


class LaguerreBasis(eqx.Module):
    """Laguerre polynomials L_n via recurrence (n+1)L_{n+1} = (2n+1-x)L_n - n L_{n-1}.

    Inputs squashed to ``[0, ~4]`` via ``2 * (1 + tanh(x))`` so the polynomials
    stay in their natural domain.
    """

    degree: int = eqx.field(static=True)

    def __init__(self, degree: int = 5):
        self.degree = degree

    @property
    def size(self) -> int:
        return self.degree + 1

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = 2.0 * (1.0 + jnp.tanh(x))
        L = [jnp.ones_like(x), 1.0 - x]
        for n in range(1, self.degree):
            n_f = float(n)
            L.append(((2 * n_f + 1 - x) * L[-1] - n_f * L[-2]) / (n_f + 1))
        return jnp.stack(L[: self.degree + 1], axis=-1)


class BernsteinBasis(eqx.Module):
    """Bernstein polynomials ``B_{i,n}(x) = C(n,i) x^i (1-x)^(n-i)`` on ``[0, 1]``.

    Inputs squashed to ``[0, 1]`` via ``sigmoid``. The basis sums to 1 (partition
    of unity), inherited from the standard Bernstein property.
    """

    degree: int = eqx.field(static=True)

    def __init__(self, degree: int = 5):
        self.degree = degree

    @property
    def size(self) -> int:
        return self.degree + 1

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        u = jax.nn.sigmoid(x)
        n = self.degree
        # Compute log-coefficients once, evaluate in log-space for numerical safety.
        ks = jnp.arange(n + 1, dtype=u.dtype)
        # log C(n, k) via lgamma
        log_binom = (jax.lax.lgamma(n + 1.0)
                     - jax.lax.lgamma(ks + 1.0)
                     - jax.lax.lgamma(n - ks + 1.0))
        eps = jnp.asarray(1e-30, dtype=u.dtype)
        log_u = jnp.log(jnp.clip(u, eps, 1.0))
        log_1mu = jnp.log(jnp.clip(1.0 - u, eps, 1.0))
        # (..., I, K) = log_binom[K] + k*log(u) + (n-k)*log(1-u)
        log_basis = (log_binom
                     + ks * log_u[..., None]
                     + (n - ks) * log_1mu[..., None])
        return jnp.exp(log_basis)


class ReLUKANBasis(eqx.Module):
    """ReLU-KAN / FasterKAN basis: products of ReLUs on a uniform grid.

    For grid points ``a_j < b_j`` with ``b_j = a_j + 2/G``, the basis is
    ``phi_j(x) = (relu(x - a_j) * relu(b_j - x)) ** order``. Compactly supported
    and infinitely differentiable inside each interval.
    """

    grid_size: int = eqx.field(static=True)
    order: int = eqx.field(static=True)
    grid_a: jnp.ndarray
    grid_b: jnp.ndarray
    norm: jnp.ndarray  # scalar

    def __init__(self, grid_size: int = 8, order: int = 2,
                 grid_range: tuple[float, float] = (-1.0, 1.0)):
        self.grid_size = grid_size
        self.order = order
        lo, hi = grid_range
        h = (hi - lo) / grid_size
        centers = jnp.linspace(lo + 0.5 * h, hi - 0.5 * h, grid_size)
        self.grid_a = centers - h
        self.grid_b = centers + h
        # Normalize so peak value of (relu(h)*relu(h))**order = 1.
        self.norm = jnp.asarray(1.0 / (h ** (2 * order)))

    @property
    def size(self) -> int:
        return self.grid_size

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        z = x[..., None]
        l = jax.nn.relu(z - self.grid_a)
        r = jax.nn.relu(self.grid_b - z)
        return self.norm * (l * r) ** self.order


class RationalBasis(eqx.Module):
    """Rational (Padé-Chebyshev) basis: ``T_k(tanh(x)) / (1 + tanh(x)**2)``.

    Approximates rational-function regression cleanly; each basis function is
    bounded on the full real line.
    """

    degree: int = eqx.field(static=True)

    def __init__(self, degree: int = 5):
        self.degree = degree

    @property
    def size(self) -> int:
        return self.degree + 1

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        xt = jnp.tanh(x)
        denom = 1.0 + xt * xt
        # Chebyshev recurrence over xt
        T = [jnp.ones_like(xt), xt]
        for _ in range(2, self.degree + 1):
            T.append(2 * xt * T[-1] - T[-2])
        out = jnp.stack(T[: self.degree + 1], axis=-1)
        return out / denom[..., None]


class SincBasis(eqx.Module):
    """Sinc basis: ``phi_k(x) = sinc((x - c_k) / h)`` on a uniform grid."""

    grid_size: int = eqx.field(static=True)
    centers: jnp.ndarray
    inv_h: jnp.ndarray  # scalar

    def __init__(self, grid_size: int = 8, grid_range: tuple[float, float] = (-2.0, 2.0)):
        self.grid_size = grid_size
        self.centers = jnp.linspace(grid_range[0], grid_range[1], grid_size)
        h = (grid_range[1] - grid_range[0]) / (grid_size - 1)
        self.inv_h = jnp.asarray(1.0 / h)

    @property
    def size(self) -> int:
        return self.grid_size

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        z = (x[..., None] - self.centers) * self.inv_h
        return jnp.sinc(z)


class GramBasis(eqx.Module):
    """Gram (orthonormal Legendre) polynomials ``sqrt(2n+1) * P_n(x)`` on ``[-1, 1]``.

    Continuous-limit Gram polynomials coincide with normalised Legendre.
    Inputs squashed by ``tanh``.
    """

    degree: int = eqx.field(static=True)

    def __init__(self, degree: int = 5):
        self.degree = degree

    @property
    def size(self) -> int:
        return self.degree + 1

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.tanh(x)
        # Legendre recurrence (n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}.
        P = [jnp.ones_like(x), x]
        for n in range(1, self.degree):
            n_f = float(n)
            P.append(((2 * n_f + 1) * x * P[-1] - n_f * P[-2]) / (n_f + 1))
        # Normalise to unit L2 norm on [-1, 1]: ||P_n||^2 = 2/(2n+1).
        ns = jnp.arange(self.degree + 1, dtype=x.dtype)
        scale = jnp.sqrt((2.0 * ns + 1.0) / 2.0)
        return jnp.stack(P[: self.degree + 1], axis=-1) * scale


class BSRBFBasis(eqx.Module):
    """Hybrid B-spline + Gaussian-RBF basis.

    Output is the concatenation along the basis axis, so
    ``size == bspline.size + rbf.size``.
    """

    bspline: BSplineBasis
    rbf: RBFBasis

    def __init__(self, in_features: int, grid_size: int = 5, spline_order: int = 3,
                 rbf_grid_size: int = 8,
                 grid_range: tuple[float, float] = (-1.0, 1.0),
                 rbf_grid_range: tuple[float, float] = (-2.0, 2.0)):
        self.bspline = BSplineBasis(in_features, grid_size, spline_order, grid_range)
        self.rbf = RBFBasis(rbf_grid_size, rbf_grid_range)

    @property
    def size(self) -> int:
        return self.bspline.size + self.rbf.size

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([self.bspline(x), self.rbf(x)], axis=-1)


# =====================================================================
# Generic basis-driven KAN layer
#
# Every variant except the original (which keeps the SiLU residual) is a thin
# wrapper around _BasisKANLayer with a different basis and skip behaviour.
# =====================================================================


def _einsum_contract(weight: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """Contract (O, I, G) weights with (..., I, G) basis activations to (..., O)."""
    return jnp.einsum("oig,...ig->...o", weight, phi)


class _BasisKANLayer(eqx.Module):
    """Shared implementation: phi(x) -> contract with edge weights, optional skip.

    Subclasses set ``basis`` and ``skip`` (the latter as ``None`` to disable).
    """

    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    basis: eqx.Module
    spline_weight: jnp.ndarray  # (out_features, in_features, basis.size)
    skip: Optional[Linear]
    skip_activation: Optional[Callable] = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        basis: eqx.Module,
        *,
        skip: bool = False,
        skip_activation: Optional[Callable] = jax.nn.silu,
        scale: float = 1.0,
        key: jax.Array,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.basis = basis
        size = basis.size
        k1, k2 = jax.random.split(key)
        # Match a Linear-style init: std ~ scale / sqrt(in_features * size).
        std = scale / jnp.sqrt(jnp.asarray(in_features * size, dtype=jnp.float32))
        self.spline_weight = jax.random.normal(k1, (out_features, in_features, size)) * std
        if skip:
            self.skip = Linear(in_features, out_features, key=k2)
            self.skip_activation = skip_activation
        else:
            self.skip = None
            self.skip_activation = None

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        phi = self.basis(x)  # (..., I, G)
        y = _einsum_contract(self.spline_weight, phi)  # (..., O)
        if self.skip is not None:
            res = self.skip_activation(x) if self.skip_activation is not None else x
            y = y + self.skip(res)
        return y


# ---- Public per-variant layers (thin constructors over _BasisKANLayer) ----


class KANLayer(_BasisKANLayer):
    """Original B-spline KAN layer with SiLU residual.

    Reference:
        Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M.,
        Hou, T. Y., & Tegmark, M. (2024). *KAN: Kolmogorov-Arnold Networks*.
        arXiv:2404.19756. Code: github.com/KindXiaoming/pykan.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        scale: float = 1.0,
        *,
        key: jax.Array,
    ):
        basis = BSplineBasis(in_features, grid_size, spline_order, grid_range)
        super().__init__(in_features, out_features, basis, skip=True,
                         skip_activation=jax.nn.silu, scale=scale, key=key)


class EfficientKANLayer(_BasisKANLayer):
    """B-spline KAN with the Blealtan/efficient-kan parameterisation.

    Functionally equivalent surface to :class:`KANLayer` for our purposes;
    the difference in the reference implementation is computational
    (memory layout, scaled-spline weights). We expose the same forward
    behaviour so the layer is interchangeable in pipes.

    Reference:
        Blealtan. *efficient-kan* (2024). github.com/Blealtan/efficient-kan.
        A memory- and speed-optimised reformulation of Liu et al. 2024
        (arXiv:2404.19756).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        scale: float = 1.0,
        *,
        key: jax.Array,
    ):
        basis = BSplineBasis(in_features, grid_size, spline_order, grid_range)
        super().__init__(in_features, out_features, basis, skip=True,
                         skip_activation=jax.nn.silu, scale=scale, key=key)


class FastKANLayer(_BasisKANLayer):
    """Gaussian-RBF KAN.

    Reference:
        Li, Z. (2024). *Kolmogorov-Arnold Networks are Radial Basis Function
        Networks*. arXiv:2405.06721. Code: github.com/ZiyaoLi/fast-kan.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        grid_range: tuple[float, float] = (-2.0, 2.0),
        use_skip: bool = True,
        scale: float = 1.0,
        *,
        key: jax.Array,
    ):
        basis = RBFBasis(grid_size, grid_range)
        super().__init__(in_features, out_features, basis, skip=use_skip,
                         skip_activation=jax.nn.silu, scale=scale, key=key)


class FourierKANLayer(_BasisKANLayer):
    """Fourier-series KAN.

    Reference:
        GistNoesis. *FourierKAN* (2024). github.com/GistNoesis/FourierKAN.
        A community-popularised KAN variant using a fixed sin/cos basis
        rather than B-splines.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_frequencies: int = 8,
        scale: float = 1.0,
        *,
        key: jax.Array,
    ):
        basis = FourierBasis(num_frequencies)
        super().__init__(in_features, out_features, basis, skip=False,
                         scale=scale, key=key)


class ChebyshevKANLayer(_BasisKANLayer):
    """Chebyshev-polynomial KAN.

    Reference:
        SS, S. S. (2024). *Chebyshev Polynomial-Based Kolmogorov-Arnold
        Networks: An Efficient Architecture for Nonlinear Function
        Approximation*. arXiv:2405.07200.
        Code: github.com/SynodicMonth/ChebyKAN.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 5,
        scale: float = 1.0,
        *,
        key: jax.Array,
    ):
        basis = ChebyshevBasis(degree)
        super().__init__(in_features, out_features, basis, skip=False,
                         scale=scale, key=key)


class JacobiKANLayer(_BasisKANLayer):
    """Jacobi-polynomial KAN. Alpha=beta=1.0 by default; set both to 0 for Legendre.

    Reference:
        Aghaei, A. A. (2024). *fKAN: Fractional Kolmogorov-Arnold Networks
        with trainable Jacobi basis functions*. arXiv:2406.07456.
        Code: github.com/alirezaafzalaghaei/fKAN.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 5,
        alpha: float = 1.0,
        beta: float = 1.0,
        scale: float = 1.0,
        *,
        key: jax.Array,
    ):
        basis = JacobiBasis(degree, alpha, beta)
        super().__init__(in_features, out_features, basis, skip=False,
                         scale=scale, key=key)


class LegendreKANLayer(_BasisKANLayer):
    """Legendre-polynomial KAN (Jacobi with alpha = beta = 0).

    Reference:
        Seydi, S. T. (2024). *Unveiling the Power of Wavelets: A Wavelet-based
        Kolmogorov-Arnold Network for Hyperspectral Image Classification* —
        appendix surveys the orthogonal-polynomial KAN family including
        Legendre. arXiv:2406.07869. See also OrthogPolyKAN family
        (arXiv:2406.02583).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 5,
        scale: float = 1.0,
        *,
        key: jax.Array,
    ):
        basis = LegendreBasis(degree)
        super().__init__(in_features, out_features, basis, skip=False,
                         scale=scale, key=key)


class WaveletKANLayer(_BasisKANLayer):
    """Wavelet KAN (mexican_hat / morlet / shannon / dog).

    Reference:
        Bozorgasl, Z. & Chen, H. (2024). *Wav-KAN: Wavelet
        Kolmogorov-Arnold Networks*. arXiv:2405.12832.
        Code: github.com/zavareh1/Wav-KAN.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_scales: int = 6,
        wavelet_type: str = "mexican_hat",
        scale: float = 1.0,
        *,
        key: jax.Array,
    ):
        basis = WaveletBasis(num_scales, wavelet_type)
        super().__init__(in_features, out_features, basis, skip=False,
                         scale=scale, key=key)


class TaylorKANLayer(_BasisKANLayer):
    """Truncated-Taylor KAN.

    Reference:
        Muyuzhierchengse. *TaylorKAN* (2024).
        github.com/Muyuzhierchengse/TaylorKAN. Community-popularised KAN
        variant using a plain truncated power-series basis.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 4,
        scale: float = 1.0,
        *,
        key: jax.Array,
    ):
        basis = TaylorBasis(degree)
        super().__init__(in_features, out_features, basis, skip=False,
                         scale=scale, key=key)


class HermiteKANLayer(_BasisKANLayer):
    """Hermite-polynomial KAN (probabilist Hermite He_n).

    Reference:
        Seydi, S. T. (2024). *Exploring the Potential of Polynomial Basis
        Functions in Kolmogorov-Arnold Networks*. arXiv:2406.02583.
        Code: github.com/seydi1370/Basis_Functions.
    """

    def __init__(self, in_features: int, out_features: int, degree: int = 5,
                 scale: float = 1.0, *, key: jax.Array):
        super().__init__(in_features, out_features, HermiteBasis(degree),
                         skip=False, scale=scale, key=key)


class LaguerreKANLayer(_BasisKANLayer):
    """Laguerre-polynomial KAN.

    Reference:
        Seydi, S. T. (2024). *Exploring the Potential of Polynomial Basis
        Functions in Kolmogorov-Arnold Networks*. arXiv:2406.02583.
    """

    def __init__(self, in_features: int, out_features: int, degree: int = 5,
                 scale: float = 1.0, *, key: jax.Array):
        super().__init__(in_features, out_features, LaguerreBasis(degree),
                         skip=False, scale=scale, key=key)


class BernsteinKANLayer(_BasisKANLayer):
    """Bernstein-polynomial KAN (basis sums to 1 on [0, 1] after sigmoid squash).

    Reference:
        Seydi, S. T. (2024). *Exploring the Potential of Polynomial Basis
        Functions in Kolmogorov-Arnold Networks*. arXiv:2406.02583.
    """

    def __init__(self, in_features: int, out_features: int, degree: int = 5,
                 scale: float = 1.0, *, key: jax.Array):
        super().__init__(in_features, out_features, BernsteinBasis(degree),
                         skip=False, scale=scale, key=key)


class ReLUKANLayer(_BasisKANLayer):
    """ReLU-KAN / FasterKAN — compactly-supported (ReLU * ReLU)^order basis.

    Reference:
        Qiu, Q., Zhu, T., Gong, H., Chen, L., & Ning, H. (2024). *ReLU-KAN:
        New Kolmogorov-Arnold Networks that Only Need Matrix Addition, Dot
        Multiplication, and ReLU*. arXiv:2406.02075.
        Related: Delis, A. *FasterKAN* (2024) — github.com/AthanasiosDelis/faster-kan.
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 8,
                 order: int = 2,
                 grid_range: tuple[float, float] = (-1.0, 1.0),
                 scale: float = 1.0, *, key: jax.Array):
        super().__init__(in_features, out_features,
                         ReLUKANBasis(grid_size, order, grid_range),
                         skip=True, skip_activation=jax.nn.silu,
                         scale=scale, key=key)


class RationalKANLayer(_BasisKANLayer):
    """Rational-Chebyshev (Padé-style) KAN.

    Reference:
        Aghaei, A. A. (2024). *rKAN: Rational Kolmogorov-Arnold Networks*.
        arXiv:2406.14495. Code: github.com/alirezaafzalaghaei/rKAN.
    """

    def __init__(self, in_features: int, out_features: int, degree: int = 5,
                 scale: float = 1.0, *, key: jax.Array):
        super().__init__(in_features, out_features, RationalBasis(degree),
                         skip=False, scale=scale, key=key)


class SincKANLayer(_BasisKANLayer):
    """Sinc-basis KAN.

    Reference:
        Yu, R., Yu, W., & Wang, X. (2024). *SincKAN: Function Approximation
        with Sinc Interpolation Inside Kolmogorov-Arnold Networks*.
        arXiv:2410.04096 (see also the broader sinc-NN literature for the
        underlying basis).
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 8,
                 grid_range: tuple[float, float] = (-2.0, 2.0),
                 scale: float = 1.0, *, key: jax.Array):
        super().__init__(in_features, out_features,
                         SincBasis(grid_size, grid_range),
                         skip=False, scale=scale, key=key)


class GramKANLayer(_BasisKANLayer):
    """Gram-polynomial (orthonormal Legendre) KAN.

    Reference:
        Igelnik, B. & Parikh, N. (2003). *Kolmogorov's Spline Network* —
        background on discrete-orthogonal Gram polynomials; the continuous
        limit equals normalised Legendre. KAN application: Seydi 2024
        (arXiv:2406.02583).
    """

    def __init__(self, in_features: int, out_features: int, degree: int = 5,
                 scale: float = 1.0, *, key: jax.Array):
        super().__init__(in_features, out_features, GramBasis(degree),
                         skip=False, scale=scale, key=key)


class BSRBFKANLayer(_BasisKANLayer):
    """Hybrid B-spline + RBF KAN (concatenated bases).

    Reference:
        Ta, H. T. (2024). *BSRBF-KAN: A combination of B-splines and Radial
        Basis Functions in Kolmogorov-Arnold Networks*. arXiv:2406.11173.
        Code: github.com/hoangthangta/BSRBF_KAN.
    """

    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 5, spline_order: int = 3,
                 rbf_grid_size: int = 8,
                 grid_range: tuple[float, float] = (-1.0, 1.0),
                 rbf_grid_range: tuple[float, float] = (-2.0, 2.0),
                 scale: float = 1.0, *, key: jax.Array):
        super().__init__(
            in_features, out_features,
            BSRBFBasis(in_features, grid_size, spline_order, rbf_grid_size,
                       grid_range, rbf_grid_range),
            skip=True, skip_activation=jax.nn.silu,
            scale=scale, key=key,
        )


# =====================================================================
# Network classes — stacked layers, MLP-style constructor surface
# =====================================================================


def _resolve_hidden(in_features: int, hidden_dims, num_layers: int) -> list[int]:
    if isinstance(hidden_dims, int):
        return [hidden_dims] * num_layers
    return list(hidden_dims)


class _StackedKAN(eqx.Module):
    """Generic ``num_layers`` stack of KAN layers; subclasses pick the layer class."""

    in_features: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    layers: list

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x


def _build_stack(layer_cls, in_features, output_dim, hidden_dims, num_layers,
                 key, **layer_kwargs):
    widths = _resolve_hidden(in_features, hidden_dims, num_layers)
    dims = [in_features] + widths + [output_dim]
    keys = jax.random.split(key, len(dims) - 1)
    return [
        layer_cls(dims[i], dims[i + 1], key=keys[i], **layer_kwargs)
        for i in range(len(dims) - 1)
    ]


class KAN(_StackedKAN):
    """Stacked B-spline KAN (original)."""

    def __init__(self, in_features: int, output_dim: int = 1,
                 hidden_dims=64, num_layers: int = 2,
                 grid_size: int = 5, spline_order: int = 3,
                 grid_range: tuple[float, float] = (-1.0, 1.0),
                 *, key: jax.Array):
        self.in_features = in_features
        self.output_dim = output_dim
        self.layers = _build_stack(
            KANLayer, in_features, output_dim, hidden_dims, num_layers, key,
            grid_size=grid_size, spline_order=spline_order, grid_range=grid_range,
        )


class EfficientKAN(_StackedKAN):
    """Stacked EfficientKAN."""

    def __init__(self, in_features: int, output_dim: int = 1,
                 hidden_dims=64, num_layers: int = 2,
                 grid_size: int = 5, spline_order: int = 3,
                 grid_range: tuple[float, float] = (-1.0, 1.0),
                 *, key: jax.Array):
        self.in_features = in_features
        self.output_dim = output_dim
        self.layers = _build_stack(
            EfficientKANLayer, in_features, output_dim, hidden_dims, num_layers, key,
            grid_size=grid_size, spline_order=spline_order, grid_range=grid_range,
        )


class FastKAN(_StackedKAN):
    """Stacked FastKAN (RBF)."""

    def __init__(self, in_features: int, output_dim: int = 1,
                 hidden_dims=64, num_layers: int = 2,
                 grid_size: int = 8,
                 grid_range: tuple[float, float] = (-2.0, 2.0),
                 *, key: jax.Array):
        self.in_features = in_features
        self.output_dim = output_dim
        self.layers = _build_stack(
            FastKANLayer, in_features, output_dim, hidden_dims, num_layers, key,
            grid_size=grid_size, grid_range=grid_range,
        )


class FourierKAN(_StackedKAN):
    """Stacked FourierKAN."""

    def __init__(self, in_features: int, output_dim: int = 1,
                 hidden_dims=64, num_layers: int = 2,
                 num_frequencies: int = 8,
                 *, key: jax.Array):
        self.in_features = in_features
        self.output_dim = output_dim
        self.layers = _build_stack(
            FourierKANLayer, in_features, output_dim, hidden_dims, num_layers, key,
            num_frequencies=num_frequencies,
        )


class ChebyshevKAN(_StackedKAN):
    """Stacked ChebyshevKAN."""

    def __init__(self, in_features: int, output_dim: int = 1,
                 hidden_dims=64, num_layers: int = 2,
                 degree: int = 5,
                 *, key: jax.Array):
        self.in_features = in_features
        self.output_dim = output_dim
        self.layers = _build_stack(
            ChebyshevKANLayer, in_features, output_dim, hidden_dims, num_layers, key,
            degree=degree,
        )


class JacobiKAN(_StackedKAN):
    """Stacked JacobiKAN."""

    def __init__(self, in_features: int, output_dim: int = 1,
                 hidden_dims=64, num_layers: int = 2,
                 degree: int = 5, alpha: float = 1.0, beta: float = 1.0,
                 *, key: jax.Array):
        self.in_features = in_features
        self.output_dim = output_dim
        self.layers = _build_stack(
            JacobiKANLayer, in_features, output_dim, hidden_dims, num_layers, key,
            degree=degree, alpha=alpha, beta=beta,
        )


class LegendreKAN(_StackedKAN):
    """Stacked LegendreKAN."""

    def __init__(self, in_features: int, output_dim: int = 1,
                 hidden_dims=64, num_layers: int = 2,
                 degree: int = 5,
                 *, key: jax.Array):
        self.in_features = in_features
        self.output_dim = output_dim
        self.layers = _build_stack(
            LegendreKANLayer, in_features, output_dim, hidden_dims, num_layers, key,
            degree=degree,
        )


class WaveletKAN(_StackedKAN):
    """Stacked WaveletKAN."""

    def __init__(self, in_features: int, output_dim: int = 1,
                 hidden_dims=64, num_layers: int = 2,
                 num_scales: int = 6, wavelet_type: str = "mexican_hat",
                 *, key: jax.Array):
        self.in_features = in_features
        self.output_dim = output_dim
        self.layers = _build_stack(
            WaveletKANLayer, in_features, output_dim, hidden_dims, num_layers, key,
            num_scales=num_scales, wavelet_type=wavelet_type,
        )


class TaylorKAN(_StackedKAN):
    """Stacked TaylorKAN."""

    def __init__(self, in_features: int, output_dim: int = 1,
                 hidden_dims=64, num_layers: int = 2,
                 degree: int = 4,
                 *, key: jax.Array):
        self.in_features = in_features
        self.output_dim = output_dim
        self.layers = _build_stack(
            TaylorKANLayer, in_features, output_dim, hidden_dims, num_layers, key,
            degree=degree,
        )


class HermiteKAN(_StackedKAN):
    """Stacked HermiteKAN."""

    def __init__(self, in_features: int, output_dim: int = 1,
                 hidden_dims=64, num_layers: int = 2, degree: int = 5,
                 *, key: jax.Array):
        self.in_features = in_features
        self.output_dim = output_dim
        self.layers = _build_stack(
            HermiteKANLayer, in_features, output_dim, hidden_dims, num_layers, key,
            degree=degree,
        )


class LaguerreKAN(_StackedKAN):
    """Stacked LaguerreKAN."""

    def __init__(self, in_features: int, output_dim: int = 1,
                 hidden_dims=64, num_layers: int = 2, degree: int = 5,
                 *, key: jax.Array):
        self.in_features = in_features
        self.output_dim = output_dim
        self.layers = _build_stack(
            LaguerreKANLayer, in_features, output_dim, hidden_dims, num_layers, key,
            degree=degree,
        )


class BernsteinKAN(_StackedKAN):
    """Stacked BernsteinKAN."""

    def __init__(self, in_features: int, output_dim: int = 1,
                 hidden_dims=64, num_layers: int = 2, degree: int = 5,
                 *, key: jax.Array):
        self.in_features = in_features
        self.output_dim = output_dim
        self.layers = _build_stack(
            BernsteinKANLayer, in_features, output_dim, hidden_dims, num_layers, key,
            degree=degree,
        )


class ReLUKAN(_StackedKAN):
    """Stacked ReLU-KAN / FasterKAN."""

    def __init__(self, in_features: int, output_dim: int = 1,
                 hidden_dims=64, num_layers: int = 2,
                 grid_size: int = 8, order: int = 2,
                 grid_range: tuple[float, float] = (-1.0, 1.0),
                 *, key: jax.Array):
        self.in_features = in_features
        self.output_dim = output_dim
        self.layers = _build_stack(
            ReLUKANLayer, in_features, output_dim, hidden_dims, num_layers, key,
            grid_size=grid_size, order=order, grid_range=grid_range,
        )


class RationalKAN(_StackedKAN):
    """Stacked RationalKAN."""

    def __init__(self, in_features: int, output_dim: int = 1,
                 hidden_dims=64, num_layers: int = 2, degree: int = 5,
                 *, key: jax.Array):
        self.in_features = in_features
        self.output_dim = output_dim
        self.layers = _build_stack(
            RationalKANLayer, in_features, output_dim, hidden_dims, num_layers, key,
            degree=degree,
        )


class SincKAN(_StackedKAN):
    """Stacked SincKAN."""

    def __init__(self, in_features: int, output_dim: int = 1,
                 hidden_dims=64, num_layers: int = 2,
                 grid_size: int = 8,
                 grid_range: tuple[float, float] = (-2.0, 2.0),
                 *, key: jax.Array):
        self.in_features = in_features
        self.output_dim = output_dim
        self.layers = _build_stack(
            SincKANLayer, in_features, output_dim, hidden_dims, num_layers, key,
            grid_size=grid_size, grid_range=grid_range,
        )


class GramKAN(_StackedKAN):
    """Stacked GramKAN."""

    def __init__(self, in_features: int, output_dim: int = 1,
                 hidden_dims=64, num_layers: int = 2, degree: int = 5,
                 *, key: jax.Array):
        self.in_features = in_features
        self.output_dim = output_dim
        self.layers = _build_stack(
            GramKANLayer, in_features, output_dim, hidden_dims, num_layers, key,
            degree=degree,
        )


class BSRBFKAN(_StackedKAN):
    """Stacked BSRBFKAN (B-spline + RBF hybrid)."""

    def __init__(self, in_features: int, output_dim: int = 1,
                 hidden_dims=64, num_layers: int = 2,
                 grid_size: int = 5, spline_order: int = 3,
                 rbf_grid_size: int = 8,
                 grid_range: tuple[float, float] = (-1.0, 1.0),
                 rbf_grid_range: tuple[float, float] = (-2.0, 2.0),
                 *, key: jax.Array):
        self.in_features = in_features
        self.output_dim = output_dim
        self.layers = _build_stack(
            BSRBFKANLayer, in_features, output_dim, hidden_dims, num_layers, key,
            grid_size=grid_size, spline_order=spline_order,
            rbf_grid_size=rbf_grid_size,
            grid_range=grid_range, rbf_grid_range=rbf_grid_range,
        )


# =====================================================================
# KAN-Convolutional layer
#
# Per-pixel KAN applied across (kH * kW * C_in) -> C_out, then placed back
# into the output feature map. Channel-last (H, W, C) -> (H, W, C').
# =====================================================================


_BASIS_FACTORIES = {
    "bspline": lambda I, **kw: BSplineBasis(I, kw.get("grid_size", 5),
                                            kw.get("spline_order", 3),
                                            kw.get("grid_range", (-1.0, 1.0))),
    "rbf": lambda I, **kw: RBFBasis(kw.get("grid_size", 8),
                                    kw.get("grid_range", (-2.0, 2.0))),
    "fourier": lambda I, **kw: FourierBasis(kw.get("num_frequencies", 8)),
    "chebyshev": lambda I, **kw: ChebyshevBasis(kw.get("degree", 5)),
    "jacobi": lambda I, **kw: JacobiBasis(kw.get("degree", 5),
                                          kw.get("alpha", 1.0),
                                          kw.get("beta", 1.0)),
    "legendre": lambda I, **kw: LegendreBasis(kw.get("degree", 5)),
    "wavelet": lambda I, **kw: WaveletBasis(kw.get("num_scales", 6),
                                            kw.get("wavelet_type", "mexican_hat")),
    "taylor": lambda I, **kw: TaylorBasis(kw.get("degree", 4)),
    "hermite": lambda I, **kw: HermiteBasis(kw.get("degree", 5)),
    "laguerre": lambda I, **kw: LaguerreBasis(kw.get("degree", 5)),
    "bernstein": lambda I, **kw: BernsteinBasis(kw.get("degree", 5)),
    "relu": lambda I, **kw: ReLUKANBasis(kw.get("grid_size", 8),
                                         kw.get("order", 2),
                                         kw.get("grid_range", (-1.0, 1.0))),
    "rational": lambda I, **kw: RationalBasis(kw.get("degree", 5)),
    "sinc": lambda I, **kw: SincBasis(kw.get("grid_size", 8),
                                      kw.get("grid_range", (-2.0, 2.0))),
    "gram": lambda I, **kw: GramBasis(kw.get("degree", 5)),
    "bsrbf": lambda I, **kw: BSRBFBasis(I, kw.get("grid_size", 5),
                                        kw.get("spline_order", 3),
                                        kw.get("rbf_grid_size", 8),
                                        kw.get("grid_range", (-1.0, 1.0)),
                                        kw.get("rbf_grid_range", (-2.0, 2.0))),
}


class KANConv2d(eqx.Module):
    """KAN-convolution over 2-D feature maps (channel-last ``(H, W, C)``).

    Extracts ``(kH * kW)`` windows with ``SAME`` padding, flattens each window
    into a ``(kH * kW * in_channels)``-vector, then applies a single KAN layer
    to map it to ``out_channels``. Stride 1.

    The univariate basis is selected with ``basis`` and tuned with the
    corresponding kwargs (``grid_size``, ``degree``, etc.).

    Reference:
        Bodner, A. D., Tepsich, A. S., Spolski, J. N., & Pourteau, S. (2024).
        *Convolutional Kolmogorov-Arnold Networks*. arXiv:2406.13155.
        Code: github.com/AntonioTepsich/Convolutional-KANs.
    """

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    basis_name: str = eqx.field(static=True)
    kan: _BasisKANLayer

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        basis: str = "bspline",
        *,
        key: jax.Array,
        **basis_kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.basis_name = basis
        if basis not in _BASIS_FACTORIES:
            raise ValueError(
                f"Unknown basis '{basis}'. Available: {list(_BASIS_FACTORIES)}"
            )
        in_features = in_channels * kernel_size * kernel_size
        b = _BASIS_FACTORIES[basis](in_features, **basis_kwargs)
        self.kan = _BasisKANLayer(
            in_features, out_channels, b,
            skip=(basis in ("bspline",)),  # original-KAN style only for bspline
            skip_activation=jax.nn.silu,
            key=key,
        )

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        # x: (H, W, C). Extract (H, W, kH*kW*C) patches with SAME padding,
        # then feed each pixel's window through the KAN.
        was_unbatched = x.ndim == 3
        if was_unbatched:
            x = x[None]
        # NCHW for conv_general_dilated patches via reshape trick.
        # Simpler: use jax.lax.reduce_window-style patch extraction via dilated identity conv.
        N, H, W, C = x.shape
        k = self.kernel_size
        pad = k // 2
        # Build patches: (N, H, W, k, k, C) using padding + slicing.
        xp = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
        patches = jnp.stack(
            [xp[:, di : di + H, dj : dj + W, :] for di in range(k) for dj in range(k)],
            axis=-2,
        )  # (N, H, W, k*k, C)
        patches = patches.reshape(N, H, W, k * k * C)
        y = self.kan(patches)  # (N, H, W, out_channels)
        if was_unbatched:
            y = y[0]
        return y


def _build_kan_conv(basis: str, in_features: int, out_channels: int,
                   key: jax.Array, **basis_kwargs) -> "_BasisKANLayer":
    """Shared constructor for KAN-conv layers: select basis, build inner KAN."""
    if basis not in _BASIS_FACTORIES:
        raise ValueError(
            f"Unknown basis '{basis}'. Available: {list(_BASIS_FACTORIES)}"
        )
    b = _BASIS_FACTORIES[basis](in_features, **basis_kwargs)
    return _BasisKANLayer(
        in_features, out_channels, b,
        skip=(basis in ("bspline", "bsrbf", "relu")),
        skip_activation=jax.nn.silu,
        key=key,
    )


class KANConv1d(eqx.Module):
    """KAN-convolution over 1-D signals (channel-last ``(W, C)``).

    Extracts length-``kernel_size`` windows with ``SAME`` padding (stride 1),
    flattens each window to ``kernel_size * in_channels``, applies a KAN.

    Reference:
        1-D analogue of Bodner et al. 2024 (arXiv:2406.13155, *Convolutional
        Kolmogorov-Arnold Networks*).
    """

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    basis_name: str = eqx.field(static=True)
    kan: _BasisKANLayer

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 basis: str = "bspline", *, key: jax.Array, **basis_kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.basis_name = basis
        in_features = in_channels * kernel_size
        self.kan = _build_kan_conv(basis, in_features, out_channels, key, **basis_kwargs)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        # x: (W, C)
        W, C = x.shape
        k = self.kernel_size
        pad = k // 2
        xp = jnp.pad(x, ((pad, pad), (0, 0)))
        patches = jnp.stack(
            [xp[di : di + W, :] for di in range(k)], axis=-2
        )  # (W, k, C)
        return self.kan(patches.reshape(W, k * C))


class KANConv3d(eqx.Module):
    """KAN-convolution over 3-D volumes (channel-last ``(D, H, W, C)``).

    Reference:
        3-D analogue of Bodner et al. 2024 (arXiv:2406.13155, *Convolutional
        Kolmogorov-Arnold Networks*).
    """

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    basis_name: str = eqx.field(static=True)
    kan: _BasisKANLayer

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 basis: str = "bspline", *, key: jax.Array, **basis_kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.basis_name = basis
        in_features = in_channels * kernel_size ** 3
        self.kan = _build_kan_conv(basis, in_features, out_channels, key, **basis_kwargs)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        D, H, W, C = x.shape
        k = self.kernel_size
        pad = k // 2
        xp = jnp.pad(x, ((pad, pad), (pad, pad), (pad, pad), (0, 0)))
        patches = jnp.stack(
            [xp[dd : dd + D, di : di + H, dj : dj + W, :]
             for dd in range(k) for di in range(k) for dj in range(k)],
            axis=-2,
        )  # (D, H, W, k^3, C)
        return self.kan(patches.reshape(D, H, W, k ** 3 * C))


# =====================================================================
# KAN-based structural blocks
# =====================================================================


def _resolve_basis_layer(in_features: int, out_features: int,
                        basis: str, key: jax.Array, **basis_kwargs) -> _BasisKANLayer:
    """Build a ``_BasisKANLayer`` from a basis name + kwargs (no convolution)."""
    if basis not in _BASIS_FACTORIES:
        raise ValueError(
            f"Unknown basis '{basis}'. Available: {list(_BASIS_FACTORIES)}"
        )
    b = _BASIS_FACTORIES[basis](in_features, **basis_kwargs)
    return _BasisKANLayer(
        in_features, out_features, b,
        skip=(basis in ("bspline", "bsrbf", "relu")),
        skip_activation=jax.nn.silu,
        key=key,
    )


class KANResBlock(eqx.Module):
    """Residual KAN block: ``out = x + kan2(act(kan1(x)))``.

    ``in_features == out_features`` is required so the skip aligns.

    Reference:
        The residual-KAN pattern is used widely in deep-KAN follow-ups; see
        e.g. SS 2024 (arXiv:2405.07200, §4 for stacked ChebyKAN with skip
        connections) and the broader ResNet pattern from He et al. 2015
        (arXiv:1512.03385).
    """

    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    kan1: _BasisKANLayer
    kan2: _BasisKANLayer
    norm_layer: Optional[eqx.Module]

    def __init__(self, features: int, basis: str = "bspline",
                 activation: Callable = jax.nn.silu,
                 use_layer_norm: bool = False,
                 *, key: jax.Array, **basis_kwargs):
        self.in_features = features
        self.out_features = features
        self.activation = activation
        k1, k2 = jax.random.split(key)
        self.kan1 = _resolve_basis_layer(features, features, basis, k1, **basis_kwargs)
        self.kan2 = _resolve_basis_layer(features, features, basis, k2, **basis_kwargs)
        self.norm_layer = eqx.nn.LayerNorm(features) if use_layer_norm else None

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        h = self.kan1(x)
        if self.norm_layer is not None:
            # LayerNorm acts on the last axis vector — vmap over any leading dims.
            ndim = h.ndim
            ln = self.norm_layer
            for _ in range(ndim - 1):
                ln = jax.vmap(ln)
            h = ln(h)
        h = self.activation(h)
        h = self.kan2(h)
        return x + h


class KANSpectralBlock1d(eqx.Module):
    """FNO-style block with KAN channel mixing for 1-D fields ``(W, C) -> (W, C')``.

    Computes ``spectral_conv(x) + kan(x)``, then activation. Drop-in replacement
    for :class:`foundax.layers.SpectralBlock1d` with a KAN-based channel mixer
    instead of a plain linear skip.

    References:
        - Li, Z. et al. (2020). *Fourier Neural Operator for Parametric Partial
          Differential Equations*. arXiv:2010.08895 — original FNO block.
        - Liu, Z. et al. (2024). *KAN: Kolmogorov-Arnold Networks*. arXiv:2404.19756.
        The combination (spectral mixing + KAN pointwise) is a foundax design
        inspired by both works.
    """

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    n_modes: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    basis_name: str = eqx.field(static=True)
    spectral_conv: eqx.Module
    kan: _BasisKANLayer

    def __init__(self, in_channels: int, out_channels: int, n_modes: int,
                 basis: str = "rbf", activation: Callable = jax.nn.gelu,
                 *, key: jax.Array, **basis_kwargs):
        from .fno import SpectralConv1d

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.activation = activation
        self.basis_name = basis
        k1, k2 = jax.random.split(key)
        self.spectral_conv = SpectralConv1d(in_channels, out_channels, n_modes, True, key=k1)
        self.kan = _resolve_basis_layer(in_channels, out_channels, basis, k2, **basis_kwargs)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self.activation(self.spectral_conv(x) + self.kan(x))


class KANSpectralBlock2d(eqx.Module):
    """KAN-spectral block for 2-D fields ``(H, W, C) -> (H, W, C')``.

    References: Li et al. 2020 (FNO, arXiv:2010.08895) +
    Liu et al. 2024 (KAN, arXiv:2404.19756) — 2-D analogue of
    :class:`KANSpectralBlock1d`.
    """

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    n_modes: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    basis_name: str = eqx.field(static=True)
    spectral_conv: eqx.Module
    kan: _BasisKANLayer

    def __init__(self, in_channels: int, out_channels: int, n_modes: int,
                 basis: str = "rbf", activation: Callable = jax.nn.gelu,
                 *, key: jax.Array, **basis_kwargs):
        from .fno import SpectralConv2d

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.activation = activation
        self.basis_name = basis
        k1, k2 = jax.random.split(key)
        self.spectral_conv = SpectralConv2d(
            in_channels, out_channels, n_modes, n_modes, True, key=k1
        )
        self.kan = _resolve_basis_layer(in_channels, out_channels, basis, k2, **basis_kwargs)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self.activation(self.spectral_conv(x) + self.kan(x))


class KANSpectralBlock3d(eqx.Module):
    """KAN-spectral block for 3-D fields ``(D, H, W, C) -> (D, H, W, C')``.

    References: Li et al. 2020 (FNO, arXiv:2010.08895) +
    Liu et al. 2024 (KAN, arXiv:2404.19756) — 3-D analogue of
    :class:`KANSpectralBlock1d`.
    """

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    n_modes: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    basis_name: str = eqx.field(static=True)
    spectral_conv: eqx.Module
    kan: _BasisKANLayer

    def __init__(self, in_channels: int, out_channels: int, n_modes: int,
                 basis: str = "rbf", activation: Callable = jax.nn.gelu,
                 *, key: jax.Array, **basis_kwargs):
        from .fno import SpectralConv3d

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.activation = activation
        self.basis_name = basis
        k1, k2 = jax.random.split(key)
        self.spectral_conv = SpectralConv3d(
            in_channels, out_channels, n_modes, n_modes, n_modes, True, key=k1
        )
        self.kan = _resolve_basis_layer(in_channels, out_channels, basis, k2, **basis_kwargs)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self.activation(self.spectral_conv(x) + self.kan(x))


class KANAttentionBlock(eqx.Module):
    """Transformer-style block whose feed-forward MLP is a stacked KAN.

    Operates on sequences ``(N, D)`` (channel-last). The attention sub-layer is
    a standard pre-norm multi-head self-attention; the second sub-layer applies
    a KAN to each token. Outputs are residual-summed.

    Reference:
        Yang, X. & Wang, X. (2024). *Kolmogorov-Arnold Transformer*.
        arXiv:2409.10594. Code: github.com/Adamdad/kat.
        Underlying attention: Vaswani et al. 2017 (arXiv:1706.03762).
    """

    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    basis_name: str = eqx.field(static=True)
    attn: eqx.nn.MultiheadAttention
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    kan: _BasisKANLayer

    def __init__(self, features: int, num_heads: int = 4,
                 basis: str = "rbf", *, key: jax.Array, **basis_kwargs):
        self.in_features = features
        self.out_features = features
        self.num_heads = num_heads
        self.basis_name = basis
        k1, k2 = jax.random.split(key)
        self.attn = eqx.nn.MultiheadAttention(num_heads=num_heads, query_size=features, key=k1)
        self.norm1 = eqx.nn.LayerNorm(features)
        self.norm2 = eqx.nn.LayerNorm(features)
        self.kan = _resolve_basis_layer(features, features, basis, k2, **basis_kwargs)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        # x: (N, D). LayerNorm wants vectors — vmap over the token axis.
        ln1 = jax.vmap(self.norm1)
        ln2 = jax.vmap(self.norm2)
        h = ln1(x)
        x = x + self.attn(h, h, h)
        h = ln2(x)
        x = x + self.kan(h)
        return x
