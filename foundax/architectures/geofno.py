"""Geo-FNO: geometry-aware Fourier neural operator (Li et al. 2022).

Reference: Z. Li, D. Z. Huang, B. Liu, A. Anandkumar, "Fourier Neural
Operator with Learned Deformations for PDEs on General Geometries",
arXiv:2207.05209. Ported from the authors' code (neuraloperator/Geo-FNO,
``elasticity/elas_geofno_v2.py``, MIT license).

A learned deformation ``iphi`` maps the physical points ``x`` to latent
coordinates ``xi = iphi(x)`` in the unit square. The first spectral layer
projects the point values onto Fourier modes at ``xi`` (a non-uniform
Fourier transform), the middle layers are ordinary FFT-based Fourier layers
on a uniform ``s1 x s2`` latent grid, and the last spectral layer evaluates
the truncated Fourier series back at the (deformed) query points.

Single example, channel-last: point arrays are ``(N, C)``.
"""

from functools import partial
from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from .linear import Linear
from .common import get_activation


def _activation(name: str) -> Callable:
    # Upstream uses torch's exact (erf) GELU; jax.nn.gelu defaults to tanh.
    if name.lower() == "gelu":
        return partial(jax.nn.gelu, approximate=False)
    act = get_activation(name)
    return act if act is not None else (lambda x: x)


def _wavenumbers(modes1: int, modes2: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Mode indices ``[0..m1-1, -m1..-1]`` and ``[0..m2-1, -(m2-1)..-1]``."""
    k1 = jnp.concatenate([jnp.arange(modes1), jnp.arange(-modes1, 0)])
    k2 = jnp.concatenate([jnp.arange(modes2), jnp.arange(-(modes2 - 1), 0)])
    return k1, k2


def fourier_factors(
    xi: jnp.ndarray, modes1: int, modes2: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Separable forward Fourier factors at latent points.

    ``exp(-2 pi i (k1 x + k2 y)) = e1[:, k1] * e2[:, k2]``; the inverse
    transform uses the complex conjugates.

    Args:
        xi: float[N, 2] - Latent coordinates (nominally in ``[0, 1]^2``).

    Returns:
        e1: complex[N, 2*modes1], e2: complex[N, 2*modes2 - 1]
    """
    k1, k2 = _wavenumbers(modes1, modes2)
    e1 = jnp.exp(-2j * jnp.pi * xi[:, 0:1] * k1[None, :].astype(xi.dtype))
    e2 = jnp.exp(-2j * jnp.pi * xi[:, 1:2] * k2[None, :].astype(xi.dtype))
    return e1, e2


class SpectralConv2dGeo(eqx.Module):
    """Fourier layer on the latent grid, with optional point-cloud input/output.

    Weights follow upstream: complex ``(in, out, modes1, modes2)`` for the
    positive and the negative ``k1`` block, stored as real/imag pairs.
    """

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    modes1: int = eqx.field(static=True)
    modes2: int = eqx.field(static=True)
    weights1_real: jnp.ndarray
    weights1_imag: jnp.ndarray
    weights2_real: jnp.ndarray
    weights2_imag: jnp.ndarray

    def __init__(
        self, in_channels: int, out_channels: int, modes1: int, modes2: int, *, key
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        # Upstream: scale * torch.rand(..., dtype=cfloat) -> U[0, scale) parts.
        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes1, modes2)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.weights1_real = jax.random.uniform(k1, shape, maxval=scale)
        self.weights1_imag = jax.random.uniform(k2, shape, maxval=scale)
        self.weights2_real = jax.random.uniform(k3, shape, maxval=scale)
        self.weights2_imag = jax.random.uniform(k4, shape, maxval=scale)

    def _mix(self, u_ft: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """u_ft: complex[K1, K2, in] -> two blocks complex[modes1, modes2, out]."""
        m1, m2 = self.modes1, self.modes2
        w1 = self.weights1_real + 1j * self.weights1_imag
        w2 = self.weights2_real + 1j * self.weights2_imag
        f1 = jnp.einsum("xyi,ioxy->xyo", u_ft[:m1, :m2], w1)
        f2 = jnp.einsum("xyi,ioxy->xyo", u_ft[-m1:, :m2], w2)
        return f1, f2

    def __call__(
        self,
        u: jnp.ndarray,
        *,
        factors_in: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        factors_out: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
    ) -> jnp.ndarray:
        """
        Args:
            u: float[s1, s2, in] on the latent grid, or float[N_in, in] at
                points when ``factors_in`` is given.
            factors_in: ``fourier_factors`` of the input points.
            factors_out: ``fourier_factors`` of the output points; ``None``
                returns values on the latent grid.
            grid_shape: Latent grid ``(s1, s2)``; required when the input is
                a point cloud and the output a grid.

        Returns:
            float[s1, s2, out] on the grid, or float[N_out, out] at points.
        """
        if factors_in is None:
            s1, s2 = u.shape[0], u.shape[1]
            u_ft = jnp.fft.rfft2(u, axes=(0, 1))
        else:
            s1, s2 = grid_shape
            e1, e2 = factors_in
            u_ft = jnp.einsum("nc,na,nb->abc", u.astype(e1.dtype), e1, e2)

        f1, f2 = self._mix(u_ft)

        if factors_out is None:
            m1, m2 = self.modes1, self.modes2
            out_ft = jnp.zeros((s1, s2 // 2 + 1, self.out_channels), dtype=f1.dtype)
            out_ft = out_ft.at[:m1, :m2].set(f1)
            out_ft = out_ft.at[-m1:, :m2].set(f2)
            return jnp.fft.irfft2(out_ft, s=(s1, s2), axes=(0, 1))

        # Extend the half spectrum to negative k2 by conjugate symmetry:
        # c(-k1, -k2) = conj(c(k1, k2)).
        out_ft = jnp.concatenate([f1, f2], axis=0)  # [2*m1, m2, out]
        n1 = out_ft.shape[0]
        neg_k1 = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), jnp.arange(n1 - 1, 0, -1)]
        )
        mirror = jnp.conj(jnp.flip(out_ft[neg_k1, 1:], axis=1))
        full = jnp.concatenate([out_ft, mirror], axis=1)  # [2*m1, 2*m2-1, out]
        e1, e2 = factors_out
        return jnp.einsum("abo,na,nb->no", full, jnp.conj(e1), jnp.conj(e2)).real


class IPHI(eqx.Module):
    """Learned deformation ``x -> xi = x + x * f(x)`` (inverse of phi).

    Features ``(x, y, angle, radius)`` about ``center``, plus NeRF-style
    sin/cos encodings at frequencies ``pi * 2^j``, ``j < width // 4``; an
    optional design ``code`` vector is embedded and concatenated.

    ``depth=3, act="tanh", center=(1e-4, 1e-4)`` is the network of the
    paper's elasticity example (``elas_geofno.py``); ``depth=2, act="gelu",
    center=(0.5, 0.5)`` is ``elas_geofno_v2.py``.
    """

    width: int = eqx.field(static=True)
    code_dim: int = eqx.field(static=True)
    center: Tuple[float, float] = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)

    fc0: Linear
    fc_code: Optional[Linear]
    fc_no_code: Optional[Linear]
    hidden: List[Linear]
    fc_out: Linear

    def __init__(
        self,
        width: int = 32,
        depth: int = 3,
        code_dim: int = 0,
        act: str = "tanh",
        center: Tuple[float, float] = (1e-4, 1e-4),
        *,
        key,
    ):
        if width % 4:
            raise ValueError(f"IPHI width must be divisible by 4, got {width}")
        self.width = width
        self.code_dim = code_dim
        self.center = tuple(float(c) for c in center)
        self.activation = _activation(act)

        keys = jax.random.split(key, depth + 3)
        self.fc0 = Linear(4, width, key=keys[0])
        if code_dim > 0:
            self.fc_code = Linear(code_dim, width, key=keys[1])
            self.fc_no_code = None
        else:
            self.fc_code = None
            self.fc_no_code = Linear(3 * width, 4 * width, key=keys[1])
        self.hidden = [Linear(4 * width, 4 * width, key=k) for k in keys[2:-1]]
        self.fc_out = Linear(4 * width, 2, key=keys[-1])

    def __call__(
        self, x: jnp.ndarray, code: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Args:
            x: float[N, 2] - Physical coordinates.
            code: float[code_dim] - Design parameters (required iff code_dim > 0).

        Returns:
            float[N, 2] - Latent coordinates.
        """
        if (code is None) != (self.code_dim == 0):
            raise ValueError(
                f"IPHI built with code_dim={self.code_dim}, got code={code is not None}"
            )
        c = jnp.asarray(self.center, dtype=x.dtype)
        d = x - c
        angle = jnp.arctan2(d[:, 1], d[:, 0])
        radius = jnp.linalg.norm(d, axis=-1)
        xd = jnp.stack([x[:, 0], x[:, 1], angle, radius], axis=-1)  # [N, 4]

        freqs = jnp.pi * 2.0 ** jnp.arange(self.width // 4, dtype=x.dtype)
        arg = (xd[:, :, None] * freqs).reshape(x.shape[0], -1)  # [N, width]
        xd = jnp.concatenate([self.fc0(xd), jnp.sin(arg), jnp.cos(arg)], axis=-1)

        if self.fc_code is not None:
            cd = jnp.broadcast_to(self.fc_code(code), (x.shape[0], self.width))
            xd = jnp.concatenate([cd, xd], axis=-1)
        else:
            xd = self.fc_no_code(xd)

        for layer in self.hidden:
            xd = self.activation(layer(xd))
        xd = self.fc_out(xd)
        return x + x * xd


class GeoFNO(eqx.Module):
    """Geo-FNO on a 2-D point cloud.

    1. Lift point features to ``width`` channels (``fc0``).
    2. ``conv_in``: point cloud -> latent ``s1 x s2`` grid via the
       non-uniform Fourier transform at ``xi = iphi(x)``.
    3. ``n_layers`` latent Fourier layers ``u' = act(K u + W u + B(grid))``.
    4. ``conv_out``: latent grid -> query points, plus ``B(x_out)``.
    5. Project with ``fc1`` / ``fc2``.

    The bias terms ``B`` are pointwise linear maps of the coordinates
    (upstream ``b0..b4``).
    """

    modes1: int = eqx.field(static=True)
    modes2: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    s1: int = eqx.field(static=True)
    s2: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)

    fc0: Linear
    conv_in: SpectralConv2dGeo
    convs: List[SpectralConv2dGeo]
    ws: List[Linear]
    bs: List[Linear]
    conv_out: SpectralConv2dGeo
    b_out: Linear
    fc1: Linear
    fc2: Linear
    iphi: Optional[IPHI]

    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        in_channels: int = 2,
        out_channels: int = 1,
        s1: int = 40,
        s2: int = 40,
        n_layers: int = 3,
        fc_dim: int = 128,
        act: str = "gelu",
        deform: bool = True,
        iphi_width: int = 32,
        iphi_depth: int = 3,
        iphi_act: str = "tanh",
        iphi_center: Tuple[float, float] = (1e-4, 1e-4),
        code_dim: int = 0,
        *,
        key,
    ):
        if 2 * modes1 > s1 or modes2 > s2 // 2 + 1:
            raise ValueError(
                f"modes ({modes1}, {modes2}) do not fit the latent grid ({s1}, {s2})"
            )
        self.modes1, self.modes2, self.width = modes1, modes2, width
        self.s1, self.s2 = s1, s2
        self.in_channels, self.out_channels = in_channels, out_channels
        self.activation = _activation(act)

        keys = iter(jax.random.split(key, 3 * n_layers + 8))
        spec = partial(SpectralConv2dGeo, width, width, modes1, modes2)
        self.fc0 = Linear(in_channels, width, key=next(keys))
        self.conv_in = spec(key=next(keys))
        self.convs = [spec(key=next(keys)) for _ in range(n_layers)]
        self.ws = [Linear(width, width, key=next(keys)) for _ in range(n_layers)]
        self.bs = [Linear(2, width, key=next(keys)) for _ in range(n_layers + 1)]
        self.conv_out = spec(key=next(keys))
        self.b_out = Linear(2, width, key=next(keys))
        self.fc1 = Linear(width, fc_dim, key=next(keys))
        self.fc2 = Linear(fc_dim, out_channels, key=next(keys))
        self.iphi = (
            IPHI(
                iphi_width,
                iphi_depth,
                code_dim,
                iphi_act,
                iphi_center,
                key=next(keys),
            )
            if deform
            else None
        )

    def latent_grid(self, dtype=jnp.float32) -> jnp.ndarray:
        """Uniform latent grid, float[s1, s2, 2] in ``[0, 1]^2``."""
        gx = jnp.linspace(0.0, 1.0, self.s1, dtype=dtype)
        gy = jnp.linspace(0.0, 1.0, self.s2, dtype=dtype)
        return jnp.stack(jnp.meshgrid(gx, gy, indexing="ij"), axis=-1)

    def deform(
        self, coords: jnp.ndarray, code: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Latent coordinates ``iphi(coords)`` (identity without deformation)."""
        return coords if self.iphi is None else self.iphi(coords, code)

    def __call__(
        self,
        coords: jnp.ndarray,
        x: Optional[jnp.ndarray] = None,
        *,
        out_coords: Optional[jnp.ndarray] = None,
        code: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Args:
            coords: float[N, 2] - Input points (physical domain).
            x: float[N, in_channels] - Input features; ``None`` uses
                ``coords`` (the elasticity setup, ``in_channels=2``).
            out_coords: float[M, 2] - Query points; ``None`` uses ``coords``.
            code: float[code_dim] - Design parameters for ``iphi``.

        Returns:
            float[M, out_channels]
        """
        if coords.shape[-1] != 2:
            raise ValueError(f"coords must be (N, 2), got {coords.shape}")
        if x is None:
            x = coords
        same_out = out_coords is None
        if same_out:
            out_coords = coords

        f_in = fourier_factors(self.deform(coords, code), self.modes1, self.modes2)
        f_out = (
            f_in
            if same_out
            else fourier_factors(
                self.deform(out_coords, code), self.modes1, self.modes2
            )
        )
        grid = self.latent_grid(coords.dtype)

        u = self.fc0(x)
        u = self.conv_in(u, factors_in=f_in, grid_shape=(self.s1, self.s2))
        u = self.activation(u + self.bs[0](grid))
        for conv, w, b in zip(self.convs, self.ws, self.bs[1:]):
            u = self.activation(conv(u) + w(u) + b(grid))

        u = self.conv_out(u, factors_out=f_out) + self.b_out(out_coords)
        u = self.activation(self.fc1(u))
        return self.fc2(u)
