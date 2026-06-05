import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pytest

import foundax as fx
from foundax.architectures.sfno import (
    RealSHT2d,
    SphericalConv2d,
    SphericalBlock2d,
    SFNO2d,
)


KEY = jax.random.PRNGKey(0)


def _bandlimited_coeffs(L, channels, rng):
    """Random spectral coefficients respecting the m ≤ l mask and real
    constraint on m=0."""
    c = (rng.standard_normal((L, L, channels)) + 1j * rng.standard_normal((L, L, channels))) / 10
    for l in range(L):
        for m in range(L):
            if m > l:
                c[l, m, :] = 0
    c[:, 0, :] = c[:, 0, :].real
    return jnp.asarray(c)


@pytest.mark.parametrize("grid", ["legendre-gauss", "equiangular"])
def test_sht_inverse_then_forward_is_identity_on_bandlimited(grid):
    """``forward ∘ inverse`` on bandlimited spectral data is identity."""
    L, nlat, nlon = 16, 32, 64
    sht = RealSHT2d(L=L, nlat=nlat, nlon=nlon, grid=grid)
    rng = np.random.default_rng(0)
    coef = _bandlimited_coeffs(L, channels=1, rng=rng)

    grid_field = sht.inverse(coef)
    coef_back = sht.forward(grid_field)

    mask = (np.arange(L)[None, :] <= np.arange(L)[:, None]).astype(np.float32)
    # Legendre-Gauss is exact for polynomials up to 2L-1 ⇒ tight tolerance.
    # Equiangular uses approximate weights ⇒ loose tolerance.
    tol = 1e-5 if grid == "legendre-gauss" else 5e-2
    assert jnp.max(jnp.abs(coef - coef_back * mask[..., None])) < tol


def test_sht_output_is_real_for_real_spectral():
    """Inverse SHT of a properly-conjugate-symmetric spectrum is real."""
    L, nlat, nlon = 8, 16, 32
    sht = RealSHT2d(L=L, nlat=nlat, nlon=nlon)
    rng = np.random.default_rng(1)
    coef = _bandlimited_coeffs(L, channels=1, rng=rng)
    field = sht.inverse(coef)
    assert field.dtype.kind == "f"
    assert jnp.all(jnp.isfinite(field))


def test_l_must_fit_grid():
    with pytest.raises(ValueError, match="≤ nlat"):
        RealSHT2d(L=33, nlat=32, nlon=64)
    with pytest.raises(ValueError, match="nlon"):
        RealSHT2d(L=20, nlat=64, nlon=16)


def test_factory_shape():
    model = fx.sfno2d(
        in_channels=3,
        out_channels=2,
        hidden_channels=16,
        L=8,
        nlat=16,
        nlon=32,
        n_layers=2,
    )
    x = jax.random.normal(KEY, (16, 32, 3))
    out = model(x)
    assert out.shape == (16, 32, 2)
    assert jnp.all(jnp.isfinite(out))


def test_default_out_channels_matches_in_channels():
    model = fx.sfno2d(in_channels=4, hidden_channels=8, L=4, nlat=8, nlon=16, n_layers=1)
    x = jax.random.normal(KEY, (8, 16, 4))
    out = model(x)
    assert out.shape == (8, 16, 4)


def test_gradient_flows_through_spectral_weights():
    model = fx.sfno2d(
        in_channels=2,
        hidden_channels=8,
        out_channels=1,
        L=4,
        nlat=8,
        nlon=16,
        n_layers=2,
    )
    x = jax.random.normal(KEY, (4, 8, 16, 2))
    y_t = jax.random.normal(KEY, (4, 8, 16, 1))

    def loss(m, x, y_t):
        return jnp.mean((jax.vmap(m)(x) - y_t) ** 2)

    grads = eqx.filter_grad(loss)(model, x, y_t)
    assert jnp.all(jnp.isfinite(grads.blocks[0].spectral.weight_real))
    assert jnp.all(jnp.isfinite(grads.blocks[0].spectral.weight_imag))
    assert jnp.all(jnp.isfinite(grads.blocks[0].linear.weight))


def test_equiangular_grid_path():
    model = fx.sfno2d(
        in_channels=1,
        hidden_channels=8,
        out_channels=1,
        L=4,
        nlat=8,
        nlon=16,
        n_layers=1,
        grid="equiangular",
    )
    x = jax.random.normal(KEY, (8, 16, 1))
    out = model(x)
    assert out.shape == (8, 16, 1)
    assert jnp.all(jnp.isfinite(out))


def test_unknown_grid_raises():
    with pytest.raises(ValueError, match="grid"):
        RealSHT2d(L=4, nlat=8, nlon=16, grid="custom-grid")


def test_sht_geometry_buffers_are_frozen_under_grad():
    """The Legendre table and quadrature weights encode the sphere
    geometry and must NOT receive gradients — otherwise training would
    silently corrupt the SHT and destroy SFNO's rotational equivariance.
    """
    model = fx.sfno2d(
        in_channels=2,
        hidden_channels=8,
        out_channels=1,
        L=4,
        nlat=8,
        nlon=16,
        n_layers=2,
    )
    x = jax.random.normal(KEY, (8, 16, 2))
    y_t = jax.random.normal(KEY, (8, 16, 1))

    def loss(m, x, y_t):
        return jnp.mean((m(x) - y_t) ** 2)

    grads = eqx.filter_grad(loss)(model, x, y_t)
    sht = grads.blocks[0].spectral.sht
    assert jnp.all(sht.legendre == 0.0)
    assert jnp.all(sht.weights == 0.0)


def test_longitude_shift_equivariance():
    """SFNO must commute with cyclic shifts along the longitude axis —
    this is the rotational property that motivates spherical harmonics
    over FFT in the first place.
    """
    model = fx.sfno2d(
        in_channels=2,
        hidden_channels=8,
        out_channels=2,
        L=6,
        nlat=12,
        nlon=24,
        n_layers=2,
        grid="legendre-gauss",
    )
    x = jax.random.normal(KEY, (12, 24, 2))
    shift = 7
    y = model(x)
    y_shifted = model(jnp.roll(x, shift, axis=1))
    assert jnp.allclose(y_shifted, jnp.roll(y, shift, axis=1), atol=1e-4)


def test_jit_compatibility():
    """Forward, inverse, and a full gradient step must trace under
    ``eqx.filter_jit``. The Legendre buffer must stay frozen under JIT."""
    sht = RealSHT2d(L=8, nlat=16, nlon=32)
    x = jax.random.normal(KEY, (16, 32, 2))

    @eqx.filter_jit
    def fwd(s, x):
        return s.forward(x)

    @eqx.filter_jit
    def inv(s, c):
        return s.inverse(c)

    f = fwd(sht, x)
    assert f.shape == (8, 8, 2)
    assert inv(sht, f).shape == (16, 32, 2)

    model = fx.sfno2d(
        in_channels=2, hidden_channels=8, out_channels=1,
        L=4, nlat=8, nlon=16, n_layers=2,
    )
    y_t = jax.random.normal(KEY, (8, 16, 1))

    @eqx.filter_jit
    def step(m, x, y_t):
        return eqx.filter_grad(lambda m: jnp.mean((m(x) - y_t) ** 2))(m)

    g = step(model, jax.random.normal(KEY, (8, 16, 2)), y_t)
    # JIT must honour stop_gradient on the geometry buffers.
    assert jnp.all(g.blocks[0].spectral.sht.legendre == 0.0)
    assert jnp.all(g.blocks[0].spectral.sht.weights == 0.0)


def test_pipe_integration_with_spherical_block():
    """SphericalBlock2d should be shape-preserving and pipe-compatible."""
    block = SphericalBlock2d(
        in_channels=4,
        out_channels=4,
        L=4,
        nlat=8,
        nlon=16,
        key=KEY,
    )
    x = jax.random.normal(KEY, (8, 16, 4))
    assert block(x).shape == x.shape
