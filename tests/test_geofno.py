import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import foundax as fx
from foundax.architectures.geofno import IPHI, SpectralConv2dGeo, fourier_factors

KEY = jax.random.PRNGKey(0)


def _points(n, seed=1):
    return jax.random.uniform(jax.random.PRNGKey(seed), (n, 2))


def _small(**kw):
    cfg = dict(modes1=4, modes2=4, width=8, s1=12, s2=12, fc_dim=16, iphi_width=8)
    cfg.update(kw)
    return fx.geofno(**cfg, key=KEY)


def test_shapes_default_query():
    m = _small()
    y = eqx.filter_jit(m)(_points(57))
    assert y.shape == (57, 1)
    assert jnp.all(jnp.isfinite(y))


def test_separate_query_and_features():
    m = _small(in_channels=3, out_channels=2)
    c, q = _points(40), _points(23, seed=2)
    feats = jax.random.normal(KEY, (40, 3))
    assert m(c, feats, out_coords=q).shape == (23, 2)


def test_query_equal_to_input_matches_default():
    m = _small()
    c = _points(30)
    np.testing.assert_allclose(m(c), m(c, out_coords=c + 0.0), rtol=1e-5, atol=1e-6)


def test_code_conditioning():
    m = _small(code_dim=5)
    c = _points(20)
    y0 = m(c, code=jnp.zeros(5))
    y1 = m(c, code=jnp.ones(5))
    assert y0.shape == (20, 1)
    assert not jnp.allclose(y0, y1)
    with pytest.raises(ValueError):
        m(c)


def test_no_deformation():
    m = _small(deform=False)
    assert m.iphi is None
    c = _points(16)
    np.testing.assert_array_equal(m.deform(c), c)
    assert m(c).shape == (16, 1)


def test_gradients_reach_deformation():
    m = _small()
    c = _points(25)
    g = eqx.filter_grad(lambda m: jnp.mean(m(c) ** 2))(m)
    assert float(jnp.linalg.norm(g.iphi.fc_out.weight)) > 0
    assert float(jnp.linalg.norm(g.conv_in.weights1_real)) > 0


def test_vmap_batch():
    m = _small()
    c = jnp.stack([_points(18, s) for s in range(3)])
    assert jax.vmap(m)(c).shape == (3, 18, 1)


def test_invalid_config():
    with pytest.raises(ValueError):
        fx.geofno(modes1=8, s1=12, key=KEY)
    with pytest.raises(ValueError):
        IPHI(width=10, key=KEY)
    with pytest.raises(ValueError):
        _small()(jnp.zeros((5, 3)))


def _grid_points(s1, s2):
    gx, gy = jnp.meshgrid(jnp.arange(s1) / s1, jnp.arange(s2) / s2, indexing="ij")
    return jnp.stack([gx, gy], -1).reshape(-1, 2)


def test_point_input_on_grid_nodes_equals_fft_path():
    """The non-uniform transform at the periodic grid nodes is the rfft2."""
    s1, s2, m1, m2, c = 12, 10, 3, 4, 2
    conv = SpectralConv2dGeo(c, 3, m1, m2, key=KEY)
    u = jax.random.normal(KEY, (s1, s2, c))
    f = fourier_factors(_grid_points(s1, s2), m1, m2)
    with jax.default_matmul_precision("highest"):  # TF32 on GPU is ~1e-4
        via_points = conv(u.reshape(-1, c), factors_in=f, grid_shape=(s1, s2))
        via_fft = conv(u)
    np.testing.assert_allclose(via_points, via_fft, rtol=1e-4, atol=1e-5)


def test_point_output_on_grid_nodes_equals_fft_path():
    """Evaluating the series at the grid nodes is s1*s2 times the irfft2.

    The negative-k1 Nyquist row (k1 = -modes1) is mirrored onto itself by
    the upstream conjugate extension, so it is zeroed for this identity.
    """
    s1, s2, m1, m2, c = 12, 10, 3, 4, 2
    conv = SpectralConv2dGeo(c, 3, m1, m2, key=KEY)
    conv = eqx.tree_at(
        lambda m: (m.weights2_real, m.weights2_imag),
        conv,
        (conv.weights2_real.at[:, :, 0].set(0), conv.weights2_imag.at[:, :, 0].set(0)),
    )
    u = jax.random.normal(KEY, (s1, s2, c))
    f = fourier_factors(_grid_points(s1, s2), m1, m2)
    with jax.default_matmul_precision("highest"):  # TF32 on GPU is ~1e-4
        via_points = conv(u, factors_out=f).reshape(s1, s2, -1) / (s1 * s2)
        via_fft = conv(u)
    np.testing.assert_allclose(via_points, via_fft, rtol=1e-4, atol=1e-5)


def test_nodal_variant_still_available():
    m = fx.geofno_nodal(2, [4, 4], [1.0, 1.0], layers=(8, 8), fc_dim=8)
    n = 10
    out = m(
        jax.random.normal(KEY, (n, 3)),
        jnp.ones((n, 1)),
        _points(n),
        jnp.full((n, 1), 1.0 / n),
    )
    assert out.shape == (n,)
