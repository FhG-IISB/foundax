"""Edge-case tests for KAN bases, layers, and networks."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import foundax as fx
from foundax.architectures.kan import (
    FourierBasis,
    ChebyshevBasis,
    TaylorBasis,
    BernsteinBasis,
    BSRBFBasis,
    KANLayer,
    FastKANLayer,
    ChebyshevKANLayer,
    BernsteinKANLayer,
    TaylorKANLayer,
    ReLUKANLayer,
    FourierKANLayer,
    KANConv1d,
    KANConv2d,
)


def _k(seed=0):
    return jax.random.PRNGKey(seed)


# ── Single-feature layers (in_features=1) ─────────────────────────────────


@pytest.mark.parametrize(
    "cls,kw",
    [
        (KANLayer, dict(grid_size=4, spline_order=2)),
        (FastKANLayer, dict(grid_size=4)),
        (ChebyshevKANLayer, dict(degree=3)),
        (TaylorKANLayer, dict(degree=3)),
        (FourierKANLayer, dict(num_frequencies=3)),
        (BernsteinKANLayer, dict(degree=3)),
    ],
)
def test_layer_in_features_one(cls, kw):
    layer = cls(1, 4, key=_k(), **kw)
    y = layer(jnp.linspace(-1, 1, 10).reshape(-1, 1))
    assert y.shape == (10, 4)


# ── Single-output layers (out_features=1) ─────────────────────────────────


def test_layer_out_features_one():
    layer = FastKANLayer(8, 1, grid_size=4, key=_k())
    y = layer(jnp.ones((6, 8)))
    assert y.shape == (6, 1)


# ── Tiny degree / smallest legal hyperparams ──────────────────────────────


def test_chebyshev_degree_zero_is_constant():
    """degree=0 → only the constant basis function T_0(x)=1."""
    basis = ChebyshevBasis(degree=0)
    x = jnp.linspace(-1, 1, 10).reshape(-1, 1)
    phi = basis(x)
    assert phi.shape == (10, 1, 1)
    assert jnp.allclose(phi[..., 0], 1.0)


def test_taylor_degree_zero_is_constant():
    basis = TaylorBasis(degree=0)
    x = jnp.linspace(-1, 1, 8).reshape(-1, 1)
    phi = basis(x)
    assert phi.shape == (8, 1, 1)
    assert jnp.allclose(phi[..., 0], 1.0)


def test_fourier_one_frequency():
    basis = FourierBasis(num_frequencies=1)
    x = jnp.linspace(-1, 1, 6).reshape(-1, 1)
    phi = basis(x)
    assert phi.shape == (6, 1, 2)  # 1 cos + 1 sin


def test_bernstein_degree_zero_partition_of_unity():
    basis = BernsteinBasis(degree=0)
    x = jnp.linspace(-5, 5, 10).reshape(-1, 1)
    phi = basis(x)
    assert jnp.allclose(phi[..., 0], 1.0, atol=1e-5)


# ── Stress: large degree / large grid still finite ────────────────────────


@pytest.mark.parametrize("degree", [10, 20])
def test_chebyshev_large_degree_finite(degree):
    layer = ChebyshevKANLayer(4, 4, degree=degree, key=_k())
    y = layer(jax.random.normal(_k(1), (5, 4)))
    assert jnp.all(jnp.isfinite(y))


def test_fastkan_large_grid_finite():
    layer = FastKANLayer(4, 4, grid_size=64, key=_k())
    y = layer(jax.random.normal(_k(1), (5, 4)) * 3)
    assert jnp.all(jnp.isfinite(y))


# ── Forward at extreme input magnitudes ───────────────────────────────────


@pytest.mark.parametrize("magnitude", [1e-6, 1e-1, 1.0, 5.0, 50.0])
def test_squashed_layers_finite_at_magnitude(magnitude):
    """Polynomial-basis layers all squash through tanh — must stay finite."""
    for cls, kw in [
        (ChebyshevKANLayer, dict(degree=5)),
        (TaylorKANLayer, dict(degree=5)),
        (BernsteinKANLayer, dict(degree=5)),
    ]:
        layer = cls(3, 4, key=_k(), **kw)
        x = jnp.full((3, 3), magnitude, dtype=jnp.float32)
        y = layer(x)
        assert jnp.all(jnp.isfinite(y)), (
            f"{cls.__name__} produced non-finite at magnitude={magnitude}"
        )


# ── ReLU-KAN compact-support corner: out-of-grid all-zero behaviour ───────


def test_relu_kan_out_of_grid_is_zero():
    layer = ReLUKANLayer(2, 4, grid_size=8, order=2, grid_range=(-1.0, 1.0), key=_k())
    # x = 100 is far outside every grid interval, so basis = 0 everywhere.
    # The layer still has a SiLU skip (Linear), so output need not be exactly
    # zero — but it equals exactly `skip(silu(x))`, not the spline term.
    x = jnp.full((1, 2), 100.0)
    # Force the skip to zero to verify the spline contribution alone.
    import equinox as eqx

    layer_no_skip = eqx.tree_at(
        lambda m: (m.skip.weight, m.skip.bias),
        layer,
        (jnp.zeros_like(layer.skip.weight), jnp.zeros_like(layer.skip.bias)),
    )
    y = layer_no_skip(x)
    assert jnp.allclose(y, 0.0, atol=1e-6)


# ── Conv variants with different kernel sizes ─────────────────────────────


@pytest.mark.parametrize("k", [1, 3, 5])
def test_kan_conv2d_kernel_sizes(k):
    conv = KANConv2d(
        in_channels=2, out_channels=3, kernel_size=k, basis="rbf", key=_k()
    )
    y = conv(jnp.ones((8, 8, 2)))
    assert y.shape == (8, 8, 3)


@pytest.mark.parametrize("k", [1, 3, 5])
def test_kan_conv1d_kernel_sizes(k):
    conv = KANConv1d(
        in_channels=2, out_channels=3, kernel_size=k, basis="rbf", key=_k()
    )
    y = conv(jnp.ones((16, 2)))
    assert y.shape == (16, 3)


# ── Single-layer networks (num_layers=1) ──────────────────────────────────


def test_kan_network_single_layer():
    """num_layers=1 → just (in_features → output_dim) with no hidden layers."""
    model = fx.kan.fast(
        in_features=3, output_dim=2, hidden_dims=4, num_layers=1, key=_k()
    )
    y = model(jnp.ones((5, 3)))
    assert y.shape == (5, 2)


# ── BSRBFBasis: size invariant under hyperparam changes ───────────────────


@pytest.mark.parametrize("gs,k,rgs", [(3, 2, 4), (5, 3, 8), (8, 4, 16)])
def test_bsrbf_size_invariant(gs, k, rgs):
    b = BSRBFBasis(in_features=1, grid_size=gs, spline_order=k, rbf_grid_size=rgs)
    assert b.size == (gs + k) + rgs


# ── Empty input edge: single-sample (in_features,) ────────────────────────


def test_layer_unbatched_single_sample():
    layer = FastKANLayer(4, 2, key=_k())
    y = layer(jnp.array([0.1, -0.2, 0.3, 0.0]))
    assert y.shape == (2,)
