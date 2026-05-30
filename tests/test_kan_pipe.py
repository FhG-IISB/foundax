"""Cross-cutting pipe / combinator tests covering all KAN variants."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")

import foundax as fx
from foundax.pipe import Block, Pipe, ShapeMismatchError
from foundax.architectures.kan import (
    KANLayer, EfficientKANLayer, FastKANLayer, FourierKANLayer,
    ChebyshevKANLayer, JacobiKANLayer, LegendreKANLayer, WaveletKANLayer,
    TaylorKANLayer,
    HermiteKANLayer, LaguerreKANLayer, BernsteinKANLayer, ReLUKANLayer,
    RationalKANLayer, SincKANLayer, GramKANLayer, BSRBFKANLayer,
)


def _ks(n, seed=0):
    return jax.random.split(jax.random.PRNGKey(seed), n)


ALL_LAYERS = [
    KANLayer, EfficientKANLayer, FastKANLayer, FourierKANLayer,
    ChebyshevKANLayer, JacobiKANLayer, LegendreKANLayer, WaveletKANLayer,
    TaylorKANLayer,
    HermiteKANLayer, LaguerreKANLayer, BernsteinKANLayer, ReLUKANLayer,
    RationalKANLayer, SincKANLayer, GramKANLayer, BSRBFKANLayer,
]


@pytest.mark.parametrize("cls", ALL_LAYERS)
def test_layer_block_sniff(cls):
    layer = cls(4, 8, key=_ks(1)[0])
    b = fx.block(layer)
    assert b._in_channels == 4
    assert b._out_channels == 8


@pytest.mark.parametrize("cls", ALL_LAYERS)
def test_layer_pipe_mismatch_raises(cls):
    a = fx.block(cls(4, 8, key=_ks(2)[0]))
    bad = fx.block(cls(7, 1, key=_ks(2)[1]))  # in=7 ≠ out=8
    with pytest.raises(ShapeMismatchError):
        _ = a | bad


def test_mixed_basis_pipeline():
    """FastKAN -> Chebyshev -> Fourier -> linear out."""
    k = _ks(4)
    p = (
        fx.block(FastKANLayer(2, 32, key=k[0]))
        | fx.block(ChebyshevKANLayer(32, 32, degree=4, key=k[1]))
        | fx.block(FourierKANLayer(32, 16, num_frequencies=4, key=k[2]))
        | fx.block(fx.layers.Linear(16, 1, key=k[3]))
    )
    y = p(jnp.ones((10, 2)))
    assert y.shape == (10, 1)


def test_mixed_basis_pipeline_jit():
    k = _ks(3)
    p = (
        fx.block(FastKANLayer(2, 16, key=k[0]))
        | fx.block(JacobiKANLayer(16, 8, degree=3, key=k[1]))
        | fx.block(TaylorKANLayer(8, 1, degree=4, key=k[2]))
    )

    @eqx.filter_jit
    def fwd(m, x):
        return m(x)

    y = fwd(p, jnp.ones((5, 2)))
    assert y.shape == (5, 1)


def test_dot_combinator_kan_branches():
    """fx.dot of a KAN branch and a KAN trunk."""
    k = _ks(2)
    branch = fx.block(fx.fastkan(in_features=3, output_dim=16, key=k[0]))
    trunk = fx.block(fx.chebyshev_kan(in_features=2, output_dim=16, key=k[1]))
    op = fx.dot(branch, trunk)
    u = jnp.ones((3,))
    y = jnp.ones((10, 2))
    out = op(u, y)
    assert out.shape == (10,)


def test_add_combinator_kan_branches():
    k = _ks(2)
    a = fx.block(fx.layers.FastKANLayer(2, 4, key=k[0]))
    b = fx.block(fx.layers.ChebyshevKANLayer(2, 4, degree=3, key=k[1]))
    op = fx.add(a, b)
    out = op(jnp.ones((5, 2)))
    assert out.shape == (5, 4)


def test_kan_mlp_hybrid():
    k = _ks(2)
    pipe = (
        fx.block(fx.fastkan(in_features=2, output_dim=16, hidden_dims=16,
                            num_layers=2, key=k[0]))
        | fx.block(fx.mlp(in_features=16, output_dim=1, hidden_dims=16, key=k[1]))
    )
    y = pipe(jnp.ones((4, 2)))
    assert y.shape == (4, 1)


def test_all_factories_exposed():
    """Every documented factory must be importable from foundax."""
    for name in (
        "kan", "efficient_kan", "fastkan", "fourier_kan",
        "chebyshev_kan", "jacobi_kan", "legendre_kan", "wavelet_kan",
        "taylor_kan", "kan_conv2d",
    ):
        assert hasattr(fx, name), f"foundax missing factory '{name}'"


def test_all_layers_exposed():
    for name in (
        "KANLayer", "EfficientKANLayer", "FastKANLayer", "FourierKANLayer",
        "ChebyshevKANLayer", "JacobiKANLayer", "LegendreKANLayer",
        "WaveletKANLayer", "TaylorKANLayer", "KANConv2d",
    ):
        assert hasattr(fx.layers, name), f"fx.layers missing '{name}'"
