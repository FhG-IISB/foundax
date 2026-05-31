"""Tests for KANConv2d."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")

import foundax as fx
from foundax.architectures.kan import KANConv2d
from tests._kan_helpers import ks


@pytest.mark.parametrize(
    "basis",
    [
        "bspline",
        "rbf",
        "fourier",
        "chebyshev",
        "jacobi",
        "legendre",
        "wavelet",
        "taylor",
    ],
)
def test_kan_conv2d_shape(basis):
    conv = KANConv2d(
        in_channels=3, out_channels=6, kernel_size=3, basis=basis, key=ks(1)[0]
    )
    y = conv(jnp.ones((8, 8, 3)))
    assert y.shape == (8, 8, 6)


def test_kan_conv2d_sniff():
    conv = KANConv2d(in_channels=3, out_channels=6, kernel_size=3, key=ks(1)[0])
    b = fx.block(conv)
    assert b._in_channels == 3
    assert b._out_channels == 6


def test_kan_conv2d_pipe():
    k1, k2 = ks(2)
    a = fx.block(KANConv2d(3, 8, kernel_size=3, basis="rbf", key=k1))
    b = fx.block(KANConv2d(8, 1, kernel_size=3, basis="chebyshev", key=k2))
    pipe = a | b
    y = pipe(jnp.ones((6, 6, 3)))
    assert y.shape == (6, 6, 1)


def test_kan_conv2d_jit_and_grad():
    conv = KANConv2d(in_channels=2, out_channels=4, kernel_size=3, key=ks(1)[0])
    x = jax.random.normal(ks(1, seed=3)[0], (6, 6, 2))

    @eqx.filter_jit
    def loss_fn(m, x):
        return jnp.mean(m(x) ** 2)

    grad = eqx.filter_grad(loss_fn)(conv, x)
    leaves = [g for g in jax.tree_util.tree_leaves(grad) if eqx.is_array(g)]
    assert leaves
    for g in leaves:
        assert jnp.all(jnp.isfinite(g))


def test_kan_conv2d_unknown_basis_raises():
    with pytest.raises(ValueError):
        KANConv2d(2, 4, kernel_size=3, basis="not_a_basis", key=ks(1)[0])


def test_kan_conv2d_factory():
    m = fx.kan_conv2d(
        3, 6, kernel_size=3, basis="fourier", num_frequencies=4, key=ks(1)[0]
    )
    y = m(jnp.ones((8, 8, 3)))
    assert y.shape == (8, 8, 6)
