"""Cartesian-product compatibility tests across all KAN variants.

Verifies that any pair of variants composes correctly through ``|`` — the
core promise of the pipe API. With 17 variants this generates 17×17 = 289
parametrized cases, of which (variant_a, variant_b) for all combinations
is the key sanity check.
"""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")

import foundax as fx
from foundax.architectures.kan import (
    KANLayer, EfficientKANLayer, FastKANLayer, FourierKANLayer,
    ChebyshevKANLayer, JacobiKANLayer, LegendreKANLayer, WaveletKANLayer,
    TaylorKANLayer, HermiteKANLayer, LaguerreKANLayer, BernsteinKANLayer,
    ReLUKANLayer, RationalKANLayer, SincKANLayer, GramKANLayer, BSRBFKANLayer,
)


ALL_LAYERS = [
    KANLayer, EfficientKANLayer, FastKANLayer, FourierKANLayer,
    ChebyshevKANLayer, JacobiKANLayer, LegendreKANLayer, WaveletKANLayer,
    TaylorKANLayer, HermiteKANLayer, LaguerreKANLayer, BernsteinKANLayer,
    ReLUKANLayer, RationalKANLayer, SincKANLayer, GramKANLayer, BSRBFKANLayer,
]

NAMES = [c.__name__ for c in ALL_LAYERS]


def _ks(n, seed=0):
    return jax.random.split(jax.random.PRNGKey(seed), n)


@pytest.mark.parametrize("cls_a", ALL_LAYERS, ids=NAMES)
@pytest.mark.parametrize("cls_b", ALL_LAYERS, ids=NAMES)
def test_pair_pipe_forward(cls_a, cls_b):
    """Any two variants must compose with `|` and produce finite output of the right shape."""
    k1, k2 = _ks(2)
    a = fx.block(cls_a(4, 8, key=k1))
    b = fx.block(cls_b(8, 2, key=k2))
    pipe = a | b
    y = pipe(jnp.ones((3, 4)))
    assert y.shape == (3, 2)
    assert jnp.all(jnp.isfinite(y))


@pytest.mark.parametrize("cls", ALL_LAYERS, ids=NAMES)
def test_variant_in_deeponet_branch(cls):
    """Every variant can serve as a DeepONet-style branch under `fx.dot`."""
    k1, k2 = _ks(2)
    branch = fx.block(cls(3, 16, key=k1))
    trunk = fx.block(fx.mlp(in_features=2, output_dim=16, hidden_dims=16, key=k2))
    op = fx.dot(branch, trunk)
    u = jnp.ones((3,))
    y = jnp.ones((7, 2))
    out = op(u, y)
    assert out.shape == (7,)
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("cls", ALL_LAYERS, ids=NAMES)
def test_variant_jit_pipeline(cls):
    """Pipeline of identical-variant layers JITs end-to-end."""
    k1, k2, k3 = _ks(3)
    pipe = (
        fx.block(cls(2, 16, key=k1))
        | fx.block(cls(16, 16, key=k2))
        | fx.block(cls(16, 1, key=k3))
    )

    @eqx.filter_jit
    def fwd(m, x):
        return m(x)

    y = fwd(pipe, jnp.ones((4, 2)))
    assert y.shape == (4, 1)
    assert jnp.all(jnp.isfinite(y))


@pytest.mark.parametrize("cls", ALL_LAYERS, ids=NAMES)
def test_variant_grad_does_not_explode(cls):
    """Backprop through 3 stacked layers should give finite gradients."""
    k1, k2, k3 = _ks(3)
    model = (
        fx.block(cls(2, 8, key=k1))
        | fx.block(cls(8, 8, key=k2))
        | fx.block(cls(8, 1, key=k3))
    )
    x = jax.random.normal(_ks(1, seed=4)[0], (6, 2))
    grad = eqx.filter_grad(lambda m: jnp.mean(m(x) ** 2))(model)
    leaves = [g for g in jax.tree_util.tree_leaves(grad) if eqx.is_array(g)]
    assert leaves
    for g in leaves:
        assert jnp.all(jnp.isfinite(g))
