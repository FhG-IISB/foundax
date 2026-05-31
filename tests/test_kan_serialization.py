"""Serialization, pytree, and JAX-compatibility tests across KAN variants."""

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


def _k(seed=0):
    return jax.random.PRNGKey(seed)


# ── eqx.tree_serialise / deserialise roundtrip ─────────────────────────────


@pytest.mark.parametrize("cls", ALL_LAYERS, ids=NAMES)
def test_serialize_layer_roundtrip(cls, tmp_path):
    layer = cls(4, 6, key=_k())
    file = tmp_path / "layer.eqx"
    eqx.tree_serialise_leaves(str(file), layer)
    skeleton = cls(4, 6, key=_k(999))  # different init → will be overwritten
    loaded = eqx.tree_deserialise_leaves(str(file), skeleton)
    x = jax.random.normal(_k(2), (3, 4))
    assert jnp.allclose(layer(x), loaded(x), atol=1e-6)


# ── Pipe of mixed variants survives serialise/deserialise ──────────────────


def test_serialize_mixed_pipe(tmp_path):
    k1, k2, k3 = jax.random.split(_k(), 3)
    pipe = (
        fx.block(FastKANLayer(2, 16, key=k1))
        | fx.block(ChebyshevKANLayer(16, 8, degree=4, key=k2))
        | fx.block(FastKANLayer(8, 1, key=k3))
    )
    file = tmp_path / "pipe.eqx"
    eqx.tree_serialise_leaves(str(file), pipe)
    # Rebuild skeleton with same structure, then load.
    s1, s2, s3 = jax.random.split(jax.random.PRNGKey(999), 3)
    skeleton = (
        fx.block(FastKANLayer(2, 16, key=s1))
        | fx.block(ChebyshevKANLayer(16, 8, degree=4, key=s2))
        | fx.block(FastKANLayer(8, 1, key=s3))
    )
    loaded = eqx.tree_deserialise_leaves(str(file), skeleton)
    x = jax.random.normal(_k(7), (5, 2))
    assert jnp.allclose(pipe(x), loaded(x), atol=1e-6)


# ── Pytree flatten/unflatten preserves forward ─────────────────────────────


@pytest.mark.parametrize("cls", ALL_LAYERS, ids=NAMES)
def test_pytree_flatten_unflatten(cls):
    layer = cls(3, 5, key=_k())
    leaves, treedef = jax.tree_util.tree_flatten(layer)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    x = jnp.linspace(-1, 1, 12).reshape(4, 3)
    assert jnp.array_equal(layer(x), rebuilt(x))


# ── eqx.filter partitions cleanly into arrays + static ─────────────────────


@pytest.mark.parametrize("cls", ALL_LAYERS, ids=NAMES)
def test_eqx_filter_partition(cls):
    layer = cls(3, 5, key=_k())
    arrays, static = eqx.partition(layer, eqx.is_array)
    # Round-trip through combine.
    layer2 = eqx.combine(arrays, static)
    x = jnp.ones((4, 3))
    assert jnp.array_equal(layer(x), layer2(x))


# ── Replacing leaves via eqx.tree_at preserves structure ───────────────────


def test_tree_at_replaces_spline_weight():
    layer = FastKANLayer(4, 6, grid_size=8, key=_k())
    new_w = jnp.ones_like(layer.spline_weight)
    layer2 = eqx.tree_at(lambda m: m.spline_weight, layer, new_w)
    assert jnp.array_equal(layer2.spline_weight, new_w)
    # Forward should differ from the original (assuming x ≠ 0)
    x = jax.random.normal(_k(3), (4, 4))
    assert not jnp.allclose(layer(x), layer2(x), atol=1e-6)


# ── jax.jit (raw) handles a KAN layer transparently ───────────────────────


@pytest.mark.parametrize("cls", ALL_LAYERS, ids=NAMES)
def test_jax_jit_on_layer(cls):
    layer = cls(3, 5, key=_k())

    @jax.jit
    def fwd(x):
        return layer(x)

    x = jnp.ones((4, 3))
    y = fwd(x)
    assert y.shape == (4, 5)


# ── grad via jax.grad (loss reduces to scalar) ─────────────────────────────


@pytest.mark.parametrize("cls", ALL_LAYERS, ids=NAMES)
def test_jax_grad_wrt_input(cls):
    layer = cls(3, 5, key=_k())

    def loss(x):
        return jnp.sum(layer(x) ** 2)

    g = jax.grad(loss)(jax.random.normal(_k(4), (3,)))
    assert jnp.all(jnp.isfinite(g))
    assert jnp.max(jnp.abs(g)) > 0
