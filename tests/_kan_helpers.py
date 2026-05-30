"""Shared helpers for KAN variant test files."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")

import foundax as fx
from foundax.pipe import ShapeMismatchError


def ks(n: int, seed: int = 0):
    return jax.random.split(jax.random.PRNGKey(seed), n)


def shape_checks(layer_factory, in_features: int, out_features: int):
    """Forward shape for unbatched and 2-D batched inputs."""
    layer = layer_factory(in_features, out_features, key=ks(1)[0])
    # unbatched
    y = layer(jnp.ones((in_features,)))
    assert y.shape == (out_features,)
    # 2-D batched
    y = layer(jnp.ones((7, in_features)))
    assert y.shape == (7, out_features)
    # 3-D leading dims
    y = layer(jnp.ones((3, 4, in_features)))
    assert y.shape == (3, 4, out_features)


def sniff_checks(layer_factory, in_features: int, out_features: int):
    layer = layer_factory(in_features, out_features, key=ks(1)[0])
    b = fx.block(layer)
    assert b._in_channels == in_features
    assert b._out_channels == out_features


def jit_checks(layer_factory, in_features: int, out_features: int):
    layer = layer_factory(in_features, out_features, key=ks(1)[0])

    @eqx.filter_jit
    def fwd(m, x):
        return m(x)

    x = jnp.ones((5, in_features))
    y1 = fwd(layer, x)
    y2 = fwd(layer, jnp.ones((9, in_features)))
    assert y1.shape == (5, out_features)
    assert y2.shape == (9, out_features)


def grad_checks(layer_factory, in_features: int, out_features: int):
    layer = layer_factory(in_features, out_features, key=ks(1)[0])
    x = jax.random.normal(ks(1, seed=2)[0], (4, in_features))

    def loss_fn(m, x):
        return jnp.mean(m(x) ** 2)

    grads = eqx.filter_grad(loss_fn)(layer, x)
    leaves = [g for g in jax.tree_util.tree_leaves(grads) if eqx.is_array(g)]
    assert leaves, "expected at least one gradient leaf"
    for g in leaves:
        assert jnp.all(jnp.isfinite(g)), "non-finite gradient"
    # At least one leaf has a non-zero magnitude (avoid all-zero pathology).
    assert any(jnp.max(jnp.abs(g)) > 0 for g in leaves)


def pipe_checks(layer_factory, in_features: int, hidden: int, out_features: int):
    """fx.block(layer1) | fx.block(layer2) → forward + mismatch raises."""
    a = fx.block(layer_factory(in_features, hidden, key=ks(2)[0]))
    b = fx.block(layer_factory(hidden, out_features, key=ks(2)[1]))
    pipe = a | b
    y = pipe(jnp.ones((6, in_features)))
    assert y.shape == (6, out_features)

    # Mismatched dims should raise at chain time.
    bad = fx.block(layer_factory(hidden + 1, out_features, key=ks(2)[1]))
    with pytest.raises(ShapeMismatchError):
        _ = a | bad


def network_train_overfit_sin(network_factory):
    """Toy training: overfit y = sin(2πx) on [0, 1]. Should reach MSE < 5e-2 fast."""
    optax = pytest.importorskip("optax")

    key = jax.random.PRNGKey(123)
    model = network_factory(in_features=1, output_dim=1, hidden_dims=32,
                            num_layers=2, key=key)

    x = jnp.linspace(0.0, 1.0, 64).reshape(-1, 1)
    y = jnp.sin(2 * jnp.pi * x)

    opt = optax.adam(5e-3)
    state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, state):
        def loss_fn(m):
            return jnp.mean((m(x) - y) ** 2)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, state = opt.update(grads, state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), state, loss

    for _ in range(500):
        model, state, loss = step(model, state)
    assert float(loss) < 5e-2, f"toy training loss too high: {float(loss):.4f}"


def dtype_checks(layer_factory, in_features: int, out_features: int):
    """Forward produces finite outputs at the default float dtype."""
    layer = layer_factory(in_features, out_features, key=ks(1)[0])
    y32 = layer(jnp.ones((3, in_features), dtype=jnp.float32))
    assert jnp.all(jnp.isfinite(y32))
    # If x64 is enabled in this session, also verify float64 forward.
    if jax.config.read("jax_enable_x64"):
        y64 = layer(jnp.ones((3, in_features), dtype=jnp.float64))
        assert jnp.all(jnp.isfinite(y64))


# ---------------------------------------------------------------------------
# Extended shared checks (correctness, robustness, JAX-compatibility)
# ---------------------------------------------------------------------------


def jit_eager_equivalence(layer_factory, in_features: int, out_features: int,
                          atol: float = 1e-5):
    """JIT(forward) must equal eager forward to floating tolerance."""
    layer = layer_factory(in_features, out_features, key=ks(1)[0])
    x = jax.random.normal(ks(1, seed=7)[0], (5, in_features))
    y_eager = layer(x)
    y_jit = eqx.filter_jit(lambda m, x: m(x))(layer, x)
    assert jnp.allclose(y_eager, y_jit, atol=atol), \
        f"JIT/eager mismatch: max abs diff = {jnp.max(jnp.abs(y_eager - y_jit)):.3e}"


def determinism_checks(layer_factory, in_features: int, out_features: int):
    """Same seed → identical layer & identical forward; different seeds differ."""
    seed_key = jax.random.PRNGKey(13)
    a = layer_factory(in_features, out_features, key=seed_key)
    b = layer_factory(in_features, out_features, key=seed_key)
    other = layer_factory(in_features, out_features,
                          key=jax.random.PRNGKey(99))

    x = jax.random.normal(jax.random.PRNGKey(2), (4, in_features))
    ya = a(x); yb = b(x); yo = other(x)
    assert jnp.array_equal(ya, yb), "same-seed forwards differ"
    # Different seeds must produce a different output (assumes non-degenerate basis).
    assert not jnp.allclose(ya, yo, atol=1e-6), "different-seed forwards identical"


def vmap_consistency(layer_factory, in_features: int, out_features: int,
                     atol: float = 1e-5):
    """A batched forward must equal vmap over the unbatched forward."""
    layer = layer_factory(in_features, out_features, key=ks(1)[0])
    x = jax.random.normal(ks(1, seed=8)[0], (7, in_features))
    y_batched = layer(x)
    y_vmapped = jax.vmap(layer)(x)
    assert jnp.allclose(y_batched, y_vmapped, atol=atol), \
        f"batched vs vmap mismatch: max abs diff = {jnp.max(jnp.abs(y_batched - y_vmapped)):.3e}"


def gradient_finite_difference(layer_factory, in_features: int, out_features: int,
                              eps: float = 1e-3, rtol: float = 5e-2):
    """Analytic ∂L/∂x must match a central finite difference (scalar reduction L)."""
    layer = layer_factory(in_features, out_features, key=ks(1)[0])
    x = jax.random.normal(ks(1, seed=11)[0], (in_features,)) * 0.3

    def loss(xv):
        return jnp.sum(layer(xv))

    grad_analytic = jax.grad(loss)(x)

    grad_fd = jnp.zeros_like(x)
    for i in range(x.shape[0]):
        xp = x.at[i].add(eps)
        xm = x.at[i].add(-eps)
        grad_fd = grad_fd.at[i].set((loss(xp) - loss(xm)) / (2 * eps))

    # Relative error on entries with non-trivial magnitude.
    mask = jnp.abs(grad_fd) > 1e-3
    if not jnp.any(mask):
        # All FD components ~ 0; analytic must also be ~ 0
        assert jnp.all(jnp.abs(grad_analytic) < 1e-2)
        return
    rel = jnp.abs(grad_analytic - grad_fd) / (jnp.abs(grad_fd) + 1e-6)
    assert jnp.max(rel[mask]) < rtol, \
        f"FD-grad mismatch: max rel err = {float(jnp.max(rel[mask])):.3e}\n" \
        f"analytic={grad_analytic}\nfinite-diff={grad_fd}"


def pytree_roundtrip(layer_factory, in_features: int, out_features: int):
    """A layer is a valid pytree: leaves+treedef → reconstruct → identical forward."""
    layer = layer_factory(in_features, out_features, key=ks(1)[0])
    leaves, treedef = jax.tree_util.tree_flatten(layer)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    x = jnp.ones((3, in_features))
    assert jnp.array_equal(layer(x), rebuilt(x))


def serialization_roundtrip(layer_factory, in_features: int, out_features: int,
                            tmp_path):
    """eqx.tree_serialise_leaves / deserialise gives back a forward-equivalent layer."""
    layer = layer_factory(in_features, out_features, key=ks(1)[0])
    file = tmp_path / "layer.eqx"
    eqx.tree_serialise_leaves(str(file), layer)
    # Build a "skeleton" using the same constructor and overwrite leaves.
    skeleton = layer_factory(in_features, out_features, key=ks(1, seed=999)[0])
    loaded = eqx.tree_deserialise_leaves(str(file), skeleton)
    x = jax.random.normal(jax.random.PRNGKey(7), (3, in_features))
    assert jnp.allclose(layer(x), loaded(x), atol=1e-6)


def numeric_robustness(layer_factory, in_features: int, out_features: int):
    """Forward stays finite for negative, near-zero, moderately-large inputs."""
    layer = layer_factory(in_features, out_features, key=ks(1)[0])
    for scale in (1e-3, 1.0, 5.0, -5.0):
        x = jnp.full((4, in_features), scale, dtype=jnp.float32)
        y = layer(x)
        assert jnp.all(jnp.isfinite(y)), \
            f"non-finite output at scale={scale}: {y}"


def output_changes_with_input(layer_factory, in_features: int, out_features: int):
    """Two different inputs should produce different outputs (rules out constant maps)."""
    layer = layer_factory(in_features, out_features, key=ks(1)[0])
    x1 = jax.random.normal(jax.random.PRNGKey(0), (in_features,))
    x2 = jax.random.normal(jax.random.PRNGKey(1), (in_features,))
    y1 = layer(x1); y2 = layer(x2)
    assert not jnp.allclose(y1, y2, atol=1e-6), "layer is constant in input"


def block_wrap_roundtrip(layer_factory, in_features: int, out_features: int):
    """fx.block(layer)(x) must equal layer(x) exactly."""
    layer = layer_factory(in_features, out_features, key=ks(1)[0])
    x = jax.random.normal(jax.random.PRNGKey(3), (5, in_features))
    assert jnp.array_equal(layer(x), fx.block(layer)(x))


def grad_through_jit(layer_factory, in_features: int, out_features: int):
    """eqx.filter_jit around a grad-of-loss returns finite, non-zero grads."""
    layer = layer_factory(in_features, out_features, key=ks(1)[0])
    x = jax.random.normal(jax.random.PRNGKey(5), (4, in_features))

    @eqx.filter_jit
    def grad_fn(m, x):
        return eqx.filter_grad(lambda m: jnp.mean(m(x) ** 2))(m)

    g = grad_fn(layer, x)
    leaves = [l for l in jax.tree_util.tree_leaves(g) if eqx.is_array(l)]
    assert leaves
    for l in leaves:
        assert jnp.all(jnp.isfinite(l))
    assert any(float(jnp.max(jnp.abs(l))) > 0 for l in leaves)


def extended_check_suite(layer_factory, in_features: int, out_features: int,
                         tmp_path):
    """Run the full extended battery as a single helper.

    Per-variant test files can either call this directly or call individual
    checks; the individual checks are also exposed so they show up as
    independent test cases in pytest output.
    """
    jit_eager_equivalence(layer_factory, in_features, out_features)
    determinism_checks(layer_factory, in_features, out_features)
    vmap_consistency(layer_factory, in_features, out_features)
    gradient_finite_difference(layer_factory, in_features, out_features)
    pytree_roundtrip(layer_factory, in_features, out_features)
    serialization_roundtrip(layer_factory, in_features, out_features, tmp_path)
    numeric_robustness(layer_factory, in_features, out_features)
    output_changes_with_input(layer_factory, in_features, out_features)
    block_wrap_roundtrip(layer_factory, in_features, out_features)
    grad_through_jit(layer_factory, in_features, out_features)
