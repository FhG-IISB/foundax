"""Tests for EfficientKANLayer + EfficientKAN."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import foundax as fx
from foundax.architectures.kan import EfficientKANLayer, BSplineBasis, KANLayer
from tests._kan_helpers import (
    shape_checks,
    sniff_checks,
    jit_checks,
    grad_checks,
    pipe_checks,
    network_train_overfit_sin,
    dtype_checks,
    jit_eager_equivalence,
    determinism_checks,
    vmap_consistency,
    gradient_finite_difference,
    pytree_roundtrip,
    serialization_roundtrip,
    numeric_robustness,
    output_changes_with_input,
    block_wrap_roundtrip,
    grad_through_jit,
)


LAYER = lambda i, o, **kw: EfficientKANLayer(
    i, o, grid_size=5, spline_order=3, key=kw["key"]
)


def test_shapes():
    shape_checks(LAYER, 4, 8)


def test_sniff():
    sniff_checks(LAYER, 4, 8)


def test_jit():
    jit_checks(LAYER, 4, 8)


def test_grad():
    grad_checks(LAYER, 4, 8)


def test_pipe():
    pipe_checks(LAYER, 3, 16, 1)


def test_dtype():
    dtype_checks(LAYER, 4, 8)


def test_network_train():
    network_train_overfit_sin(
        lambda **kw: fx.kan.efficient(**kw, grid_size=8, spline_order=3)
    )


def test_efficient_basis_matches_bspline():
    """EfficientKAN wraps the same B-spline basis as the original KAN.

    The functional difference in the upstream reference is a memory/layout
    optimisation, not a different basis. So the layer's ``basis`` must be a
    ``BSplineBasis`` and produce identical values to a standalone one with
    matching config."""
    grid_size, spline_order, grid_range = 5, 3, (-1.0, 1.0)
    in_features = 4
    layer = EfficientKANLayer(
        in_features,
        8,
        grid_size=grid_size,
        spline_order=spline_order,
        grid_range=grid_range,
        key=jax.random.PRNGKey(0),
    )
    assert isinstance(layer.basis, BSplineBasis)

    standalone = BSplineBasis(in_features, grid_size, spline_order, grid_range)
    x = jax.random.normal(jax.random.PRNGKey(3), (9, in_features)) * 0.5
    assert jnp.allclose(layer.basis(x), standalone(x), atol=1e-6)


def test_efficient_basis_matches_original_kan():
    """EfficientKANLayer and KANLayer with matching config use the same
    basis function (B-spline). Both should produce identical basis values
    on the same input."""
    grid_size, spline_order, grid_range = 5, 3, (-1.0, 1.0)
    in_features = 4
    eff = EfficientKANLayer(
        in_features,
        8,
        grid_size=grid_size,
        spline_order=spline_order,
        grid_range=grid_range,
        key=jax.random.PRNGKey(0),
    )
    orig = KANLayer(
        in_features,
        8,
        grid_size=grid_size,
        spline_order=spline_order,
        grid_range=grid_range,
        key=jax.random.PRNGKey(0),
    )
    x = jax.random.normal(jax.random.PRNGKey(5), (7, in_features)) * 0.5
    assert jnp.allclose(eff.basis(x), orig.basis(x), atol=1e-6)


# ── Extended correctness / robustness suite ────────────────────────────────


def test_jit_eager_equiv():
    jit_eager_equivalence(LAYER, 4, 8)


def test_determinism():
    determinism_checks(LAYER, 4, 8)


def test_vmap_consistency():
    vmap_consistency(LAYER, 4, 8)


def test_fd_grad():
    gradient_finite_difference(LAYER, 4, 8)


def test_pytree():
    pytree_roundtrip(LAYER, 4, 8)


def test_serialize(tmp_path):
    serialization_roundtrip(LAYER, 4, 8, tmp_path)


def test_numeric_robustness():
    numeric_robustness(LAYER, 4, 8)


def test_changes_with_input():
    output_changes_with_input(LAYER, 4, 8)


def test_block_roundtrip():
    block_wrap_roundtrip(LAYER, 4, 8)


def test_grad_through_jit():
    grad_through_jit(LAYER, 4, 8)
