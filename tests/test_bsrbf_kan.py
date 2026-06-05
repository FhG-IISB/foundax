"""Tests for BSRBFKANLayer + BSRBFKAN (B-spline + RBF hybrid)."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import foundax as fx
from foundax.architectures.kan import (
    BSRBFKANLayer,
    BSRBFBasis,
    BSplineBasis,
    RBFBasis,
)
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


LAYER = lambda i, o, **kw: BSRBFKANLayer(
    i, o, grid_size=5, rbf_grid_size=8, key=kw["key"]
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
        lambda **kw: fx.kan.bsrbf(**kw, grid_size=8, rbf_grid_size=8)
    )


def test_bsrbf_size_is_sum():
    basis = BSRBFBasis(in_features=1, grid_size=5, spline_order=3, rbf_grid_size=8)
    assert basis.size == basis.bspline.size + basis.rbf.size  # (5+3) + 8 = 16
    x = jnp.linspace(-0.5, 0.5, 7).reshape(-1, 1)
    phi = basis(x)
    assert phi.shape == (7, 1, basis.size)


def test_bsrbf_is_concatenation_of_components():
    """BSRBFBasis(x) must equal concat([BSplineBasis(x), RBFBasis(x)], axis=-1)."""
    in_features = 3
    grid_size, spline_order, rbf_grid_size = 5, 3, 8
    grid_range = (-1.0, 1.0)
    rbf_grid_range = (-2.0, 2.0)
    hybrid = BSRBFBasis(
        in_features=in_features,
        grid_size=grid_size,
        spline_order=spline_order,
        rbf_grid_size=rbf_grid_size,
        grid_range=grid_range,
        rbf_grid_range=rbf_grid_range,
    )
    bspline = BSplineBasis(in_features, grid_size, spline_order, grid_range)
    rbf = RBFBasis(rbf_grid_size, rbf_grid_range)
    x = jax.random.normal(jax.random.PRNGKey(7), (11, in_features)) * 0.6
    phi_hybrid = hybrid(x)
    phi_concat = jnp.concatenate([bspline(x), rbf(x)], axis=-1)
    assert phi_hybrid.shape == phi_concat.shape == (11, in_features, hybrid.size)
    assert jnp.allclose(phi_hybrid, phi_concat, atol=1e-6)
    # The B-spline slice is positions [: bspline.size]; RBF fills the rest.
    assert jnp.allclose(phi_hybrid[..., : bspline.size], bspline(x), atol=1e-6)
    assert jnp.allclose(phi_hybrid[..., bspline.size :], rbf(x), atol=1e-6)


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
