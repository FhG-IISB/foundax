"""Tests for BSRBFKANLayer + BSRBFKAN (B-spline + RBF hybrid)."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import foundax as fx
from foundax.architectures.kan import BSRBFKANLayer, BSRBFBasis
from tests._kan_helpers import (
    shape_checks, sniff_checks, jit_checks, grad_checks,
    pipe_checks, network_train_overfit_sin, dtype_checks,
    jit_eager_equivalence, determinism_checks, vmap_consistency,
    gradient_finite_difference, pytree_roundtrip, serialization_roundtrip,
    numeric_robustness, output_changes_with_input, block_wrap_roundtrip,
    grad_through_jit,
)


LAYER = lambda i, o, **kw: BSRBFKANLayer(i, o, grid_size=5, rbf_grid_size=8,
                                          key=kw["key"])


def test_shapes(): shape_checks(LAYER, 4, 8)
def test_sniff(): sniff_checks(LAYER, 4, 8)
def test_jit(): jit_checks(LAYER, 4, 8)
def test_grad(): grad_checks(LAYER, 4, 8)
def test_pipe(): pipe_checks(LAYER, 3, 16, 1)
def test_dtype(): dtype_checks(LAYER, 4, 8)


def test_network_train():
    network_train_overfit_sin(
        lambda **kw: fx.bsrbf_kan(**kw, grid_size=8, rbf_grid_size=8)
    )


def test_bsrbf_size_is_sum():
    basis = BSRBFBasis(in_features=1, grid_size=5, spline_order=3,
                       rbf_grid_size=8)
    assert basis.size == basis.bspline.size + basis.rbf.size  # (5+3) + 8 = 16
    x = jnp.linspace(-0.5, 0.5, 7).reshape(-1, 1)
    phi = basis(x)
    assert phi.shape == (7, 1, basis.size)


# ── Extended correctness / robustness suite ────────────────────────────────


def test_jit_eager_equiv(): jit_eager_equivalence(LAYER, 4, 8)
def test_determinism(): determinism_checks(LAYER, 4, 8)
def test_vmap_consistency(): vmap_consistency(LAYER, 4, 8)
def test_fd_grad(): gradient_finite_difference(LAYER, 4, 8)
def test_pytree(): pytree_roundtrip(LAYER, 4, 8)
def test_serialize(tmp_path): serialization_roundtrip(LAYER, 4, 8, tmp_path)
def test_numeric_robustness(): numeric_robustness(LAYER, 4, 8)
def test_changes_with_input(): output_changes_with_input(LAYER, 4, 8)
def test_block_roundtrip(): block_wrap_roundtrip(LAYER, 4, 8)
def test_grad_through_jit(): grad_through_jit(LAYER, 4, 8)
