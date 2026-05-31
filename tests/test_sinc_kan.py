"""Tests for SincKANLayer + SincKAN."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np = pytest.importorskip("numpy")

import foundax as fx
from foundax.architectures.kan import SincKANLayer, SincBasis
from tests._kan_helpers import (
    shape_checks, sniff_checks, jit_checks, grad_checks,
    pipe_checks, network_train_overfit_sin, dtype_checks,
    jit_eager_equivalence, determinism_checks, vmap_consistency,
    gradient_finite_difference, pytree_roundtrip, serialization_roundtrip,
    numeric_robustness, output_changes_with_input, block_wrap_roundtrip,
    grad_through_jit,
)


LAYER = lambda i, o, **kw: SincKANLayer(i, o, grid_size=8, key=kw["key"])


def test_shapes(): shape_checks(LAYER, 4, 8)
def test_sniff(): sniff_checks(LAYER, 4, 8)
def test_jit(): jit_checks(LAYER, 4, 8)
def test_grad(): grad_checks(LAYER, 4, 8)
def test_pipe(): pipe_checks(LAYER, 3, 16, 1)
def test_dtype(): dtype_checks(LAYER, 4, 8)


def test_network_train():
    network_train_overfit_sin(lambda **kw: fx.sinc_kan(**kw, grid_size=16))


def test_sinc_basis_matches_closed_form():
    basis = SincBasis(grid_size=8, grid_range=(-2.0, 2.0))
    x = jnp.linspace(-1.5, 1.5, 20).reshape(-1, 1)
    phi = np.asarray(basis(x))[:, 0, :]  # (20, 8)
    centers = np.asarray(basis.centers)
    h = (2.0 - (-2.0)) / (8 - 1)
    ref = np.sinc((np.asarray(x)[:, 0:1] - centers) / h)
    assert np.allclose(phi, ref, atol=1e-6)


def test_sinc_basis_kronecker_at_centers():
    """phi_k evaluated at center c_k = 1, at other centers = 0 (Kronecker)."""
    basis = SincBasis(grid_size=6, grid_range=(-1.0, 1.0))
    cs = basis.centers.reshape(-1, 1)
    phi = basis(cs)[:, 0, :]  # (6, 6), phi[i, k] = sinc((c_i - c_k) / h)
    assert jnp.allclose(jnp.diag(phi), 1.0, atol=1e-6)
    off_diag = phi - jnp.eye(6)
    assert jnp.allclose(off_diag, 0.0, atol=1e-6)


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
