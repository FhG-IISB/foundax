"""Tests for the original B-spline KAN layer + network."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np = pytest.importorskip("numpy")

import foundax as fx
from foundax.architectures.kan import KANLayer, BSplineBasis
from tests._kan_helpers import (
    shape_checks, sniff_checks, jit_checks, grad_checks,
    pipe_checks, network_train_overfit_sin, dtype_checks, ks,
    jit_eager_equivalence, determinism_checks, vmap_consistency,
    gradient_finite_difference, pytree_roundtrip, serialization_roundtrip,
    numeric_robustness, output_changes_with_input, block_wrap_roundtrip,
    grad_through_jit,
)


LAYER = lambda i, o, **kw: KANLayer(i, o, grid_size=5, spline_order=3, key=kw["key"])


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
    network_train_overfit_sin(lambda **kw: fx.kan(**kw, grid_size=8, spline_order=3))


def test_bspline_partition_of_unity():
    """Sum of B-spline basis functions equals 1 on the interior (de Boor)."""
    basis = BSplineBasis(in_features=1, grid_size=10, spline_order=3,
                         grid_range=(-1.0, 1.0))
    x = jnp.linspace(-0.7, 0.7, 50).reshape(-1, 1)
    phi = basis(x)  # (50, 1, G+k)
    assert jnp.allclose(phi.sum(axis=-1), 1.0, atol=1e-5)


def test_bspline_matches_scipy():
    """B-spline basis must match scipy.interpolate.BSpline evaluation."""
    scipy_interp = pytest.importorskip("scipy.interpolate")
    grid_size, k = 5, 3
    basis = BSplineBasis(in_features=1, grid_size=grid_size, spline_order=k,
                         grid_range=(-1.0, 1.0))
    x = jnp.linspace(-0.5, 0.5, 20).reshape(-1, 1)
    phi = np.asarray(basis(x))[:, 0, :]  # (20, G+k)
    knots = np.asarray(basis.grid[0])
    # scipy: for each basis index, build a unit-coefficient spline.
    n = grid_size + k  # number of basis functions
    ref = np.zeros((x.shape[0], n))
    for j in range(n):
        coeffs = np.zeros(len(knots) - k - 1)
        coeffs[j] = 1.0
        sp = scipy_interp.BSpline(knots, coeffs, k, extrapolate=False)
        ref[:, j] = np.nan_to_num(sp(np.asarray(x)[:, 0]))
    assert np.allclose(phi, ref, atol=1e-5)


def test_pipe_mixed_with_mlp():
    """KAN layer should chain seamlessly with a plain MLP."""
    k1, k2 = ks(2)
    a = fx.block(KANLayer(4, 16, key=k1))
    b = fx.block(fx.mlp(in_features=16, output_dim=1, hidden_dims=16, key=k2))
    pipe = a | b
    y = pipe(jnp.ones((5, 4)))
    assert y.shape == (5, 1)


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
