"""Tests for RationalKANLayer + RationalKAN (rational-Chebyshev / Padé)."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np = pytest.importorskip("numpy")

import foundax as fx
from foundax.architectures.kan import RationalKANLayer, RationalBasis
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


LAYER = lambda i, o, **kw: RationalKANLayer(i, o, degree=5, key=kw["key"])


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
    network_train_overfit_sin(lambda **kw: fx.rational_kan(**kw, degree=8))


def test_rational_matches_closed_form():
    """phi_k(x) = T_k(tanh(x)) / (1 + tanh(x)**2)."""
    from numpy.polynomial import chebyshev as Tcheb

    basis = RationalBasis(degree=4)
    x = jnp.linspace(-0.8, 0.8, 20).reshape(-1, 1)
    phi = np.asarray(basis(x))[:, 0, :]  # (20, 5)
    xt = np.tanh(np.asarray(x)[:, 0])
    denom = 1.0 + xt * xt
    ref = np.zeros((xt.shape[0], 5))
    for n in range(5):
        c = np.zeros(n + 1)
        c[n] = 1.0
        ref[:, n] = Tcheb.chebval(xt, c) / denom
    assert np.allclose(phi, ref, atol=1e-5)


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
