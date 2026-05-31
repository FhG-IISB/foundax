"""Tests for JacobiKANLayer + JacobiKAN."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import foundax as fx
from foundax.architectures.kan import JacobiKANLayer, JacobiBasis
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


LAYER = lambda i, o, **kw: JacobiKANLayer(
    i, o, degree=5, alpha=1.0, beta=1.0, key=kw["key"]
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
        lambda **kw: fx.jacobi_kan(**kw, degree=8, alpha=1.0, beta=1.0)
    )


def test_jacobi_at_zero():
    """P_0(x)=1, P_1(x) = ((a-b)+(a+b+2)x)/2 for any (alpha, beta)."""
    basis = JacobiBasis(degree=3, alpha=2.0, beta=1.0)
    x = jnp.array([[0.1], [0.5]])
    phi = basis(x)[:, 0, :]
    assert jnp.allclose(phi[:, 0], 1.0, atol=1e-6)
    xt = jnp.tanh(x[:, 0])
    expected = 0.5 * ((2.0 - 1.0) + (2.0 + 1.0 + 2.0) * xt)
    assert jnp.allclose(phi[:, 1], expected, atol=1e-6)


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
