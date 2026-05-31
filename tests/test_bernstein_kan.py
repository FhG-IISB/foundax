"""Tests for BernsteinKANLayer + BernsteinKAN."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np = pytest.importorskip("numpy")

import foundax as fx
from foundax.architectures.kan import BernsteinKANLayer, BernsteinBasis
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


LAYER = lambda i, o, **kw: BernsteinKANLayer(i, o, degree=5, key=kw["key"])


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
    network_train_overfit_sin(lambda **kw: fx.bernstein_kan(**kw, degree=8))


def test_bernstein_partition_of_unity():
    """sum_i B_{i,n}(x) == 1 for any x — Bernstein basis is a partition of unity."""
    basis = BernsteinBasis(degree=6)
    x = jnp.linspace(-3.0, 3.0, 20).reshape(-1, 1)
    phi = basis(x)
    assert jnp.allclose(phi.sum(axis=-1), 1.0, atol=1e-5)


def test_bernstein_closed_form_at_one_point():
    from math import comb

    basis = BernsteinBasis(degree=4)
    x = jnp.array([[0.0]])  # sigmoid(0) = 0.5
    phi = np.asarray(basis(x))[0, 0, :]
    u = 0.5
    ref = np.array([comb(4, i) * (u**i) * ((1 - u) ** (4 - i)) for i in range(5)])
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
