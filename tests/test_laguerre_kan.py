"""Tests for LaguerreKANLayer + LaguerreKAN."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np = pytest.importorskip("numpy")

import foundax as fx
from foundax.architectures.kan import LaguerreKANLayer, LaguerreBasis
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


LAYER = lambda i, o, **kw: LaguerreKANLayer(i, o, degree=5, key=kw["key"])


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
    network_train_overfit_sin(lambda **kw: fx.kan.laguerre(**kw, degree=8))


def test_laguerre_matches_numpy_polynomial():
    from numpy.polynomial import laguerre as Llag

    basis = LaguerreBasis(degree=5)
    x = jnp.linspace(-0.7, 0.7, 15).reshape(-1, 1)
    phi = np.asarray(basis(x))[:, 0, :]  # (15, 6)
    # input squashed to [0, ~4] inside basis
    xn = 2.0 * (1.0 + np.tanh(np.asarray(x)[:, 0]))
    ref = np.zeros((xn.shape[0], 6))
    for n in range(6):
        c = np.zeros(n + 1)
        c[n] = 1.0
        ref[:, n] = Llag.lagval(xn, c)
    assert np.allclose(phi, ref, atol=1e-4)


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
