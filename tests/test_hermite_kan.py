"""Tests for HermiteKANLayer + HermiteKAN."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np = pytest.importorskip("numpy")

import foundax as fx
from foundax.architectures.kan import HermiteKANLayer, HermiteBasis
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


LAYER = lambda i, o, **kw: HermiteKANLayer(i, o, degree=5, key=kw["key"])


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
    # Hermite polynomials grow rapidly outside [-1, 1] and converge slower than
    # bounded bases on the sin overfit task, so allow a looser tolerance.
    network_train_overfit_sin(
        lambda **kw: fx.kan.hermite(**kw, degree=8), loss_threshold=2e-1
    )


def test_hermite_recurrence():
    """He_0=1, He_1=x, He_2 = x^2 - 1, He_3 = x^3 - 3x — verify on probabilist Hermite."""
    basis = HermiteBasis(degree=4)
    x = jnp.linspace(-0.7, 0.7, 15).reshape(-1, 1)
    phi = np.asarray(basis(x))[:, 0, :]  # (15, 5)
    xt = np.tanh(np.asarray(x)[:, 0])
    ref = np.stack(
        [
            np.ones_like(xt),
            xt,
            xt**2 - 1.0,
            xt**3 - 3.0 * xt,
            xt**4 - 6.0 * xt**2 + 3.0,
        ],
        axis=-1,
    )
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
