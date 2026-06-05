"""Tests for JacobiKANLayer + JacobiKAN."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np = pytest.importorskip("numpy")
sp = pytest.importorskip("scipy.special")

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
        lambda **kw: fx.kan.jacobi(**kw, degree=8, alpha=1.0, beta=1.0)
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


@pytest.mark.parametrize("alpha,beta", [(0.5, 0.5), (1.0, 1.0), (2.0, 1.5)])
def test_jacobi_matches_scipy_eval_jacobi(alpha, beta):
    """JacobiBasis must match scipy.special.eval_jacobi for all degrees, all
    (alpha, beta). This exercises the full three-term recurrence, not just
    P_0 and P_1."""
    degree = 6
    basis = JacobiBasis(degree=degree, alpha=alpha, beta=beta)
    x = jnp.linspace(-0.9, 0.9, 25).reshape(-1, 1)
    phi = np.asarray(basis(x))[:, 0, :]  # (25, degree+1)
    xn = np.tanh(np.asarray(x)[:, 0])
    ref = np.stack(
        [sp.eval_jacobi(n, alpha, beta, xn) for n in range(degree + 1)],
        axis=-1,
    )
    assert np.allclose(phi, ref, atol=1e-5), (
        f"max diff = {np.max(np.abs(phi - ref)):.3e}"
    )


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
