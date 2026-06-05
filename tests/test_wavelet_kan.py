"""Tests for WaveletKANLayer + WaveletKAN."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
np = pytest.importorskip("numpy")

import foundax as fx
from foundax.architectures.kan import WaveletKANLayer, WaveletBasis
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


LAYER = lambda i, o, **kw: WaveletKANLayer(
    i, o, num_scales=6, wavelet_type="mexican_hat", key=kw["key"]
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
        lambda **kw: fx.kan.wavelet(**kw, num_scales=6, wavelet_type="morlet")
    )


@pytest.mark.parametrize("wavelet", ["mexican_hat", "morlet", "shannon", "dog"])
def test_wavelet_types(wavelet):
    basis = WaveletBasis(num_scales=4, wavelet_type=wavelet)
    x = jnp.linspace(-2.0, 2.0, 20).reshape(-1, 1)
    phi = basis(x)
    assert phi.shape == (20, 1, 4)
    assert jnp.all(jnp.isfinite(phi))


def test_wavelet_unknown_raises():
    with pytest.raises(ValueError):
        WaveletBasis(num_scales=4, wavelet_type="not_a_wavelet")


def test_mexican_hat_closed_form():
    """ψ(x) = (1 - x²) exp(-x²/2) at scale=1 (the first scale 2**0)."""
    basis = WaveletBasis(num_scales=1, wavelet_type="mexican_hat")
    x = jnp.linspace(-2.0, 2.0, 15).reshape(-1, 1)
    phi = np.asarray(basis(x))[:, 0, 0]
    xn = np.asarray(x)[:, 0]
    ref = (1.0 - xn * xn) * np.exp(-0.5 * xn * xn)
    assert np.allclose(phi, ref, atol=1e-6)


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
