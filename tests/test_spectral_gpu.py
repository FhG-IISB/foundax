"""Inverse real FFTs of mixed spectra give the CPU (pocketfft) result on every backend (B10)."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")

import foundax as fx  # noqa: E402
from foundax.architectures.common import irfftn_hermitian  # noqa: E402

KEY = jax.random.PRNGKey(0)


@pytest.mark.parametrize(
    "shape, s, axes",
    [((16, 9, 3), (16, 16), (0, 1)), ((16, 9, 3), (16, 17), (0, 1)), ((4, 8, 5, 2), (4, 8, 8), (0, 1, 2)),
     ((2, 12, 7), (12, 12), (1, 2))],
    ids=["2d-even", "2d-odd", "3d", "batched"],
)  # fmt: skip
def test_irfftn_hermitian_is_the_cpu_inverse(shape, s, axes):
    rng = np.random.default_rng(0)
    x = rng.normal(size=shape) + 1j * rng.normal(size=shape)  # not Hermitian
    np.testing.assert_allclose(
        irfftn_hermitian(jnp.asarray(x), s=s, axes=axes),
        np.fft.irfftn(x, s=s, axes=axes),
        atol=1e-6,
    )


def _afno():
    fx.dpot.ensure_repo_on_path("jax_dpot")
    from jax_dpot.model_eqx import AFNO2D

    return AFNO2D(32, num_blocks=4, modes=32, key=KEY)


def _geo_conv():
    from foundax.architectures.geofno import SpectralConv2dGeo

    return SpectralConv2dGeo(8, 8, 16, 16, key=KEY)


def _x(*shape):
    return jax.random.normal(jax.random.PRNGKey(1), shape)


LAYERS = {
    "dpot-afno2d": (_afno, lambda: (_x(1, 128, 128, 32),)),
    "geofno-conv": (_geo_conv, lambda: (_x(128, 128, 8),)),
    "fno2d": (lambda: fx.fno2d(8, 16, 16, n_layers=2, d_model=(128, 128), linear_conv=False, key=KEY), lambda: (_x(128, 128, 8),)),
    "ffno2d": (lambda: fx.ffno2d(8, 16, n_modes=16, n_layers=2, key=KEY), lambda: (_x(128, 128, 8),)),
    "sfno2d": (lambda: fx.sfno2d(8, 1, hidden_channels=16, L=32, nlat=64, nlon=128, n_layers=2, key=KEY), lambda: (_x(64, 128, 8),)),
}  # fmt: skip


def _gpu():
    try:
        return jax.devices("gpu")[0]
    except RuntimeError:
        return None


@pytest.mark.gpu
@pytest.mark.parametrize("name", LAYERS)
def test_spectral_layers_agree_on_cpu_and_gpu(name):
    """At n=128 cuFFT takes its power-of-2 path, where a non-Hermitian inverse differed from the CPU."""
    gpu = _gpu()
    if gpu is None:
        pytest.skip("no GPU")
    build, inputs = LAYERS[name]
    with jax.default_device(jax.devices("cpu")[0]):
        model, args = build(), inputs()
    outs = []
    for device in (jax.devices("cpu")[0], gpu):
        m, a = jax.device_put((model, args), device)
        with jax.default_matmul_precision("highest"):  # no TF32 on the GPU
            outs.append(np.asarray(eqx.filter_jit(lambda m, a: m(*a))(m, a)))
    rel = np.max(np.abs(outs[0] - outs[1])) / np.max(np.abs(outs[0]))
    assert rel < 1e-5, f"GPU differs from CPU by {rel:.1e}"
