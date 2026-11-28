"""Normalisation layers on one channel-last sample, and the ``norm`` option of the models that use them (B2).

``make_norm(kind, channels)`` returns a module for a field ``(..., C)`` of any spatial rank. ``"batch"`` raises:
under the unbatched convention there is no batch to take statistics over (vmapped over the points, the old
``BatchNorm`` normalised over zero axes and returned its bias).
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import foundax as fx  # noqa: E402
import foundax.layers as fl  # noqa: E402
from foundax.architectures.common import make_norm  # noqa: E402

KEY = jax.random.PRNGKey(0)
KINDS = ["layer", "rms", "instance", "group"]


def _field(*shape):
    return jax.random.normal(KEY, shape) * 3.0 + 1.5


@pytest.mark.parametrize(
    "shape", [(16, 8), (6, 5, 8), (3, 4, 5, 8)], ids=["1d", "2d", "3d"]
)
def test_statistics_of_each_kind(shape):
    x = _field(*shape)
    spatial = tuple(range(len(shape) - 1))

    y = make_norm("layer", 8)(x)
    np.testing.assert_allclose(y.mean(-1), 0, atol=1e-5)
    np.testing.assert_allclose(y.var(-1), 1, rtol=1e-3)

    y = make_norm("rms", 8)(x)
    np.testing.assert_allclose(jnp.mean(y**2, -1), 1, rtol=1e-3)

    y = make_norm("instance", 8)(x)
    np.testing.assert_allclose(y.mean(spatial), 0, atol=1e-5)
    np.testing.assert_allclose(y.var(spatial), 1, rtol=1e-3)

    y = make_norm("group", 8, groups=2)(x).reshape(shape[:-1] + (2, 4))
    np.testing.assert_allclose(y.mean(spatial + (-1,)), 0, atol=1e-5)
    np.testing.assert_allclose(y.var(spatial + (-1,)), 1, rtol=1e-3)


def test_none_and_unknown_and_batch():
    assert make_norm(None, 8) is None
    with pytest.raises(ValueError, match="instance"):
        make_norm("batch", 8)
    with pytest.raises(ValueError, match="unknown"):
        make_norm("weight", 8)
    with pytest.raises(ValueError, match="divide"):
        make_norm("group", 8, groups=3)


def _ramp(*spatial, c=2):
    grids = jnp.meshgrid(*[jnp.linspace(0, 1, n) for n in spatial], indexing="ij")
    return jnp.stack([sum(grids)] * c, axis=-1)


MODELS = {
    "fno1d": (lambda norm, k: fx.fno1d(2, 8, 4, n_layers=2, norm=norm, key=k), (32,)),
    "fno2d": (
        lambda norm, k: fx.fno2d(
            2, 8, 4, n_layers=2, d_model=(16, 16), norm=norm, key=k
        ),
        (16, 16),
    ),
    "fno3d": (
        lambda norm, k: fx.fno3d(
            2, 8, 4, n_layers=2, d_model=(8, 8, 8), norm=norm, key=k
        ),
        (8, 8, 8),
    ),
    "SpectralBlock1d": (
        lambda norm, k: fl.SpectralBlock1d(2, 8, n_modes=4, norm=norm, key=k),
        (32,),
    ),
    "SpectralBlock2d": (
        lambda norm, k: fl.SpectralBlock2d(2, 8, n_modes=4, norm=norm, key=k),
        (16, 16),
    ),
    "SpectralBlock3d": (
        lambda norm, k: fl.SpectralBlock3d(2, 8, n_modes=4, norm=norm, key=k),
        (8, 8, 8),
    ),
}


@pytest.mark.parametrize("kind", [None] + KINDS)
@pytest.mark.parametrize("model", MODELS)
def test_model_norm_keeps_the_field(model, kind):
    """A normalised model still sees the input: its output on a ramp is not constant."""
    build, spatial = MODELS[model]
    y = build(kind, KEY)(_ramp(*spatial))
    assert float(jnp.std(y)) > 1e-3 * float(jnp.mean(jnp.abs(y))) + 1e-6


@pytest.mark.parametrize("model", MODELS)
def test_model_norm_batch_raises(model):
    with pytest.raises(ValueError, match="instance"):
        MODELS[model][0]("batch", KEY)


def test_fno1d_instance_is_instance_norm():
    model = fx.fno1d(2, 8, 4, n_layers=1, norm="instance", key=KEY)
    norm = model.spectral_layers.norm_layers[0]
    x = _field(32, 8)
    np.testing.assert_allclose(norm(x).mean(0), 0, atol=1e-5)
