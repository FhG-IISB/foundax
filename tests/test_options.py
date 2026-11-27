"""Options that were accepted and ignored, or did something other than documented (B7)."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import foundax as fx  # noqa: E402
from foundax.architectures.common import Norm  # noqa: E402
from foundax.architectures.gnot import MLP as GnotMLP  # noqa: E402
from foundax.architectures.kan import KANConv2d  # noqa: E402

KEY = jax.random.PRNGKey(0)


def _x(*shape, seed=1):
    return jax.random.normal(jax.random.PRNGKey(seed), shape)


# -- GNOT ----------------------------------------------------------------------------------------


def _gnot(attn_type):
    return fx.gnot(2, [3], n_layers=1, n_hidden=8, attn_type=attn_type, key=KEY)


def test_gnot_attn_type_reaches_the_attention():
    xt, xb = _x(1, 5, 2), [_x(1, 7, 3, seed=2)]
    out = {a: _gnot(a)(xt, xb) for a in ("linear", "l1", "galerkin", "l2")}
    np.testing.assert_array_equal(
        out["linear"], out["l1"]
    )  # "linear" is the l1 kernel, the upstream default
    assert float(jnp.abs(out["galerkin"] - out["l1"]).max()) > 1e-5
    assert float(jnp.abs(out["l2"] - out["l1"]).max()) > 1e-5
    with pytest.raises(ValueError, match="attn_type"):
        _gnot("softmax")


def test_gnot_mlp_layer_count():
    assert len(GnotMLP(3, 8, 2, n_layers=1, key=KEY).layers) == 1
    layers = GnotMLP(3, 8, 2, n_layers=3, key=KEY).layers
    assert len(layers) == 3
    assert not np.allclose(
        layers[0].weight[:2, :3], layers[-1].weight[:2, :3]
    )  # not one key for all


# -- DeepONet ------------------------------------------------------------------------------------


@pytest.mark.parametrize("norm", [None, "layer", "rms", "group"])
def test_deeponet_norms(norm):
    model = fx.deeponet(
        n_sensors=10, basis_functions=8, hidden_dim=16, n_layers=2, norm=norm, key=KEY
    )
    out = model(_x(10), jax.random.uniform(KEY, (5, 1)))
    assert out.shape == (5,) and bool(jnp.all(jnp.isfinite(out)))


def test_deeponet_rms_is_rms_norm_and_batch_raises():
    model = fx.deeponet(
        n_sensors=10, basis_functions=8, hidden_dim=16, n_layers=2, norm="rms", key=KEY
    )
    norms = [
        m
        for m in jax.tree.leaves(model, is_leaf=lambda m: isinstance(m, Norm))
        if isinstance(m, Norm)
    ]
    assert norms and all(n.kind == "rms" for n in norms)
    with pytest.raises(ValueError, match="instance"):
        fx.deeponet(
            n_sensors=10,
            basis_functions=8,
            hidden_dim=16,
            n_layers=2,
            norm="batch",
            key=KEY,
        )


# -- PiT -----------------------------------------------------------------------------------------


def test_pit_locality_100_does_not_mask():
    """At locality 100 every key is kept, so no percentile (a sort) is computed."""
    model = fx.pit(
        2, 1, hid_channels=8, n_head=2, localities=(100, 100, 100),
        input_res=(8, 8), latent_res=(4, 4), output_res=(8, 8), key=KEY,
    )  # fmt: skip
    jaxpr = str(jax.make_jaxpr(model)(_x(1, 64, 2)))
    assert "sort" not in jaxpr


# -- FFNO ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build, shape",
    [(lambda: fx.ffno2d(2, 8, n_modes=4, n_layers=1, key=KEY), (16, 16, 2)),
     (lambda: fx.ffno3d(2, 8, n_modes=4, n_layers=1, key=KEY), (8, 8, 8, 2))],
    ids=["ffno2d", "ffno3d"],
)  # fmt: skip
def test_ffno_keeps_double_precision(build, shape):
    with jax.enable_x64(True):
        model = build()
        jaxpr = str(jax.make_jaxpr(model)(jnp.ones(shape, jnp.float64)))
        assert model(jnp.ones(shape, jnp.float64)).dtype == jnp.float64
    assert "c64" not in jaxpr, "a complex64 array in a float64 model"


# -- MgNO ----------------------------------------------------------------------------------------


def test_mgno_padding_mode():
    x = _x(16, 16, 2)
    out = {
        m: fx.mgno2d(
            (16, 16),
            num_layer=1,
            num_channel_u=4,
            num_channel_f=2,
            padding_mode=m,
            key=KEY,
        )(x)
        for m in ("CIRCULAR", "zeros", "reflect")
    }
    assert float(jnp.abs(out["zeros"] - out["CIRCULAR"]).max()) > 1e-5
    assert float(jnp.abs(out["reflect"] - out["CIRCULAR"]).max()) > 1e-5
    with pytest.raises(ValueError, match="padding_mode"):
        fx.mgno2d(
            (16, 16),
            num_layer=1,
            num_channel_u=4,
            num_channel_f=2,
            padding_mode="spiral",
            key=KEY,
        )


# -- KAN convolutions ----------------------------------------------------------------------------


@pytest.mark.parametrize("basis", ["bspline", "bsrbf", "relu", "rbf", "chebyshev"])
def test_kan_conv2d_skip_policy_matches_conv1d_and_conv3d(basis):
    two = KANConv2d(2, 3, basis=basis, key=KEY)
    one = fx.kan.conv1d(2, 3, basis=basis, key=KEY)
    three = fx.kan.conv3d(2, 3, basis=basis, key=KEY)
    assert (two.kan.skip is None) == (one.kan.skip is None) == (three.kan.skip is None)


def test_kan_conv2d_takes_one_sample():
    conv = KANConv2d(2, 3, key=KEY)
    assert conv(_x(8, 8, 2)).shape == (8, 8, 3)
    with pytest.raises(ValueError, match="vmap"):
        conv(_x(4, 8, 8, 2))
