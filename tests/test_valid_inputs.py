"""Valid inputs and options that used to crash or return the wrong shape (B5)."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import foundax as fx  # noqa: E402

KEY = jax.random.PRNGKey(0)


def _x(*shape):
    return jax.random.normal(jax.random.PRNGKey(1), shape)


UNETS = {
    "unet1d": (lambda **kw: fx.unet1d(2, 1, depth=2, wf=3, key=KEY, **kw), (16, 2)),
    "unet2d": (lambda **kw: fx.unet2d(2, 1, depth=2, wf=3, key=KEY, **kw), (16, 16, 2)),
    "unet3d": (
        lambda **kw: fx.unet3d(2, 1, depth=2, wf=3, key=KEY, **kw),
        (8, 8, 8, 2),
    ),
}


@pytest.mark.parametrize("norm", [None, "layer", "instance", "group"])
@pytest.mark.parametrize("name", UNETS)
def test_unet_norms_on_channel_last_input(name, norm):
    build, shape = UNETS[name]
    y = build(norm=norm)(_x(*shape))
    assert y.shape == shape[:-1] + (1,)
    assert bool(jnp.all(jnp.isfinite(y)))


@pytest.mark.parametrize("name", UNETS)
def test_unet_batch_norm_raises(name):
    with pytest.raises(ValueError, match="instance"):
        UNETS[name][0](norm="batch")


@pytest.mark.parametrize(
    "name, shape", [("unet2d", (15, 13, 2)), ("unet3d", (9, 7, 11, 2))]
)
def test_unet_odd_sizes(name, shape):
    y = UNETS[name][0]()(_x(*shape))
    assert y.shape == shape[:-1] + (1,)


def test_unet1d_groups():
    assert fx.unet1d(2, 1, depth=2, wf=3, groups=2, key=KEY)(_x(16, 2)).shape == (16, 1)


def test_unet3d_default_activation_and_groups():
    assert fx.unet3d(2, 1, depth=2, wf=3, key=KEY)(_x(8, 8, 8, 2)).shape == (8, 8, 8, 1)
    y = fx.unet3d(2, 1, depth=2, wf=3, groups=2, key=KEY)(_x(8, 8, 8, 2))
    assert y.shape == (8, 8, 8, 1)
    with pytest.raises(ValueError, match="activation"):
        fx.unet3d(2, 1, depth=2, wf=3, activation="nope", key=KEY)


@pytest.mark.parametrize("shape", [(15, 15), (16, 15), (9, 12)])
def test_mgno2d_odd_grids(shape):
    model = fx.mgno2d(shape, num_layer=1, num_channel_u=4, num_channel_f=2, key=KEY)
    assert model(_x(*shape, 2)).shape == shape + (1,)


@pytest.mark.parametrize("n", [15, 16])
def test_mgno1d_odd_and_even_lengths(n):
    model = fx.mgno1d(n, num_layer=1, num_channel_u=4, num_channel_f=2, key=KEY)
    assert model(_x(n, 2)).shape == (n, 1)


def test_mgno1d_stays_one_dimensional():
    """MgNO1D runs MgNO on an (n, 1) grid; no intermediate may widen it to (n, n) by broadcasting."""
    n = 16
    model = fx.mgno1d(n, num_layer=1, num_channel_u=4, num_channel_f=2, key=KEY)
    jaxpr = jax.make_jaxpr(model)(_x(n, 2))
    widths = [
        v.aval.shape[2]
        for eqn in jaxpr.eqns
        for v in eqn.outvars
        if len(v.aval.shape) == 4
    ]
    # padding for the 3x3 kernels widens to 3 and the transpose conv gives 2; the bug broadcast to n
    assert widths and max(widths) <= 4, f"intermediates as wide as {max(widths)}"


def test_mlp_layer_norm_on_points():
    model = fx.mlp(3, 2, hidden_dims=8, layer_norm=True, key=KEY)
    pts = _x(5, 3)
    np.testing.assert_allclose(model(pts), jax.vmap(model)(pts), rtol=1e-5, atol=1e-6)


def test_mlp_batch_norm_raises():
    with pytest.raises(ValueError, match="layer_norm"):
        fx.mlp(3, 2, hidden_dims=8, batch_norm=True, key=KEY)


@pytest.mark.parametrize("n", [1, 7])
def test_geofno_nodal_keeps_the_channel_axis(n):
    model = fx.geofno_nodal(2, [4, 4], [1.0, 1.0], layers=(8, 8), fc_dim=8, key=KEY)
    args = (
        _x(n, 3),
        jnp.ones((n, 1)),
        jax.random.uniform(KEY, (n, 2)),
        jnp.full((n, 1), 1.0 / n),
    )
    assert model(*args).shape == (n, 1)


def test_cno_one_sample_in_one_sample_out():
    model = fx.cno2d(
        2, 1, size=16, N_layers=1, N_res=1, N_res_neck=1, channel_multiplier=4, key=KEY
    )
    x = _x(16, 16, 2)
    assert model(x).shape == (16, 16, 1)
    assert model(x[None]).shape == (1, 16, 16, 1)  # a batch stays a batch


@pytest.mark.parametrize(
    "build, shape",
    [
        (
            lambda: fx.fno2d(
                2, 8, 4, n_layers=1, d_model=(16, 16), use_positions=True, key=KEY
            ),
            (16, 16, 2),
        ),
        (
            lambda: fx.fno3d(
                2, 8, 4, n_layers=1, d_model=(8, 8, 8), use_positions=True, key=KEY
            ),
            (8, 8, 8, 2),
        ),
    ],
    ids=["fno2d", "fno3d"],
)
def test_fno_with_positions(build, shape):
    """``in_features`` counts the data channels; the lift adds the coordinate channels."""
    assert build()(_x(*shape)).shape == shape[:-1] + (1,)
