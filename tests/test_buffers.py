"""The float arrays of every core model are trainable parameters, and nothing else.

Training code picks the parameters with ``eqx.partition(model, eqx.is_inexact_array)`` (jNO does), so a fixed array
stored as a float leaf (a coordinate grid, KAN knots, a Legendre table, a sinusoidal encoding) is trained and
weight-decayed. ``jax.lax.stop_gradient`` in the forward pass does not prevent that: weight decay still moves it.

Each model has an explicit list of the leaf names that are parameters (the last attribute name on the path). A
leaf with any other name fails the test, so a new parameter name is added here on purpose, and a buffer is not.
"""

import pytest

jax = pytest.importorskip("jax")
eqx = pytest.importorskip("equinox")

import foundax as fx  # noqa: E402

from .test_factory_smoke import CORE, KAN, KEY  # noqa: E402

LINEAR = {"weight", "bias"}
SPECTRAL_1D = {"weight_real", "weight_imag"}
SPECTRAL_2D = {"weight_1_real", "weight_1_imag", "weight_2_real", "weight_2_imag"}
SPECTRAL_3D = SPECTRAL_2D | {
    "weight_3_real",
    "weight_3_imag",
    "weight_4_real",
    "weight_4_imag",
}
NODAL = {"weights_c", "weights_s", "weights_0"}
KAN_NET = {"spline_weight"}

BUILDS = {name: case[0] for name, case in CORE.items()}
BUILDS |= {"kan." + name: case[0] for name, case in KAN.items()}
# configurations that store a fixed array the defaults above do not
BUILDS |= {
    "fno2d_positions": lambda k: fx.fno2d(
        2, 8, 4, n_layers=1, d_model=(16, 16), use_positions=True, key=k
    ),
    "fno3d_positions": lambda k: fx.fno3d(
        2, 8, 4, n_layers=1, d_model=(8, 8, 8), use_positions=True, key=k
    ),
    "deeponet_fourier": lambda k: fx.deeponet(
        n_sensors=10,
        basis_functions=8,
        hidden_dim=16,
        n_layers=2,
        coord_embedding="fourier",
        key=k,
    ),
}

PARAMS = {name: LINEAR for name in BUILDS}
PARAMS |= {
    "fno1d": LINEAR | SPECTRAL_1D,
    "fno2d": LINEAR | SPECTRAL_2D,
    "fno2d_positions": LINEAR | SPECTRAL_2D,
    "fno3d": LINEAR | SPECTRAL_3D,
    "fno3d_positions": LINEAR | SPECTRAL_3D,
    "deeponet": LINEAR | {"output_bias"},
    "deeponet_fourier": LINEAR | {"output_bias"},
    "geofno": LINEAR
    | {"weights1_real", "weights1_imag", "weights2_real", "weights2_imag"},
    "geofno_nodal": LINEAR | NODAL,
    "pcno": LINEAR
    | NODAL
    | {"inv_L_scale_latent"},  # the learned length scale (train_inv_L_scale=True)
    "pit": LINEAR | {"r"},  # the learned locality of each head
    "ffno2d": LINEAR
    | {"weight_x_real", "weight_x_imag", "weight_y_real", "weight_y_imag"},
    "ffno3d": LINEAR | {f"weight_{a}_{p}" for a in "dhw" for p in ("real", "imag")},
    "transolver": LINEAR | {"temperature", "placeholder"},
    "transolver2d": LINEAR | {"temperature", "placeholder"},
    "transolver3d": LINEAR | {"temperature", "placeholder"},
    "sfno2d": LINEAR | SPECTRAL_1D,
}
PARAMS |= {"kan." + n: KAN_NET for n in KAN}
PARAMS |= {
    "kan." + n: KAN_NET | LINEAR
    for n in ("kan", "default", "efficient", "fast", "bsrbf", "res_block")
}
PARAMS |= {
    "kan." + n: KAN_NET | LINEAR
    for n in ("conv1d", "conv2d", "conv3d", "attention_block")
}
PARAMS["kan.relu"] = (
    KAN_NET | LINEAR | {"grid_a", "grid_b"}
)  # ReLU-KAN trains its phases
PARAMS |= {"kan.spectral_block1d": KAN_NET | SPECTRAL_1D}
PARAMS |= {
    "kan.spectral_block2d": KAN_NET | SPECTRAL_2D,
    "kan.spectral_block3d": KAN_NET | SPECTRAL_3D,
}


def _leaf_name(path):
    """The last attribute name on a pytree path (list and dict indices are skipped)."""
    for entry in reversed(path):
        if isinstance(entry, jax.tree_util.GetAttrKey):
            return entry.name
    return jax.tree_util.keystr(path)


@pytest.mark.parametrize("name", list(BUILDS))
def test_float_leaves_are_parameters(name):
    model = BUILDS[name](KEY)
    leaves = jax.tree_util.tree_flatten_with_path(
        eqx.filter(model, eqx.is_inexact_array)
    )[0]
    extra = sorted(
        {
            jax.tree_util.keystr(p)
            for p, _ in leaves
            if _leaf_name(p) not in PARAMS[name]
        }
    )
    assert not extra, f"float leaves that are not parameters: {extra}"
