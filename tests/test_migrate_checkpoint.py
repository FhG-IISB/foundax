"""``scripts/migrate_checkpoint.py`` reads 0.2 checkpoints, where the fixed arrays were leaves, into 0.3 models."""

import importlib.util
import pathlib
import pickle

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")

import foundax as fx  # noqa: E402
from foundax.architectures.common import Constant  # noqa: E402

from .test_buffers import BUILDS  # noqa: E402
from .test_factory_smoke import KEY  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "migrate_checkpoint",
    pathlib.Path(__file__).parents[1] / "scripts" / "migrate_checkpoint.py",
)
migrate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migrate)

WITH_CONSTANTS = [
    "fno2d_positions", "fno3d_positions", "deeponet_fourier", "transformer", "geofno_nodal", "pcno", "pit",
    "gaot_S", "sfno2d", "kan.kan", "kan.fast", "kan.relu", "kan.sinc", "kan.bsrbf", "kan.conv2d",
]  # fmt: skip
BUILDS = BUILDS | {
    "pit_m_dists": lambda k: fx.pit(
        2,
        1,
        hid_channels=8,
        n_head=2,
        localities=(100, 50, 100),
        m_dists=[jnp.ones((4, 6))] * 3,
        key=k,
    ),
}
WITH_CONSTANTS.append("pit_m_dists")


def _shift(tree, by):
    return jax.tree.map(lambda x: x + by if eqx.is_inexact_array(x) else x, tree)


@pytest.mark.parametrize("name", WITH_CONSTANTS)
def test_load_v02_drops_the_saved_constants(name, tmp_path):
    model = BUILDS[name](KEY)
    old, marks = migrate.legacy_view(model)
    flags = jax.tree.leaves(marks)
    assert any(flags), "the model has no constants"
    assert len(jax.tree.leaves(old)) == len(jax.tree.leaves(model)) + sum(flags)

    # a 0.2 file: parameters trained (+0.5) and the fixed arrays drifted (+0.01)
    leaves = [
        x + (0.01 if f else 0.5) if eqx.is_inexact_array(x) else x
        for x, f in zip(jax.tree.leaves(old), flags)
    ]
    path = tmp_path / "old.eqx"
    eqx.tree_serialise_leaves(path, jax.tree.unflatten(jax.tree.structure(old), leaves))

    with pytest.warns(UserWarning, match="fixed arrays differ"):
        loaded = migrate.load_v02(path, model)
    expected = _shift(model, 0.5)
    for got, want in zip(jax.tree.leaves(loaded), jax.tree.leaves(expected)):
        if eqx.is_array(want):
            np.testing.assert_array_equal(got, want)
    assert eqx.tree_equal(
        eqx.filter(loaded, eqx.is_array, inverse=True),
        eqx.filter(model, eqx.is_array, inverse=True),
    )


def test_cli_writes_a_03_checkpoint(tmp_path):
    expr = "fx.fno2d(2, 8, 4, n_layers=1, d_model=(16, 16), use_positions=True, key=jax.random.PRNGKey(0))"
    model = eval(expr, {"fx": fx, "jax": jax})
    old, _ = migrate.legacy_view(model)
    eqx.tree_serialise_leaves(tmp_path / "old.eqx", old)
    migrate.main(
        [str(tmp_path / "old.eqx"), str(tmp_path / "new.eqx"), "--model", expr]
    )
    again = eqx.tree_deserialise_leaves(tmp_path / "new.eqx", model)
    assert eqx.tree_equal(again, model)


def test_constant_is_hashable_by_content_and_read_only():
    a, b = (
        Constant(np.arange(6.0).reshape(2, 3)),
        Constant(np.arange(6.0).reshape(2, 3)),
    )
    assert a == b and hash(a) == hash(b) and a is not b
    assert a != Constant(np.arange(6.0)) and a != Constant(
        np.arange(6, dtype=np.float32).reshape(2, 3)
    )
    assert Constant(np.array([np.nan])) == Constant(np.array([np.nan]))
    with pytest.raises(ValueError):
        a.value[0, 0] = 1.0
    c = pickle.loads(pickle.dumps(a))
    assert c == a and not c.value.flags.writeable


def test_models_with_constants_jit_without_recompiling():
    """Equal constants hash equal, so a rebuilt model hits the jit cache."""
    traces = []

    @eqx.filter_jit
    def run(m, x):
        traces.append(1)
        return m(x)

    x = jnp.ones((8, 16, 2))
    for _ in range(2):
        run(BUILDS["sfno2d"](KEY), x)
    assert len(traces) == 1
