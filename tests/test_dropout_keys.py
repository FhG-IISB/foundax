"""Dropout draws an independent mask for every layer, sublayer, query point and sample (B6)."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")

import foundax as fx  # noqa: E402

KEY = jax.random.PRNGKey(0)
DROP = jax.random.PRNGKey(7)


@pytest.fixture
def dropout_keys(monkeypatch):
    """Record the key of every eager ``eqx.nn.Dropout`` call."""
    seen = []
    original = eqx.nn.Dropout.__call__

    def recording(self, x, *args, key=None, **kwargs):
        if key is not None and not isinstance(key, jax.core.Tracer):
            seen.append(np.asarray(key).tobytes())
        return original(self, x, *args, key=key, **kwargs)

    monkeypatch.setattr(eqx.nn.Dropout, "__call__", recording)
    return seen


GNOT_FAMILY = {
    "cgptno": (
        lambda: fx.cgptno(
            2, [3], n_layers=3, n_hidden=8, ffn_dropout=0.5, attn_dropout=0.5, key=KEY
        ),
        lambda: (
            jax.random.normal(KEY, (1, 5, 2)),
            [jax.random.normal(KEY, (1, 7, 3))],
        ),
    ),
    "gnot": (
        lambda: fx.gnot(
            2, [3], n_layers=3, n_hidden=8, ffn_dropout=0.5, attn_dropout=0.5, key=KEY
        ),
        lambda: (
            jax.random.normal(KEY, (1, 5, 2)),
            [jax.random.normal(KEY, (1, 7, 3))],
        ),
    ),
    "moegptno": (
        lambda: fx.moegptno(
            2, 3, n_layers=3, n_hidden=8, ffn_dropout=0.5, attn_dropout=0.5, key=KEY
        ),
        lambda: (jax.random.normal(KEY, (1, 5, 2)), jax.random.normal(KEY, (1, 7, 3))),
    ),
}


@pytest.mark.parametrize("name", GNOT_FAMILY)
def test_gnot_family_uses_a_new_key_for_every_dropout(name, dropout_keys):
    build, inputs = GNOT_FAMILY[name]
    build()(*inputs(), key=DROP)
    # 3 blocks x (cross-attention, its residual, self-attention, its residual)
    assert len(dropout_keys) == 12
    assert len(set(dropout_keys)) == len(dropout_keys), "two dropout layers share a key"


def _deeponet(**kw):
    return fx.deeponet(
        n_sensors=10,
        basis_functions=8,
        hidden_dim=16,
        n_layers=2,
        dropout_rate=0.5,
        key=KEY,
        **kw,
    )


def test_deeponet_masks_differ_between_query_points():
    """Identical query points get different dropout masks, so different outputs; without a key, the same."""
    model = _deeponet()
    u = jax.random.normal(KEY, (10,))
    y = jnp.full((6, 1), 0.3)
    assert float(jnp.std(model(u, y))) < 1e-6  # rounding: vectorised lanes may differ
    assert float(jnp.std(model(u, y, key=DROP))) > 1e-4


def test_deeponet_masks_differ_between_samples():
    model = _deeponet()
    u = jnp.broadcast_to(jax.random.normal(KEY, (10,)), (4, 10))
    y = jax.random.uniform(KEY, (3, 1))
    out = model.evaluate_batch(u, y, key=DROP)
    assert float(jnp.abs(out - out[0]).max()) > 1e-4


def test_deeponet_transformer_branch_applies_its_dropout(dropout_keys):
    model = _deeponet(branch_type="transformer", n_heads=2)
    u, y = jax.random.normal(KEY, (10,)), jax.random.uniform(KEY, (3, 1))
    a, b = model(u, y), model(u, y, key=DROP)
    assert float(jnp.abs(a - b).max()) > 1e-4
    # 2 blocks x (attention weights, two residuals)
    assert len(dropout_keys) == 6
    assert len(dropout_keys) == len(set(dropout_keys)), "two dropout layers share a key"
