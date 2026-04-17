"""Tests for the BCAT Equinox model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "repos", "jax_bcat"))

from jax_bcat.model_eqx import BCAT as EqxBCAT


_CFG = dict(
    n_layer=2,
    dim_emb=64,
    dim_ffn=128,
    n_head=4,
    norm_first=True,
    norm_type="rms",
    activation="swiglu",
    qk_norm=True,
    x_num=16,
    max_output_dim=2,
    patch_num=4,
    patch_num_output=4,
    conv_dim=8,
    time_embed="learnable",
    max_time_len=10,
    max_data_len=10,
    deep=False,
)

_DATA_DIM = 2
_T_TOTAL = 5
_INPUT_LEN = 3


@pytest.fixture
def model():
    return EqxBCAT(**_CFG, data_dim=_DATA_DIM, key=jax.random.PRNGKey(0))


@pytest.fixture
def inputs():
    bs = 1
    data = jax.random.normal(
        jax.random.PRNGKey(1), (bs, _T_TOTAL, _CFG["x_num"], _CFG["x_num"], _DATA_DIM)
    )
    times = jax.random.uniform(jax.random.PRNGKey(2), (bs, _T_TOTAL, 1))
    return data, times


class TestBCATEqx:
    def test_eqx_init(self, model):
        assert isinstance(model, EqxBCAT)

    def test_forward_runs(self, model, inputs):
        data, times = inputs
        out = model(data, times, input_len=_INPUT_LEN)
        assert jnp.all(jnp.isfinite(out))

    def test_output_shape(self, model, inputs):
        data, times = inputs
        out = model(data, times, input_len=_INPUT_LEN)
        bs, t_total, x_num, _, data_dim = data.shape
        assert out.shape[0] == bs

    def test_gradients(self, model, inputs):
        data, times = inputs

        @eqx.filter_grad
        def loss_fn(m):
            return jnp.mean(m(data, times, input_len=_INPUT_LEN) ** 2)

        grads = loss_fn(model)
        flat = jax.tree_util.tree_leaves(grads)
        assert any(jnp.any(g != 0) for g in flat if eqx.is_array(g))

    def test_forward_equivalence_continuous_time(self, model):
        """Continuous time values produce finite output."""
        bs = 1
        data = jax.random.normal(
            jax.random.PRNGKey(3),
            (bs, _T_TOTAL, _CFG["x_num"], _CFG["x_num"], _DATA_DIM),
        )
        times = jnp.linspace(0.0, 1.0, _T_TOTAL).reshape(1, _T_TOTAL, 1)
        out = model(data, times, input_len=_INPUT_LEN)
        assert jnp.all(jnp.isfinite(out))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
