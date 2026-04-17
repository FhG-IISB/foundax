"""Tests for the DPOT (DPOTNet) Equinox model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "repos", "jax_dpot"))

from jax_dpot.model_eqx import DPOTNet as EqxDPOTNet


_KWARGS = dict(
    img_size=32,
    patch_size=4,
    mixing_type="afno",
    in_channels=2,
    out_channels=2,
    in_timesteps=3,
    out_timesteps=1,
    n_blocks=4,
    embed_dim=32,
    out_layer_dim=16,
    depth=2,
    modes=8,
    mlp_ratio=1.0,
    n_cls=4,
    normalize=False,
    act="gelu",
    time_agg="exp_mlp",
)

_KWARGS_NORM = dict(_KWARGS, normalize=True)


@pytest.fixture
def model():
    return EqxDPOTNet(**_KWARGS, key=jax.random.PRNGKey(0))


@pytest.fixture
def x():
    return jnp.zeros((1, 32, 32, 3, 2))


class TestDPOTEquivalence:
    def test_forward_runs(self, model, x):
        e_pred, e_cls = model(x)
        assert jnp.all(jnp.isfinite(e_pred))
        assert jnp.all(jnp.isfinite(e_cls))

    def test_output_shapes(self, model, x):
        e_pred, e_cls = model(x)
        assert e_pred.shape[0] == x.shape[0]
        assert e_cls.shape[0] == x.shape[0]

    def test_gradient_flows(self, model, x):
        @eqx.filter_grad
        def loss(m):
            pred, cls = m(x)
            return jnp.mean(pred**2) + jnp.mean(cls**2)

        grads = loss(model)
        flat = jax.tree_util.tree_leaves(grads)
        assert any(jnp.any(g != 0) for g in flat if eqx.is_array(g))


class TestDPOTWithNormalize:
    @pytest.fixture
    def model_norm(self):
        return EqxDPOTNet(**_KWARGS_NORM, key=jax.random.PRNGKey(0))

    def test_forward_with_normalize(self, model_norm, x):
        e_pred, e_cls = model_norm(x)
        assert jnp.all(jnp.isfinite(e_pred))
        assert jnp.all(jnp.isfinite(e_cls))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
