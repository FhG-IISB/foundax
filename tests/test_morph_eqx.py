"""Tests for the MORPH Equinox model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "repos", "jax_morph"))

from jax_morph.model_eqx import ViT3DRegression as EqxViT3D


def _make_config():
    return dict(
        patch_size=4,
        dim=32,
        depth=2,
        heads=2,
        heads_xa=2,
        mlp_dim=64,
        max_components=2,
        conv_filter=8,
        max_ar=1,
        max_patches=64,
        max_fields=2,
        model_size="Ti",
        dropout=0.0,
        emb_dropout=0.0,
    )


@pytest.fixture
def model_and_input():
    cfg = _make_config()
    model = EqxViT3D(**cfg, key=jax.random.PRNGKey(0))
    vol = jax.random.normal(jax.random.PRNGKey(42), (1, 1, 1, 1, 4, 4, 4))
    return model, vol


def test_forward_runs(model_and_input):
    model, vol = model_and_input
    enc, z, out = model(vol)
    assert jnp.all(jnp.isfinite(out))


def test_output_shapes(model_and_input):
    model, vol = model_and_input
    enc, z, out = model(vol)
    B, t, F, C, D, H, W = vol.shape
    assert enc.shape[0] == B and enc.shape[-1] == model.dim
    assert z.shape == enc.shape
    assert out.shape == (B, F, C, D, H, W)


def test_gradients(model_and_input):
    model, vol = model_and_input

    @eqx.filter_grad
    def loss_fn(m):
        _, _, out = m(vol)
        return jnp.mean(out**2)

    grads = loss_fn(model)
    flat = jax.tree_util.tree_leaves(grads)
    assert any(jnp.any(g != 0) for g in flat if eqx.is_array(g))


def test_zero_input(model_and_input):
    model, vol = model_and_input
    zero_vol = jnp.zeros_like(vol)
    _, _, out = model(zero_vol)
    assert jnp.all(jnp.isfinite(out))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
