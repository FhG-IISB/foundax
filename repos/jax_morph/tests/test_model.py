"""Minimal forward-pass tests for jax_morph (no checkpoint required)."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_platform_name", "cpu")


@pytest.fixture(scope="module")
def tiny_model():
    """ViT3DRegression with barely-enough dims to run quickly on CPU."""
    from jax_morph import ViT3DRegression

    return ViT3DRegression(
        patch_size=8,
        dim=32,
        depth=1,
        heads=2,
        heads_xa=2,
        mlp_dim=64,
        max_components=1,
        conv_filter=4,
        max_ar=1,
        max_patches=64,
        max_fields=1,
        dropout=0.0,
        emb_dropout=0.0,
        model_size="Ti",
    )


@pytest.fixture(scope="module")
def tiny_input():
    # (B, t, F, C, D, H, W) – one sample, one timestep, one field, one component
    return jnp.ones((1, 1, 1, 1, 8, 8, 8))


@pytest.fixture(scope="module")
def tiny_params(tiny_model, tiny_input):
    rng = jax.random.PRNGKey(0)
    return tiny_model.init(rng, tiny_input, deterministic=True)


def test_import():
    from jax_morph import ViT3DRegression, morph_Ti, morph_S, MORPH_CONFIGS

    assert "Ti" in MORPH_CONFIGS


def test_init(tiny_params):
    leaves = jax.tree_util.tree_leaves(tiny_params)
    assert len(leaves) > 0


def test_param_count(tiny_params):
    n = sum(x.size for x in jax.tree_util.tree_leaves(tiny_params))
    assert n > 0


def test_forward_shape(tiny_model, tiny_input, tiny_params):
    enc, z, pred = tiny_model.apply(tiny_params, tiny_input, deterministic=True)
    B, F, C, D, H, W = 1, 1, 1, 8, 8, 8
    assert pred.shape == (B, F, C, D, H, W)


def test_forward_finite(tiny_model, tiny_input, tiny_params):
    _, _, pred = tiny_model.apply(tiny_params, tiny_input, deterministic=True)
    assert jnp.all(jnp.isfinite(pred))


def test_convenience_constructors():
    from jax_morph import morph_Ti, morph_S

    assert morph_Ti() is not None
    assert morph_S() is not None
