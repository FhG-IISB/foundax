"""Tests for the Equinox ScOT (Poseidon) model.

Run with:
    python -m pytest tests/test_poseidon_eqx.py -v
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

import sys, os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "repos", "jax_poseidon")
)

from jax_poseidon.scot import ScOTConfig
from jax_poseidon.scot_eqx import ScOT as EqxScOT


def _make_config(**overrides):
    """Create a small test config."""
    defaults = dict(
        name="test",
        image_size=32,
        patch_size=4,
        num_channels=2,
        num_out_channels=2,
        embed_dim=24,
        depths=(2, 2),
        num_heads=(3, 6),
        skip_connections=(1, 0),
        window_size=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.0,
        hidden_act="gelu",
        use_absolute_embeddings=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        p=1,
        channel_slice_list_normalized_loss=None,
        residual_model="convnext",
        use_conditioning=True,
        learn_residual=False,
        pretrained_window_sizes=(0, 0),
    )
    defaults.update(overrides)
    return ScOTConfig(**defaults)


# =====================================================================
# Tests
# =====================================================================


class TestPoseidonEquivalence:
    """Tests for Equinox ScOT with time conditioning."""

    @pytest.fixture
    def config(self):
        return _make_config()

    @pytest.fixture
    def model(self, config):
        return EqxScOT(
            config=config,
            use_conditioning=config.use_conditioning,
            key=jax.random.PRNGKey(0),
        )

    def test_eqx_init(self, model):
        assert isinstance(model, EqxScOT)

    def test_forward_runs(self, config, model):
        x = jax.random.normal(
            jax.random.PRNGKey(1),
            (1, config.image_size, config.image_size, config.num_channels),
        )
        t = jnp.array([0.5])
        out = model(pixel_values=x, time=t, deterministic=True, return_dict=False)
        assert jnp.all(jnp.isfinite(out[0]))

    def test_output_shapes(self, config, model):
        B = 2
        x = jnp.ones((B, config.image_size, config.image_size, config.num_channels))
        t = jnp.array([0.1, 0.2])
        out = model(pixel_values=x, time=t, deterministic=True, return_dict=False)
        assert out[0].shape == (
            B,
            config.image_size,
            config.image_size,
            config.num_out_channels,
        )

    def test_zero_input(self, config, model):
        x = jnp.zeros((1, config.image_size, config.image_size, config.num_channels))
        t = jnp.zeros((1,))
        out = model(pixel_values=x, time=t, deterministic=True, return_dict=False)
        assert jnp.all(jnp.isfinite(out[0]))

    def test_gradients(self, config, model):
        x = jax.random.normal(
            jax.random.PRNGKey(99),
            (1, config.image_size, config.image_size, config.num_channels),
        )
        t = jnp.array([0.5])

        @eqx.filter_grad
        def loss_fn(m):
            out = m(pixel_values=x, time=t, deterministic=True, return_dict=False)
            return jnp.mean(out[0] ** 2)

        grads = loss_fn(model)
        for leaf in jax.tree_util.tree_leaves(grads):
            if hasattr(leaf, "shape"):
                assert jnp.all(jnp.isfinite(leaf))


class TestPoseidonNoConditioning:
    """Tests for Equinox ScOT without conditioning."""

    @pytest.fixture
    def config(self):
        return _make_config(use_conditioning=False)

    @pytest.fixture
    def model(self, config):
        return EqxScOT(config=config, use_conditioning=False, key=jax.random.PRNGKey(0))

    def test_eqx_init(self, model):
        assert isinstance(model, EqxScOT)

    def test_forward_no_cond(self, config, model):
        x = jax.random.normal(
            jax.random.PRNGKey(7),
            (1, config.image_size, config.image_size, config.num_channels),
        )
        out = model(pixel_values=x, deterministic=True, return_dict=False)
        assert jnp.all(jnp.isfinite(out[0]))

    def test_output_shapes_no_cond(self, config, model):
        x = jnp.ones((1, config.image_size, config.image_size, config.num_channels))
        out = model(pixel_values=x, deterministic=True, return_dict=False)
        assert out[0].shape == (
            1,
            config.image_size,
            config.image_size,
            config.num_out_channels,
        )

    def test_gradients_no_cond(self, config, model):
        x = jax.random.normal(
            jax.random.PRNGKey(3),
            (1, config.image_size, config.image_size, config.num_channels),
        )

        @eqx.filter_grad
        def loss_fn(m):
            out = m(pixel_values=x, deterministic=True, return_dict=False)
            return jnp.mean(out[0] ** 2)

        grads = loss_fn(model)
        for leaf in jax.tree_util.tree_leaves(grads):
            if hasattr(leaf, "shape"):
                assert jnp.all(jnp.isfinite(leaf))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
