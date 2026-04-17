"""Tests for the MPP (AViT) Equinox model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "repos", "jax_mpp"))

from jax_mpp.avit_eqx import AViT as EqxAViT


def _make_config():
    return dict(
        patch_size=(16, 16),
        embed_dim=32,
        processor_blocks=2,
        n_states=3,
        drop_path=0.0,
        bias_type="rel",
        num_heads=2,
    )


@pytest.fixture
def model_and_input():
    cfg = _make_config()
    model = EqxAViT(**cfg, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(42), (2, 1, 2, 32, 32))
    labels = jnp.array([0, 1])
    bcs = jnp.zeros((1, 2), dtype=jnp.int32)
    return model, x, labels, bcs


def test_forward_runs(model_and_input):
    model, x, labels, bcs = model_and_input
    out = model(x, labels, bcs, deterministic=True)
    assert jnp.all(jnp.isfinite(out))


def test_output_shapes(model_and_input):
    model, x, labels, bcs = model_and_input
    out = model(x, labels, bcs, deterministic=True)
    T, B, C, H, W = x.shape
    assert out.shape == (B, C, H, W)


def test_gradients(model_and_input):
    model, x, labels, bcs = model_and_input

    @eqx.filter_grad
    def loss_fn(m):
        out = m(x, labels, bcs, deterministic=True)
        return jnp.mean(out**2)

    grads = loss_fn(model)
    flat = jax.tree_util.tree_leaves(grads)
    assert any(jnp.any(g != 0) for g in flat if eqx.is_array(g))


def test_zero_input(model_and_input):
    model, x, labels, bcs = model_and_input
    out = model(jnp.zeros_like(x), labels, bcs, deterministic=True)
    assert jnp.all(jnp.isfinite(out))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
