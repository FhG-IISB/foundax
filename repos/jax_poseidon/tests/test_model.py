"""Minimal forward-pass tests for jax_poseidon (no checkpoint required)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_platform_name", "cpu")

# poseidonT expects 128×128 inputs but ScOT handles mismatched sizes,
# so we pass 32×32 to keep the test fast.
INPUT_H = INPUT_W = 32
N_CHANNELS = 4


@pytest.fixture(scope="module")
def model():
    from jax_poseidon import poseidonT

    return poseidonT()


@pytest.fixture(scope="module")
def dummy_input():
    # (B, H, W, C) – channels-last, as expected by ScOT
    x = jnp.ones((1, INPUT_H, INPUT_W, N_CHANNELS), dtype=jnp.float32)
    t = jnp.array([0.1], dtype=jnp.float32)
    return x, t


@pytest.fixture(scope="module")
def params(model, dummy_input):
    x, t = dummy_input
    rng = jax.random.PRNGKey(0)
    return model.init(rng, pixel_values=x, time=t, deterministic=True)


def test_import():
    from jax_poseidon import ScOT, ScOTConfig, poseidonT, poseidonB, poseidonL


def test_init(params):
    leaves = jax.tree_util.tree_leaves(params)
    assert len(leaves) > 0


def test_param_count(params):
    n = sum(x.size for x in jax.tree_util.tree_leaves(params))
    assert n > 0


def test_forward_returns_output(model, dummy_input, params):
    x, t = dummy_input
    out = model.apply(params, pixel_values=x, time=t, deterministic=True)
    assert out.output is not None


def test_forward_shape(model, dummy_input, params):
    x, t = dummy_input
    out = model.apply(params, pixel_values=x, time=t, deterministic=True)
    pred = out.output
    # (B, H, W, C_out) – same spatial as input (model up/downsamples internally)
    B = x.shape[0]
    assert pred.ndim == 4
    assert pred.shape[0] == B
    assert pred.shape[-1] == N_CHANNELS


def test_forward_finite(model, dummy_input, params):
    x, t = dummy_input
    out = model.apply(params, pixel_values=x, time=t, deterministic=True)
    assert jnp.all(jnp.isfinite(out.output))
