"""Minimal forward-pass tests for jax_pdeformer2 (no checkpoint required)."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_platform_name", "cpu")

# The Conv2dFuncEncoderV3 applies 3×(stride-4 conv) so needs resolution=128
# to produce 2×2=4 spatial outputs matching num_branches=4 exactly.
_RESOLUTION = 128
_NUM_POINTS_FUNCTION = _RESOLUTION**2  # 16384
_NUM_SCALAR = 80
_NUM_FUNCTION = 6
_NUM_BRANCHES = 4
_NUM_POINTS = 10  # evaluation points – keep tiny for speed


def _mini_config():
    from jax_pdeformer2 import PDEFORMER_SMALL_CONFIG

    return PDEFORMER_SMALL_CONFIG


@pytest.fixture(scope="module")
def mini_model():
    from jax_pdeformer2 import create_pdeformer_from_config

    return create_pdeformer_from_config({"model": _mini_config()})


@pytest.fixture(scope="module")
def dummy_inputs():
    from jax_pdeformer2.utils import create_dummy_inputs

    return create_dummy_inputs(
        num_scalar=_NUM_SCALAR,
        num_function=_NUM_FUNCTION,
        num_branches=_NUM_BRANCHES,
        num_points_function=_NUM_POINTS_FUNCTION,
        num_points=_NUM_POINTS,
    )


@pytest.fixture(scope="module")
def mini_params(mini_model, dummy_inputs):
    rng = jax.random.PRNGKey(0)
    return mini_model.init(rng, **dummy_inputs)


def test_import():
    from jax_pdeformer2 import (
        PDEformer,
        create_pdeformer_from_config,
        PDEFORMER_SMALL_CONFIG,
        PDEFORMER_BASE_CONFIG,
        PDEFORMER_FAST_CONFIG,
    )

    assert PDEFORMER_SMALL_CONFIG is not None


def test_init(mini_params):
    leaves = jax.tree_util.tree_leaves(mini_params)
    assert len(leaves) > 0


def test_param_count(mini_params):
    n = sum(x.size for x in jax.tree_util.tree_leaves(mini_params))
    assert n > 0


def test_forward_shape(mini_model, dummy_inputs, mini_params):
    out = mini_model.apply(mini_params, **dummy_inputs)
    # Output: (n_graph, num_points, 1)
    assert out.ndim == 3
    assert out.shape[0] == 1  # n_graph
    assert out.shape[1] == _NUM_POINTS
    assert out.shape[2] == 1


def test_forward_finite(mini_model, dummy_inputs, mini_params):
    out = mini_model.apply(mini_params, **dummy_inputs)
    assert jnp.all(jnp.isfinite(out))
