"""Tests for the PDEformer-2 Equinox model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

import sys, os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "repos", "jax_pdeformer2")
)

from jax_pdeformer2.model_eqx import PDEformer as EqxPDEformer


# ---------------------------------------------------------------------------
# Tiny test config
# ---------------------------------------------------------------------------
_CFG = {
    "graphormer": {
        "num_node_type": 8,
        "num_in_degree": 4,
        "num_out_degree": 4,
        "num_spatial": 4,
        "num_encoder_layers": 2,
        "embed_dim": 32,
        "ffn_embed_dim": 64,
        "num_heads": 4,
        "pre_layernorm": True,
    },
    "scalar_encoder": {"dim_hidden": 16, "num_layers": 2},
    "function_encoder": {
        "type": "cnn2dv3",
        "num_branches": 1,
        "resolution": 64,
        "conv2d_input_txyz": False,
        "cnn_keep_nchw": True,
    },
    "inr": {
        "type": "poly_inr",
        "num_layers": 3,
        "dim_hidden": 16,
        "poly_inr": {
            "enable_affine": False,
            "enable_shift": True,
            "enable_scale": True,
            "activation_fn": "sin",
            "affine_act_fn": "identity",
        },
    },
    "hypernet": {"dim_hidden": 16, "num_layers": 2, "shared": False},
    "multi_inr": {"enable": False},
}

NUM_SCALAR = 4
NUM_FUNCTION = 2
NUM_BRANCHES = 1
N_NODE = NUM_SCALAR + NUM_FUNCTION * NUM_BRANCHES
NUM_POINTS_FUNC = 64 * 64
NUM_POINTS = 10


def _make_model():
    gc = _CFG["graphormer"]
    sc = _CFG["scalar_encoder"]
    fc = _CFG["function_encoder"]
    ic = _CFG["inr"]
    pc = ic["poly_inr"]
    hc = _CFG["hypernet"]
    mc = _CFG["multi_inr"]
    return EqxPDEformer(
        num_node_type=gc["num_node_type"],
        num_in_degree=gc["num_in_degree"],
        num_out_degree=gc["num_out_degree"],
        num_spatial=gc["num_spatial"],
        num_encoder_layers=gc["num_encoder_layers"],
        embed_dim=gc["embed_dim"],
        ffn_embed_dim=gc["ffn_embed_dim"],
        num_heads=gc["num_heads"],
        pre_layernorm=gc.get("pre_layernorm", True),
        scalar_dim_hidden=sc["dim_hidden"],
        scalar_num_layers=sc["num_layers"],
        func_enc_resolution=fc["resolution"],
        func_enc_input_txyz=fc.get("conv2d_input_txyz", False),
        func_enc_keep_nchw=fc.get("cnn_keep_nchw", True),
        inr_dim_hidden=ic["dim_hidden"],
        inr_num_layers=ic["num_layers"],
        enable_affine=pc.get("enable_affine", False),
        enable_shift=pc.get("enable_shift", True),
        enable_scale=pc.get("enable_scale", True),
        activation_fn=pc.get("activation_fn", "sin"),
        affine_act_fn=pc.get("affine_act_fn", "identity"),
        hyper_dim_hidden=hc["dim_hidden"],
        hyper_num_layers=hc["num_layers"],
        share_hypernet=hc.get("shared", False),
        multi_inr=mc.get("enable", False),
        separate_latent=mc.get("separate_latent", False),
        key=jax.random.PRNGKey(0),
    )


def _make_inputs(key):
    keys = jax.random.split(key, 8)
    n_graph = 1
    return dict(
        node_type=jax.random.randint(
            keys[0], (n_graph, N_NODE, 1), 1, 8, dtype=jnp.int32
        ),
        node_scalar=jax.random.normal(keys[1], (n_graph, NUM_SCALAR, 1)),
        node_function=jax.random.normal(
            keys[2], (n_graph, NUM_FUNCTION, NUM_POINTS_FUNC, 5)
        ),
        in_degree=jax.random.randint(keys[3], (n_graph, N_NODE), 0, 4, dtype=jnp.int32),
        out_degree=jax.random.randint(
            keys[4], (n_graph, N_NODE), 0, 4, dtype=jnp.int32
        ),
        attn_bias=jax.random.normal(keys[5], (n_graph, N_NODE, N_NODE)),
        spatial_pos=jax.random.randint(
            keys[6], (n_graph, N_NODE, N_NODE), 0, 4, dtype=jnp.int32
        ),
        coordinate=jax.random.uniform(keys[7], (n_graph, NUM_POINTS, 4)),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture
def model():
    return _make_model()


@pytest.fixture
def inputs(rng):
    return _make_inputs(rng)


class TestPDEformerEqx:
    def test_eqx_init(self, model):
        assert isinstance(model, EqxPDEformer)

    def test_forward_runs(self, model, inputs):
        out = model(**inputs)
        assert jnp.all(jnp.isfinite(out))

    def test_output_shape(self, model, inputs):
        out = model(**inputs)
        assert out.shape[0] == 1  # n_graph

    def test_gradients(self, model, inputs):
        @eqx.filter_grad
        def loss(m):
            return jnp.mean(m(**inputs) ** 2)

        grads = loss(model)
        flat = jax.tree_util.tree_leaves(grads)
        assert any(jnp.any(g != 0) for g in flat if eqx.is_array(g))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
