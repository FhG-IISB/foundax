"""Tests verifying Equinox PDEformer-2 matches Flax PDEformer-2 forward pass."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from jax_pdeformer2.pdeformer import (
    PDEformer as FlaxPDEformer,
    create_pdeformer_from_config,
)
from jax_pdeformer2.model_eqx import PDEformer as EqxPDEformer

# ---------------------------------------------------------------------------
# Tiny test config (quick forward pass)
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
    "scalar_encoder": {
        "dim_hidden": 16,
        "num_layers": 2,
    },
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
    "hypernet": {
        "dim_hidden": 16,
        "num_layers": 2,
        "shared": False,
    },
    "multi_inr": {
        "enable": False,
    },
}

NUM_SCALAR = 4
NUM_FUNCTION = 2
NUM_BRANCHES = 1  # (64/64)^2 = 1
N_NODE = NUM_SCALAR + NUM_FUNCTION * NUM_BRANCHES  # 6
NUM_POINTS_FUNC = 64 * 64  # 4096
NUM_POINTS = 10


# ---------------------------------------------------------------------------
# Weight transfer helpers
# ---------------------------------------------------------------------------


def _transfer_dense(flax_params, eqx_linear):
    new = eqx.tree_at(lambda m: m.weight, eqx_linear, flax_params["kernel"].T)
    if "bias" in flax_params and eqx_linear.bias is not None:
        new = eqx.tree_at(lambda m: m.bias, new, flax_params["bias"])
    return new


def _transfer_conv(flax_params, eqx_conv):
    """Flax Conv kernel (kH,kW,Cin,Cout) → eqx Conv2d weight (Cout,Cin,kH,kW)."""
    new = eqx.tree_at(
        lambda m: m.weight, eqx_conv, flax_params["kernel"].transpose(3, 2, 0, 1)
    )
    if "bias" in flax_params and eqx_conv.bias is not None:
        new = eqx.tree_at(lambda m: m.bias, new, flax_params["bias"].reshape(-1, 1, 1))
    return new


def _transfer_embedding(flax_params, eqx_emb):
    return eqx.tree_at(lambda m: m.weight, eqx_emb, flax_params["embedding"])


def _transfer_layernorm(flax_params, eqx_ln):
    new = eqx.tree_at(lambda m: m.weight, eqx_ln, flax_params["scale"])
    if "bias" in flax_params:
        new = eqx.tree_at(lambda m: m.bias, new, flax_params["bias"])
    return new


def _transfer_mlp(flax_params, eqx_mlp):
    """Transfer Flax MLP (Dense_0, Dense_1, ...) → eqx MLP (list of Linear)."""
    new = eqx_mlp
    for i, layer in enumerate(eqx_mlp.layers):
        key = f"Dense_{i}"
        new = eqx.tree_at(
            lambda m, idx=i: m.layers[idx],
            new,
            _transfer_dense(flax_params[key], layer),
        )
    return new


def _transfer_conv2d_func_encoder(flax_params, eqx_encoder):
    new = eqx_encoder
    for name, attr in [("conv1", "conv1"), ("conv2", "conv2"), ("conv3", "conv3")]:
        new = eqx.tree_at(
            lambda m, a=attr: getattr(m, a),
            new,
            _transfer_conv(flax_params[name], getattr(eqx_encoder, attr)),
        )
    return new


def _transfer_graph_node_feature(flax_params, eqx_gnf):
    new = eqx_gnf
    for name in ("node_encoder", "in_degree_encoder", "out_degree_encoder"):
        new = eqx.tree_at(
            lambda m, n=name: getattr(m, n),
            new,
            _transfer_embedding(flax_params[name], getattr(eqx_gnf, name)),
        )
    return new


def _transfer_graph_attn_bias(flax_params, eqx_gab):
    new = eqx_gab
    for name in ("spatial_pos_encoder", "spatial_pos_encoder_rev"):
        new = eqx.tree_at(
            lambda m, n=name: getattr(m, n),
            new,
            _transfer_embedding(flax_params[name], getattr(eqx_gab, name)),
        )
    return new


def _transfer_mha(flax_params, eqx_mha):
    new = eqx_mha
    for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
        new = eqx.tree_at(
            lambda m, p=proj: getattr(m, p),
            new,
            _transfer_dense(flax_params[proj], getattr(eqx_mha, proj)),
        )
    return new


def _transfer_graphormer_layer(flax_params, eqx_layer):
    new = eqx_layer
    new = eqx.tree_at(
        lambda m: m.multihead_attn,
        new,
        _transfer_mha(flax_params["multihead_attn"], eqx_layer.multihead_attn),
    )
    new = eqx.tree_at(
        lambda m: m.fc1,
        new,
        _transfer_dense(flax_params["fc1"], eqx_layer.fc1),
    )
    new = eqx.tree_at(
        lambda m: m.fc2,
        new,
        _transfer_dense(flax_params["fc2"], eqx_layer.fc2),
    )
    new = eqx.tree_at(
        lambda m: m.attn_layer_norm,
        new,
        _transfer_layernorm(flax_params["attn_layer_norm"], eqx_layer.attn_layer_norm),
    )
    new = eqx.tree_at(
        lambda m: m.ffn_layer_norm,
        new,
        _transfer_layernorm(flax_params["ffn_layer_norm"], eqx_layer.ffn_layer_norm),
    )
    return new


def _transfer_graphormer_encoder(flax_params, eqx_enc):
    new = eqx_enc
    new = eqx.tree_at(
        lambda m: m.graph_node_feature,
        new,
        _transfer_graph_node_feature(
            flax_params["graph_node_feature"], eqx_enc.graph_node_feature
        ),
    )
    new = eqx.tree_at(
        lambda m: m.graph_attn_bias,
        new,
        _transfer_graph_attn_bias(
            flax_params["graph_attn_bias"], eqx_enc.graph_attn_bias
        ),
    )
    for i, layer in enumerate(eqx_enc.layers):
        new = eqx.tree_at(
            lambda m, idx=i: m.layers[idx],
            new,
            _transfer_graphormer_layer(flax_params[f"layers_{i}"], layer),
        )
    if eqx_enc.emb_layer_norm is not None and "emb_layer_norm" in flax_params:
        new = eqx.tree_at(
            lambda m: m.emb_layer_norm,
            new,
            _transfer_layernorm(flax_params["emb_layer_norm"], eqx_enc.emb_layer_norm),
        )
    return new


def _transfer_poly_inr(flax_params, eqx_inr):
    new = eqx_inr
    for i in range(len(eqx_inr.affines)):
        new = eqx.tree_at(
            lambda m, idx=i: m.affines[idx],
            new,
            _transfer_dense(flax_params[f"affines_{i}"], eqx_inr.affines[i]),
        )
    for i in range(len(eqx_inr.dense_layers)):
        new = eqx.tree_at(
            lambda m, idx=i: m.dense_layers[idx],
            new,
            _transfer_dense(flax_params[f"dense_layers_{i}"], eqx_inr.dense_layers[i]),
        )
    new = eqx.tree_at(
        lambda m: m.last_layer,
        new,
        _transfer_dense(flax_params["last_layer"], eqx_inr.last_layer),
    )
    return new


def _transfer_poly_inr_with_hypernet(flax_params, eqx_inr_hyp):
    new = eqx_inr_hyp
    new = eqx.tree_at(
        lambda m: m.inr,
        new,
        _transfer_poly_inr(flax_params["inr"], eqx_inr_hyp.inr),
    )

    num = len(eqx_inr_hyp.inr.affines)
    if eqx_inr_hyp.shift_hypernets is not None:
        for i in range(num):
            key = f"shift_hypernets_{i}"
            if key in flax_params:
                new = eqx.tree_at(
                    lambda m, idx=i: m.shift_hypernets[idx],
                    new,
                    _transfer_mlp(flax_params[key], eqx_inr_hyp.shift_hypernets[i]),
                )
    if eqx_inr_hyp.scale_hypernets is not None:
        for i in range(num):
            key = f"scale_hypernets_{i}"
            if key in flax_params:
                new = eqx.tree_at(
                    lambda m, idx=i: m.scale_hypernets[idx],
                    new,
                    _transfer_mlp(flax_params[key], eqx_inr_hyp.scale_hypernets[i]),
                )
    if eqx_inr_hyp.affine_hypernets is not None:
        for i in range(num):
            key = f"affine_hypernets_{i}"
            if key in flax_params:
                new = eqx.tree_at(
                    lambda m, idx=i: m.affine_hypernets[idx],
                    new,
                    _transfer_mlp(flax_params[key], eqx_inr_hyp.affine_hypernets[i]),
                )
    return new


def _transfer_pde_encoder(flax_params, eqx_enc):
    new = eqx_enc
    new = eqx.tree_at(
        lambda m: m.scalar_encoder,
        new,
        _transfer_mlp(flax_params["scalar_encoder"], eqx_enc.scalar_encoder),
    )
    new = eqx.tree_at(
        lambda m: m.function_encoder,
        new,
        _transfer_conv2d_func_encoder(
            flax_params["function_encoder"], eqx_enc.function_encoder
        ),
    )
    new = eqx.tree_at(
        lambda m: m.graphormer,
        new,
        _transfer_graphormer_encoder(flax_params["graphormer"], eqx_enc.graphormer),
    )
    return new


def transfer_weights(flax_params, eqx_model):
    """Transfer all weights from Flax PDEformer params → Equinox PDEformer."""
    p = flax_params["params"]
    new = eqx_model
    new = eqx.tree_at(
        lambda m: m.pde_encoder,
        new,
        _transfer_pde_encoder(p["pde_encoder"], eqx_model.pde_encoder),
    )
    new = eqx.tree_at(
        lambda m: m.inr,
        new,
        _transfer_poly_inr_with_hypernet(p["inr"], eqx_model.inr),
    )
    if eqx_model.inr2 is not None and "inr2" in p:
        new = eqx.tree_at(
            lambda m: m.inr2,
            new,
            _transfer_poly_inr_with_hypernet(p["inr2"], eqx_model.inr2),
        )
    return new


# ---------------------------------------------------------------------------
# Helper: create both models
# ---------------------------------------------------------------------------


def _make_flax_model():
    return create_pdeformer_from_config({"model": _CFG})


def _make_eqx_model():
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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture
def flax_model():
    return _make_flax_model()


@pytest.fixture
def eqx_model():
    return _make_eqx_model()


@pytest.fixture
def flax_params(flax_model, rng):
    inputs = _make_inputs(jax.random.PRNGKey(0))
    return flax_model.init(rng, **inputs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPDEformerEqx:
    def test_flax_init(self, flax_model, flax_params):
        assert "params" in flax_params

    def test_eqx_init(self, eqx_model):
        assert isinstance(eqx_model, EqxPDEformer)

    def test_weight_transfer(self, eqx_model, flax_params):
        transferred = transfer_weights(flax_params, eqx_model)
        assert isinstance(transferred, EqxPDEformer)

    def test_forward_equivalence(self, flax_model, eqx_model, flax_params, rng):
        inputs = _make_inputs(rng)

        flax_out = flax_model.apply(flax_params, **inputs)
        eqx_transferred = transfer_weights(flax_params, eqx_model)
        eqx_out = eqx_transferred(**inputs)

        diff = jnp.max(jnp.abs(flax_out - eqx_out))
        print(f"\nMax absolute difference: {diff:.2e}")
        # Higher tolerance because sin-based INR amplifies small encoder diffs
        assert diff < 5e-3, f"Outputs differ by {diff:.2e}"
