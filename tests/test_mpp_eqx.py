"""Tests for MPP (AViT) Equinox ↔ Flax forward-pass equivalence."""

from __future__ import annotations

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "repos", "jax_mpp"))

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from jax_mpp.avit import AViT as FlaxAViT
from jax_mpp.avit_eqx import AViT as EqxAViT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# --- Weight transfer ---


def _transfer_conv2d(flax_params, eqx_conv):
    """Flax Conv kernel (kH, kW, Cin, Cout) -> eqx Conv2d weight (Cout, Cin, kH, kW)."""
    k = flax_params["kernel"]
    w = k.transpose(3, 2, 0, 1)
    eqx_conv = eqx.tree_at(lambda m: m.weight, eqx_conv, w)
    if "bias" in flax_params:
        b = flax_params["bias"]
        eqx_conv = eqx.tree_at(lambda m: m.bias, eqx_conv, b.reshape(-1, 1, 1))
    return eqx_conv


def _transfer_conv_transpose2d(flax_params, eqx_conv):
    """Flax ConvTranspose (transpose_kernel=True) kernel (kH,kW,Cout,Cin) ->
    eqx ConvTranspose2d weight (Cout,Cin,kH,kW) with spatial flip."""
    k = flax_params["kernel"]
    # transpose_kernel=True: Flax stores (kH,kW,Cout,Cin), need (Cout,Cin,kH,kW)
    # Plus spatial flip since Flax internally flips transpose_kernel=True
    w = k[::-1, ::-1, :, :].transpose(2, 3, 0, 1)
    eqx_conv = eqx.tree_at(lambda m: m.weight, eqx_conv, w)
    if "bias" in flax_params:
        b = flax_params["bias"]
        eqx_conv = eqx.tree_at(lambda m: m.bias, eqx_conv, b.reshape(-1, 1, 1))
    return eqx_conv


def _transfer_dense(flax_params, eqx_linear):
    """Flax Dense kernel (in, out) -> eqx Linear weight (out, in)."""
    w = flax_params["kernel"].T
    eqx_linear = eqx.tree_at(lambda m: m.weight, eqx_linear, w)
    if "bias" in flax_params:
        eqx_linear = eqx.tree_at(lambda m: m.bias, eqx_linear, flax_params["bias"])
    return eqx_linear


def _transfer_layernorm(flax_params, eqx_ln):
    """Flax LayerNorm scale/bias -> eqx LayerNorm weight/bias."""
    eqx_ln = eqx.tree_at(lambda m: m.weight, eqx_ln, flax_params["scale"])
    eqx_ln = eqx.tree_at(lambda m: m.bias, eqx_ln, flax_params["bias"])
    return eqx_ln


def _transfer_rms_instance_norm(flax_params, eqx_norm):
    eqx_norm = eqx.tree_at(lambda m: m.weight, eqx_norm, flax_params["weight"])
    eqx_norm = eqx.tree_at(lambda m: m.bias, eqx_norm, flax_params["bias"])
    return eqx_norm


def _transfer_instance_norm(flax_params, eqx_norm):
    eqx_norm = eqx.tree_at(lambda m: m.weight, eqx_norm, flax_params["weight"])
    eqx_norm = eqx.tree_at(lambda m: m.bias, eqx_norm, flax_params["bias"])
    return eqx_norm


def _transfer_rel_pos_bias(flax_params, eqx_rpb):
    """Transfer RelativePositionBias embedding."""
    emb = flax_params["relative_attention_bias"]["embedding"]
    eqx_rpb = eqx.tree_at(lambda m: m.embedding, eqx_rpb, emb)
    return eqx_rpb


def _transfer_continuous_pos_bias(flax_params, eqx_cpb):
    eqx_cpb = eqx.tree_at(
        lambda m: m.cpb_mlp_0,
        eqx_cpb,
        _transfer_dense(flax_params["cpb_mlp_0"], eqx_cpb.cpb_mlp_0),
    )
    eqx_cpb = eqx.tree_at(
        lambda m: m.cpb_mlp_2,
        eqx_cpb,
        _transfer_dense(flax_params["cpb_mlp_2"], eqx_cpb.cpb_mlp_2),
    )
    return eqx_cpb


def _transfer_mlp(flax_params, eqx_mlp):
    eqx_mlp = eqx.tree_at(
        lambda m: m.fc1, eqx_mlp, _transfer_dense(flax_params["fc1"], eqx_mlp.fc1)
    )
    eqx_mlp = eqx.tree_at(
        lambda m: m.fc2, eqx_mlp, _transfer_dense(flax_params["fc2"], eqx_mlp.fc2)
    )
    return eqx_mlp


def _transfer_pos_bias(flax_params, eqx_bias, bias_type):
    if bias_type == "rel":
        return _transfer_rel_pos_bias(flax_params, eqx_bias)
    elif bias_type == "continuous":
        return _transfer_continuous_pos_bias(flax_params, eqx_bias)
    return eqx_bias


def _transfer_temporal_block(flax_params, eqx_block, bias_type):
    eqx_block = eqx.tree_at(
        lambda m: m.norm1,
        eqx_block,
        _transfer_instance_norm(flax_params["norm1"], eqx_block.norm1),
    )
    eqx_block = eqx.tree_at(
        lambda m: m.input_head,
        eqx_block,
        _transfer_conv2d(flax_params["input_head"], eqx_block.input_head),
    )
    eqx_block = eqx.tree_at(
        lambda m: m.qnorm,
        eqx_block,
        _transfer_layernorm(flax_params["qnorm"], eqx_block.qnorm),
    )
    eqx_block = eqx.tree_at(
        lambda m: m.knorm,
        eqx_block,
        _transfer_layernorm(flax_params["knorm"], eqx_block.knorm),
    )
    if eqx_block.rel_pos_bias is not None and "rel_pos_bias" in flax_params:
        eqx_block = eqx.tree_at(
            lambda m: m.rel_pos_bias,
            eqx_block,
            _transfer_pos_bias(
                flax_params["rel_pos_bias"], eqx_block.rel_pos_bias, bias_type
            ),
        )
    eqx_block = eqx.tree_at(
        lambda m: m.norm2,
        eqx_block,
        _transfer_instance_norm(flax_params["norm2"], eqx_block.norm2),
    )
    eqx_block = eqx.tree_at(
        lambda m: m.output_head,
        eqx_block,
        _transfer_conv2d(flax_params["output_head"], eqx_block.output_head),
    )
    if "gamma" in flax_params and eqx_block.gamma is not None:
        eqx_block = eqx.tree_at(lambda m: m.gamma, eqx_block, flax_params["gamma"])
    return eqx_block


def _transfer_spatial_block(flax_params, eqx_block, bias_type):
    eqx_block = eqx.tree_at(
        lambda m: m.norm1,
        eqx_block,
        _transfer_rms_instance_norm(flax_params["norm1"], eqx_block.norm1),
    )
    eqx_block = eqx.tree_at(
        lambda m: m.input_head,
        eqx_block,
        _transfer_conv2d(flax_params["input_head"], eqx_block.input_head),
    )
    eqx_block = eqx.tree_at(
        lambda m: m.qnorm,
        eqx_block,
        _transfer_layernorm(flax_params["qnorm"], eqx_block.qnorm),
    )
    eqx_block = eqx.tree_at(
        lambda m: m.knorm,
        eqx_block,
        _transfer_layernorm(flax_params["knorm"], eqx_block.knorm),
    )
    if eqx_block.rel_pos_bias is not None and "rel_pos_bias" in flax_params:
        eqx_block = eqx.tree_at(
            lambda m: m.rel_pos_bias,
            eqx_block,
            _transfer_pos_bias(
                flax_params["rel_pos_bias"], eqx_block.rel_pos_bias, bias_type
            ),
        )
    eqx_block = eqx.tree_at(
        lambda m: m.norm2,
        eqx_block,
        _transfer_rms_instance_norm(flax_params["norm2"], eqx_block.norm2),
    )
    eqx_block = eqx.tree_at(
        lambda m: m.output_head,
        eqx_block,
        _transfer_conv2d(flax_params["output_head"], eqx_block.output_head),
    )
    if "gamma_att" in flax_params and eqx_block.gamma_att is not None:
        eqx_block = eqx.tree_at(
            lambda m: m.gamma_att, eqx_block, flax_params["gamma_att"]
        )
    if "gamma_mlp" in flax_params and eqx_block.gamma_mlp is not None:
        eqx_block = eqx.tree_at(
            lambda m: m.gamma_mlp, eqx_block, flax_params["gamma_mlp"]
        )
    eqx_block = eqx.tree_at(
        lambda m: m.mlp, eqx_block, _transfer_mlp(flax_params["mlp"], eqx_block.mlp)
    )
    eqx_block = eqx.tree_at(
        lambda m: m.mlp_norm,
        eqx_block,
        _transfer_rms_instance_norm(flax_params["mlp_norm"], eqx_block.mlp_norm),
    )
    return eqx_block


def _transfer_spacetime_block(flax_params, eqx_block, bias_type):
    eqx_block = eqx.tree_at(
        lambda m: m.temporal,
        eqx_block,
        _transfer_temporal_block(
            flax_params["temporal"], eqx_block.temporal, bias_type
        ),
    )
    eqx_block = eqx.tree_at(
        lambda m: m.spatial,
        eqx_block,
        _transfer_spatial_block(flax_params["spatial"], eqx_block.spatial, bias_type),
    )
    return eqx_block


def _transfer_hmlp_stem(flax_params, eqx_stem):
    eqx_stem = eqx.tree_at(
        lambda m: m.in_proj_0,
        eqx_stem,
        _transfer_conv2d(flax_params["in_proj_0"], eqx_stem.in_proj_0),
    )
    eqx_stem = eqx.tree_at(
        lambda m: m.in_proj_1,
        eqx_stem,
        _transfer_rms_instance_norm(flax_params["in_proj_1"], eqx_stem.in_proj_1),
    )
    eqx_stem = eqx.tree_at(
        lambda m: m.in_proj_3,
        eqx_stem,
        _transfer_conv2d(flax_params["in_proj_3"], eqx_stem.in_proj_3),
    )
    eqx_stem = eqx.tree_at(
        lambda m: m.in_proj_4,
        eqx_stem,
        _transfer_rms_instance_norm(flax_params["in_proj_4"], eqx_stem.in_proj_4),
    )
    eqx_stem = eqx.tree_at(
        lambda m: m.in_proj_6,
        eqx_stem,
        _transfer_conv2d(flax_params["in_proj_6"], eqx_stem.in_proj_6),
    )
    eqx_stem = eqx.tree_at(
        lambda m: m.in_proj_7,
        eqx_stem,
        _transfer_rms_instance_norm(flax_params["in_proj_7"], eqx_stem.in_proj_7),
    )
    return eqx_stem


def _transfer_hmlp_output(flax_params, eqx_debed):
    eqx_debed = eqx.tree_at(
        lambda m: m.out_proj_0,
        eqx_debed,
        _transfer_conv_transpose2d(flax_params["out_proj_0"], eqx_debed.out_proj_0),
    )
    eqx_debed = eqx.tree_at(
        lambda m: m.out_proj_1,
        eqx_debed,
        _transfer_rms_instance_norm(flax_params["out_proj_1"], eqx_debed.out_proj_1),
    )
    eqx_debed = eqx.tree_at(
        lambda m: m.out_proj_3,
        eqx_debed,
        _transfer_conv_transpose2d(flax_params["out_proj_3"], eqx_debed.out_proj_3),
    )
    eqx_debed = eqx.tree_at(
        lambda m: m.out_proj_4,
        eqx_debed,
        _transfer_rms_instance_norm(flax_params["out_proj_4"], eqx_debed.out_proj_4),
    )
    eqx_debed = eqx.tree_at(
        lambda m: m.out_kernel, eqx_debed, flax_params["out_kernel"]
    )
    eqx_debed = eqx.tree_at(lambda m: m.out_bias, eqx_debed, flax_params["out_bias"])
    return eqx_debed


def transfer_all_params(flax_params, eqx_model, bias_type="rel"):
    # space_bag
    eqx_model = eqx.tree_at(
        lambda m: m.space_bag.weight, eqx_model, flax_params["space_bag"]["weight"]
    )
    eqx_model = eqx.tree_at(
        lambda m: m.space_bag.bias, eqx_model, flax_params["space_bag"]["bias"]
    )

    # embed
    eqx_model = eqx.tree_at(
        lambda m: m.embed,
        eqx_model,
        _transfer_hmlp_stem(flax_params["embed"], eqx_model.embed),
    )

    # blocks
    for i in range(eqx_model.processor_blocks):
        flax_key = f"blocks_{i}"
        eqx_block = _transfer_spacetime_block(
            flax_params[flax_key], eqx_model.blocks[i], bias_type
        )
        eqx_model = eqx.tree_at(lambda m, idx=i: m.blocks[idx], eqx_model, eqx_block)

    # debed
    eqx_model = eqx.tree_at(
        lambda m: m.debed,
        eqx_model,
        _transfer_hmlp_output(flax_params["debed"], eqx_model.debed),
    )

    return eqx_model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def models_and_input():
    cfg = _make_config()
    key = jax.random.PRNGKey(42)

    T, B, C, H, W = 2, 1, 2, 32, 32
    x = jax.random.normal(key, (T, B, C, H, W))
    labels = jnp.array([0, 1])
    bcs = jnp.zeros((B, 2), dtype=jnp.int32)

    flax_model = FlaxAViT(**cfg)
    variables = flax_model.init(
        {"params": jax.random.PRNGKey(0), "drop_path": jax.random.PRNGKey(0)},
        x,
        labels,
        bcs,
        deterministic=True,
    )
    flax_params = variables["params"]

    eqx_model = EqxAViT(**cfg, key=jax.random.PRNGKey(1))
    eqx_model = transfer_all_params(flax_params, eqx_model, bias_type=cfg["bias_type"])

    return flax_model, flax_params, eqx_model, x, labels, bcs


def test_forward_equivalence(models_and_input):
    flax_model, flax_params, eqx_model, x, labels, bcs = models_and_input

    out_f = flax_model.apply(
        {"params": flax_params},
        x,
        labels,
        bcs,
        deterministic=True,
        rngs={"drop_path": jax.random.PRNGKey(0)},
    )
    out_e = eqx_model(x, labels, bcs, deterministic=True)

    diff = jnp.max(jnp.abs(out_f - out_e))
    print(f"Max diff: {diff:.2e}")
    assert diff < 1e-4, f"diff too large: {diff}"


def test_output_shapes(models_and_input):
    _, _, eqx_model, x, labels, bcs = models_and_input
    out = eqx_model(x, labels, bcs, deterministic=True)
    T, B, C, H, W = x.shape
    assert out.shape == (B, C, H, W)


def test_gradients(models_and_input):
    _, _, eqx_model, x, labels, bcs = models_and_input

    @eqx.filter_grad
    def loss_fn(model):
        out = model(x, labels, bcs, deterministic=True)
        return jnp.mean(out**2)

    grads = loss_fn(eqx_model)
    flat = jax.tree_util.tree_leaves(grads)
    assert any(jnp.any(g != 0) for g in flat if eqx.is_array(g))


def test_zero_input(models_and_input):
    flax_model, flax_params, eqx_model, x, labels, bcs = models_and_input
    zero_x = jnp.zeros_like(x)

    out_f = flax_model.apply(
        {"params": flax_params},
        zero_x,
        labels,
        bcs,
        deterministic=True,
        rngs={"drop_path": jax.random.PRNGKey(0)},
    )
    out_e = eqx_model(zero_x, labels, bcs, deterministic=True)

    diff = jnp.max(jnp.abs(out_f - out_e))
    print(f"Max diff (zero input): {diff:.2e}")
    assert diff < 1e-4, f"zero-input diff too large: {diff}"
