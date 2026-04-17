"""Tests for MORPH Equinox ↔ Flax forward-pass equivalence."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "repos", "jax_morph"))

from jax_morph.model import ViT3DRegression as FlaxViT3D
from jax_morph.model_eqx import ViT3DRegression as EqxViT3D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# --- Weight transfer utilities ---


def _transfer_dense(flax_params, eqx_linear):
    """Flax Dense kernel (in,out) -> eqx Linear weight (out,in)."""
    w = flax_params["kernel"].T
    b = flax_params["bias"] if "bias" in flax_params else None
    eqx_linear = eqx.tree_at(lambda m: m.weight, eqx_linear, w)
    if b is not None:
        eqx_linear = eqx.tree_at(lambda m: m.bias, eqx_linear, b)
    return eqx_linear


def _transfer_dense_general(flax_params, eqx_linear):
    """Flax DenseGeneral kernel (E, heads, head_dim) -> eqx Linear weight (heads*head_dim, E).
    Flax DenseGeneral bias (heads, head_dim) -> eqx Linear bias (heads*head_dim,)."""
    kernel = flax_params["kernel"]  # (E, heads, head_dim)
    E = kernel.shape[0]
    out_features = kernel.shape[1] * kernel.shape[2]
    w = kernel.reshape(E, out_features).T  # (out, E)
    eqx_linear = eqx.tree_at(lambda m: m.weight, eqx_linear, w)
    if "bias" in flax_params:
        bias = flax_params["bias"]  # (heads, head_dim)
        eqx_linear = eqx.tree_at(
            lambda m: m.bias, eqx_linear, bias.reshape(out_features)
        )
    return eqx_linear


def _transfer_conv3d(flax_params, eqx_conv):
    """Flax Conv kernel (kD,kH,kW,Cin,Cout) -> eqx Conv3d weight (Cout,Cin,kD,kH,kW)."""
    k = flax_params["kernel"]
    w = k.transpose(4, 3, 0, 1, 2)
    eqx_conv = eqx.tree_at(lambda m: m.weight, eqx_conv, w)
    if "bias" in flax_params:
        b = flax_params["bias"]
        eqx_conv = eqx.tree_at(lambda m: m.bias, eqx_conv, b.reshape(-1, 1, 1, 1))
    return eqx_conv


def _transfer_layernorm(flax_params, eqx_ln):
    """Flax LayerNorm scale/bias -> eqx LayerNorm weight/bias."""
    eqx_ln = eqx.tree_at(lambda m: m.weight, eqx_ln, flax_params["scale"])
    eqx_ln = eqx.tree_at(lambda m: m.bias, eqx_ln, flax_params["bias"])
    return eqx_ln


def _transfer_lora_linear(flax_params, eqx_ll):
    """Transfer LoRALinear: base Dense + optional A, B."""
    eqx_ll = eqx.tree_at(
        lambda m: m.base, eqx_ll, _transfer_dense(flax_params["base"], eqx_ll.base)
    )
    if "A" in flax_params and eqx_ll.A is not None:
        eqx_ll = eqx.tree_at(lambda m: m.A, eqx_ll, flax_params["A"])
        eqx_ll = eqx.tree_at(lambda m: m.B, eqx_ll, flax_params["B"])
    return eqx_ll


def _transfer_lora_mha(flax_params, eqx_mha):
    """Transfer LoRAMHA: q, k, v, o projections."""
    for name in ["q", "k", "v", "o"]:
        eqx_proj = getattr(eqx_mha, name)
        eqx_proj = _transfer_lora_linear(flax_params[name], eqx_proj)
        eqx_mha = eqx.tree_at(lambda m, n=name: getattr(m, n), eqx_mha, eqx_proj)
    return eqx_mha


def _transfer_axial_attn(flax_params, eqx_aa):
    """Transfer AxialAttention3DSpaceTime."""
    for name in ["attn_t", "attn_d", "attn_h", "attn_w"]:
        eqx_mha = getattr(eqx_aa, name)
        eqx_mha = _transfer_lora_mha(flax_params[name], eqx_mha)
        eqx_aa = eqx.tree_at(lambda m, n=name: getattr(m, n), eqx_aa, eqx_mha)
    return eqx_aa


def _transfer_encoder_block(flax_params, eqx_block):
    """Transfer EncoderBlock."""
    eqx_block = eqx.tree_at(
        lambda m: m.norm1,
        eqx_block,
        _transfer_layernorm(flax_params["norm1"], eqx_block.norm1),
    )
    eqx_block = eqx.tree_at(
        lambda m: m.norm2,
        eqx_block,
        _transfer_layernorm(flax_params["norm2"], eqx_block.norm2),
    )
    eqx_block = eqx.tree_at(
        lambda m: m.axial_attn,
        eqx_block,
        _transfer_axial_attn(flax_params["axial_attn"], eqx_block.axial_attn),
    )
    eqx_block = eqx.tree_at(
        lambda m: m.mlp_0,
        eqx_block,
        _transfer_lora_linear(flax_params["mlp_0"], eqx_block.mlp_0),
    )
    eqx_block = eqx.tree_at(
        lambda m: m.mlp_1,
        eqx_block,
        _transfer_lora_linear(flax_params["mlp_1"], eqx_block.mlp_1),
    )
    return eqx_block


def _transfer_conv_operator(flax_params, eqx_co):
    """Transfer ConvOperator."""
    eqx_co = eqx.tree_at(
        lambda m: m.input_proj,
        eqx_co,
        _transfer_conv3d(flax_params["input_proj"], eqx_co.input_proj),
    )
    for i, conv in enumerate(eqx_co.conv_stack):
        key = f"conv_stack_{i}"
        conv = _transfer_conv3d(flax_params[key], conv)
        eqx_co = eqx.tree_at(lambda m, idx=i: m.conv_stack[idx], eqx_co, conv)
    return eqx_co


def _transfer_field_cross_attention(flax_params, eqx_fca):
    """Transfer FieldCrossAttention."""
    eqx_fca = eqx.tree_at(lambda m: m.q, eqx_fca, flax_params["q"])

    # DenseGeneral projections
    for name in ["q_proj", "k_proj", "v_proj"]:
        eqx_proj = getattr(eqx_fca, name)
        eqx_proj = _transfer_dense_general(flax_params[name], eqx_proj)
        eqx_fca = eqx.tree_at(lambda m, n=name: getattr(m, n), eqx_fca, eqx_proj)

    eqx_fca = eqx.tree_at(
        lambda m: m.out_proj,
        eqx_fca,
        _transfer_dense(flax_params["out_proj"], eqx_fca.out_proj),
    )
    return eqx_fca


def _transfer_patch_embedding(flax_params, eqx_pe):
    """Transfer HybridPatchEmbedding3D."""
    eqx_pe = eqx.tree_at(
        lambda m: m.conv_features,
        eqx_pe,
        _transfer_conv_operator(flax_params["conv_features"], eqx_pe.conv_features),
    )
    eqx_pe = eqx.tree_at(
        lambda m: m.projection,
        eqx_pe,
        _transfer_dense(flax_params["projection"], eqx_pe.projection),
    )
    eqx_pe = eqx.tree_at(
        lambda m: m.field_attn,
        eqx_pe,
        _transfer_field_cross_attention(flax_params["field_attn"], eqx_pe.field_attn),
    )
    return eqx_pe


def _transfer_decoder(flax_params, eqx_dec):
    """Transfer SimpleDecoder."""
    eqx_dec = eqx.tree_at(
        lambda m: m.norm,
        eqx_dec,
        _transfer_layernorm(flax_params["norm"], eqx_dec.norm),
    )
    eqx_dec = eqx.tree_at(
        lambda m: m.linear,
        eqx_dec,
        _transfer_dense(flax_params["linear"], eqx_dec.linear),
    )
    return eqx_dec


def transfer_all_params(flax_params, eqx_model):
    """Transfer all parameters from Flax to Equinox model."""
    # Patch embedding
    eqx_model = eqx.tree_at(
        lambda m: m.patch_embedding,
        eqx_model,
        _transfer_patch_embedding(
            flax_params["patch_embedding"], eqx_model.patch_embedding
        ),
    )

    # Positional encoding
    eqx_model = eqx.tree_at(
        lambda m: m.pos_encoding.pos_embedding,
        eqx_model,
        flax_params["pos_encoding"]["pos_embedding"],
    )

    # Transformer blocks
    for i in range(eqx_model.depth):
        flax_key = f"transformer_blocks_{i}"
        eqx_block = _transfer_encoder_block(
            flax_params[flax_key], eqx_model.transformer_blocks[i]
        )
        eqx_model = eqx.tree_at(
            lambda m, idx=i: m.transformer_blocks[idx], eqx_model, eqx_block
        )

    # Decoder
    eqx_model = eqx.tree_at(
        lambda m: m.decoder,
        eqx_model,
        _transfer_decoder(flax_params["decoder"], eqx_model.decoder),
    )

    return eqx_model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def models_and_input():
    cfg = _make_config()
    key = jax.random.PRNGKey(42)

    # Flax
    flax_model = FlaxViT3D(**cfg)
    vol = jax.random.normal(key, (1, 1, 1, 1, 4, 4, 4))
    variables = flax_model.init(jax.random.PRNGKey(0), vol, deterministic=True)
    flax_params = variables["params"]

    # Equinox
    eqx_model = EqxViT3D(**cfg, key=jax.random.PRNGKey(1))
    eqx_model = transfer_all_params(flax_params, eqx_model)

    return flax_model, flax_params, eqx_model, vol


def test_forward_equivalence(models_and_input):
    flax_model, flax_params, eqx_model, vol = models_and_input

    enc_f, z_f, out_f = flax_model.apply(
        {"params": flax_params}, vol, deterministic=True
    )
    enc_e, z_e, out_e = eqx_model(vol)

    diff_enc = jnp.max(jnp.abs(enc_f - enc_e))
    diff_z = jnp.max(jnp.abs(z_f - z_e))
    diff_out = jnp.max(jnp.abs(out_f - out_e))

    print(f"Max diff enc: {diff_enc:.2e}, z: {diff_z:.2e}, out: {diff_out:.2e}")
    assert diff_enc < 1e-4, f"enc diff too large: {diff_enc}"
    assert diff_z < 1e-4, f"z diff too large: {diff_z}"
    assert diff_out < 1e-4, f"out diff too large: {diff_out}"


def test_output_shapes(models_and_input):
    _, _, eqx_model, vol = models_and_input
    enc, z, out = eqx_model(vol)
    B, t, F, C, D, H, W = vol.shape
    assert enc.shape[0] == B and enc.shape[-1] == eqx_model.dim
    assert z.shape == enc.shape
    assert out.shape == (B, F, C, D, H, W)


def test_gradients(models_and_input):
    _, _, eqx_model, vol = models_and_input

    @eqx.filter_grad
    def loss_fn(model):
        _, _, out = model(vol)
        return jnp.mean(out**2)

    grads = loss_fn(eqx_model)
    flat = jax.tree_util.tree_leaves(grads)
    assert any(jnp.any(g != 0) for g in flat if eqx.is_array(g))


def test_zero_input(models_and_input):
    flax_model, flax_params, eqx_model, vol = models_and_input
    zero_vol = jnp.zeros_like(vol)

    _, _, out_f = flax_model.apply(
        {"params": flax_params}, zero_vol, deterministic=True
    )
    _, _, out_e = eqx_model(zero_vol)

    diff = jnp.max(jnp.abs(out_f - out_e))
    print(f"Max diff (zero input): {diff:.2e}")
    assert diff < 1e-4, f"zero-input diff too large: {diff}"
