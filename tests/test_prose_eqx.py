"""Tests verifying Equinox PROSE matches Flax PROSE forward pass for all 4 variants."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import pytest

from jax_prose.load import prose_fd_1to1, prose_fd_2to1, prose_ode_2to1, prose_pde_2to1
from jax_prose.config import (
    PROSE1to1Config,
    EmbedderConfig,
    DataEncoderConfig,
    DataDecoderConfig,
)
from jax_prose.prose_fd_2to1 import Prose2to1Config
from jax_prose.prose_ode_pde_2to1 import ProseTextData2to1Config
from jax_prose.model_eqx import (
    PROSE1to1 as EqxPROSE1to1,
    PROSE2to1 as EqxPROSE2to1,
    PROSEODE2to1 as EqxPROSEODE2to1,
    PROSEPDE2to1 as EqxPROSEPDE2to1,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Generic weight transfer helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _set_weight(eqx_linear, kernel, bias=None):
    """Flax Dense kernel (in, out) → eqx Linear weight (out, in)."""
    new = eqx.tree_at(lambda m: m.weight, eqx_linear, kernel.T)
    if bias is not None and eqx_linear.bias is not None:
        new = eqx.tree_at(lambda m: m.bias, new, bias)
    return new


def _set_conv2d(eqx_conv, kernel, bias=None):
    """Flax Conv kernel (kH,kW,Cin,Cout) → eqx Conv2d weight (Cout,Cin,kH,kW)."""
    new = eqx.tree_at(lambda m: m.weight, eqx_conv, kernel.transpose(3, 2, 0, 1))
    if bias is not None and eqx_conv.bias is not None:
        new = eqx.tree_at(lambda m: m.bias, new, bias.reshape(-1, 1, 1))
    return new


def _set_conv_transpose2d(eqx_conv, kernel, bias=None):
    """Flax ConvTranspose(transpose_kernel=False) kernel (kH,kW,Cin,Cout)
    → eqx ConvTranspose2d weight (Cout,Cin,kH,kW)."""
    new = eqx.tree_at(lambda m: m.weight, eqx_conv, kernel.transpose(3, 2, 0, 1))
    if bias is not None and eqx_conv.bias is not None:
        new = eqx.tree_at(lambda m: m.bias, new, bias.reshape(-1, 1, 1))
    return new


def _set_rmsnorm(eqx_norm, scale):
    return eqx.tree_at(lambda m: m.weight, eqx_norm, scale)


def _set_layernorm(eqx_ln, flax_params):
    new = eqx.tree_at(lambda m: m.weight, eqx_ln, flax_params["scale"])
    if "bias" in flax_params:
        new = eqx.tree_at(lambda m: m.bias, new, flax_params["bias"])
    return new


def _set_embedding(eqx_emb, flax_params):
    return eqx.tree_at(lambda m: m.weight, eqx_emb, flax_params["embedding"])


def _transfer_flax_mha(flax_mha_params, eqx_mha):
    """Transfer Flax nn.MultiHeadDotProductAttention → eqx MultiHeadAttention1to1.

    Flax DenseGeneral layout:
      query/key/value kernel: (features, heads, head_dim)
      query/key/value bias:   (heads, head_dim)
      out kernel:             (heads, head_dim, features)
      out bias:               (features,)
    """
    new = eqx_mha
    for flax_name, eqx_attr in [
        ("query", "q_proj"),
        ("key", "k_proj"),
        ("value", "v_proj"),
    ]:
        fp = flax_mha_params[flax_name]
        kernel = fp["kernel"]  # (E, H, D)
        E = kernel.shape[0]
        HD = kernel.shape[1] * kernel.shape[2]
        w = kernel.reshape(E, HD).T  # (HD, E)
        proj = getattr(new, eqx_attr)
        proj = eqx.tree_at(lambda m: m.weight, proj, w)
        if "bias" in fp:
            proj = eqx.tree_at(lambda m: m.bias, proj, fp["bias"].reshape(HD))
        new = eqx.tree_at(lambda m, a=eqx_attr: getattr(m, a), new, proj)

    # out projection: kernel (H, D, E)
    fp = flax_mha_params["out"]
    kernel = fp["kernel"]  # (H, D, E)
    HD = kernel.shape[0] * kernel.shape[1]
    E = kernel.shape[2]
    w = kernel.reshape(HD, E).T  # (E, HD)
    out = new.out_proj
    out = eqx.tree_at(lambda m: m.weight, out, w)
    if "bias" in fp:
        out = eqx.tree_at(lambda m: m.bias, out, fp["bias"])
    new = eqx.tree_at(lambda m: m.out_proj, new, out)
    return new


def _transfer_dense(flax_params, eqx_linear):
    return _set_weight(eqx_linear, flax_params["kernel"], flax_params.get("bias"))


# ═══════════════════════════════════════════════════════════════════════════════
# PROSE1to1 weight transfer
# ═══════════════════════════════════════════════════════════════════════════════


def _transfer_1to1_encoder_block(fp, eqx_block):
    """EncoderBlock (nn.compact): RMSNorm_0, RMSNorm_1, MHDA_0, Dense_0, Dense_1."""
    new = eqx_block
    new = eqx.tree_at(
        lambda m: m.norm1, new, _set_rmsnorm(new.norm1, fp["RMSNorm_0"]["scale"])
    )
    new = eqx.tree_at(
        lambda m: m.norm2, new, _set_rmsnorm(new.norm2, fp["RMSNorm_1"]["scale"])
    )
    new = eqx.tree_at(
        lambda m: m.attn,
        new,
        _transfer_flax_mha(fp["MultiHeadDotProductAttention_0"], new.attn),
    )
    new = eqx.tree_at(
        lambda m: m.dense1, new, _transfer_dense(fp["Dense_0"], new.dense1)
    )
    new = eqx.tree_at(
        lambda m: m.dense2, new, _transfer_dense(fp["Dense_1"], new.dense2)
    )
    return new


def _transfer_1to1_decoder_block(fp, eqx_block):
    """OperatorDecoderBlock (nn.compact): same structure as encoder block."""
    return _transfer_1to1_encoder_block(fp, eqx_block)


def transfer_weights_1to1(flax_params, eqx_model):
    """Transfer all Flax PROSE1to1 params to Equinox."""
    p = flax_params["params"]
    model = eqx_model

    # --- Embedder ---
    ep = p["embedder"]
    emb = model.embedder
    emb = eqx.tree_at(
        lambda m: m.patch_position_embeddings, emb, ep["patch_position_embeddings"]
    )
    if emb.time_embed is not None:
        emb = eqx.tree_at(lambda m: m.time_embed, emb, ep["time_embed"])
    else:
        # continuous time: Sequential = [Dense_0, gelu, Dense_1]
        tp = ep["time_proj"]
        emb = eqx.tree_at(
            lambda m: m.time_proj_dense1,
            emb,
            _transfer_dense(tp["layers_0"], emb.time_proj_dense1),
        )
        emb = eqx.tree_at(
            lambda m: m.time_proj_dense2,
            emb,
            _transfer_dense(tp["layers_2"], emb.time_proj_dense2),
        )

    emb = eqx.tree_at(
        lambda m: m.conv_proj_0,
        emb,
        _set_conv2d(
            emb.conv_proj_0, ep["conv_proj_0"]["kernel"], ep["conv_proj_0"].get("bias")
        ),
    )
    emb = eqx.tree_at(
        lambda m: m.conv_proj_1,
        emb,
        _set_conv2d(
            emb.conv_proj_1, ep["conv_proj_1"]["kernel"], ep["conv_proj_1"].get("bias")
        ),
    )
    emb = eqx.tree_at(
        lambda m: m.deconv,
        emb,
        _set_conv_transpose2d(
            emb.deconv, ep["deconv"]["kernel"], ep["deconv"].get("bias")
        ),
    )
    emb = eqx.tree_at(
        lambda m: m.post_conv_0,
        emb,
        _set_conv2d(
            emb.post_conv_0, ep["post_conv_0"]["kernel"], ep["post_conv_0"].get("bias")
        ),
    )
    emb = eqx.tree_at(
        lambda m: m.post_conv_1,
        emb,
        _set_conv2d(
            emb.post_conv_1, ep["post_conv_1"]["kernel"], ep["post_conv_1"].get("bias")
        ),
    )
    model = eqx.tree_at(lambda m: m.embedder, model, emb)

    # --- Data Encoder ---
    dep = p["data_encoder"]
    for i, layer in enumerate(model.data_encoder.layers):
        model = eqx.tree_at(
            lambda m, idx=i: m.data_encoder.layers[idx],
            model,
            _transfer_1to1_encoder_block(dep[f"layer_{i}"], layer),
        )

    # --- Data Decoder ---
    ddp = p["data_decoder"]
    dec = model.data_decoder
    dec = eqx.tree_at(lambda m: m.time_embed, dec, ddp["time_embed"])
    dec = eqx.tree_at(
        lambda m: m.patch_position_embeddings, dec, ddp["patch_position_embeddings"]
    )
    for i, layer in enumerate(dec.layers):
        dec = eqx.tree_at(
            lambda m, idx=i: m.layers[idx],
            dec,
            _transfer_1to1_decoder_block(ddp[f"layer_{i}"], layer),
        )
    if dec.final_norm is not None and "RMSNorm_0" in ddp:
        dec = eqx.tree_at(
            lambda m: m.final_norm,
            dec,
            _set_rmsnorm(dec.final_norm, ddp["RMSNorm_0"]["scale"]),
        )
    model = eqx.tree_at(lambda m: m.data_decoder, model, dec)

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# PROSE2to1 weight transfer
# ═══════════════════════════════════════════════════════════════════════════════


def _transfer_custom_mha(fp, eqx_mha):
    """Transfer CustomMHA: linear_q, linear_k, linear_v, out_proj — all nn.Dense."""
    new = eqx_mha
    for name in ("linear_q", "linear_k", "linear_v", "out_proj"):
        new = eqx.tree_at(
            lambda m, n=name: getattr(m, n),
            new,
            _transfer_dense(fp[name], getattr(new, name)),
        )
    return new


def _transfer_rmsnorm_scale(fp, eqx_norm):
    return eqx.tree_at(lambda m: m.scale, eqx_norm, fp["scale"])


def _transfer_encoder_layer_2to1(fp, eqx_layer):
    new = eqx_layer
    new = eqx.tree_at(
        lambda m: m.self_attn, new, _transfer_custom_mha(fp["self_attn"], new.self_attn)
    )
    new = eqx.tree_at(
        lambda m: m.linear1, new, _transfer_dense(fp["linear1"], new.linear1)
    )
    new = eqx.tree_at(
        lambda m: m.linear2, new, _transfer_dense(fp["linear2"], new.linear2)
    )
    new = eqx.tree_at(
        lambda m: m.norm1, new, _transfer_rmsnorm_scale(fp["norm1"], new.norm1)
    )
    new = eqx.tree_at(
        lambda m: m.norm2, new, _transfer_rmsnorm_scale(fp["norm2"], new.norm2)
    )
    return new


def _transfer_encoder_2to1(fp, eqx_enc):
    new = eqx_enc
    for i, layer in enumerate(new.layers):
        new = eqx.tree_at(
            lambda m, idx=i: m.layers[idx],
            new,
            _transfer_encoder_layer_2to1(fp[f"layers_{i}"], layer),
        )
    new = eqx.tree_at(
        lambda m: m.norm, new, _transfer_rmsnorm_scale(fp["norm"], new.norm)
    )
    return new


def _transfer_decoder_layer_2to1(fp, eqx_layer):
    new = eqx_layer
    new = eqx.tree_at(
        lambda m: m.multihead_attn,
        new,
        _transfer_custom_mha(fp["multihead_attn"], new.multihead_attn),
    )
    new = eqx.tree_at(
        lambda m: m.linear1, new, _transfer_dense(fp["linear1"], new.linear1)
    )
    new = eqx.tree_at(
        lambda m: m.linear2, new, _transfer_dense(fp["linear2"], new.linear2)
    )
    new = eqx.tree_at(
        lambda m: m.norm1, new, _transfer_rmsnorm_scale(fp["norm1"], new.norm1)
    )
    new = eqx.tree_at(
        lambda m: m.norm2, new, _transfer_rmsnorm_scale(fp["norm2"], new.norm2)
    )
    return new


def _transfer_conv_embedder_2to1(fp, eqx_emb):
    new = eqx_emb
    new = eqx.tree_at(
        lambda m: m.patch_position_embeddings, new, fp["patch_position_embeddings"]
    )
    new = eqx.tree_at(lambda m: m.time_embed, new, fp["time_embed"])
    new = eqx.tree_at(
        lambda m: m.conv_proj_0,
        new,
        _set_conv2d(
            new.conv_proj_0, fp["conv_proj_0"]["kernel"], fp["conv_proj_0"].get("bias")
        ),
    )
    new = eqx.tree_at(
        lambda m: m.conv_proj_1,
        new,
        _set_conv2d(
            new.conv_proj_1, fp["conv_proj_1"]["kernel"], fp["conv_proj_1"].get("bias")
        ),
    )
    new = eqx.tree_at(
        lambda m: m.deconv,
        new,
        _set_conv_transpose2d(
            new.deconv, fp["deconv"]["kernel"], fp["deconv"].get("bias")
        ),
    )
    new = eqx.tree_at(
        lambda m: m.post_conv_0,
        new,
        _set_conv2d(
            new.post_conv_0, fp["post_conv_0"]["kernel"], fp["post_conv_0"].get("bias")
        ),
    )
    new = eqx.tree_at(
        lambda m: m.post_conv_1,
        new,
        _set_conv2d(
            new.post_conv_1, fp["post_conv_1"]["kernel"], fp["post_conv_1"].get("bias")
        ),
    )
    return new


def transfer_weights_2to1(flax_params, eqx_model):
    p = flax_params["params"]
    model = eqx_model

    # Embedder
    model = eqx.tree_at(
        lambda m: m.embedder,
        model,
        _transfer_conv_embedder_2to1(p["embedder"], model.embedder),
    )

    # Data encoder
    model = eqx.tree_at(
        lambda m: m.data_encoder,
        model,
        _transfer_encoder_2to1(p["data_encoder"], model.data_encoder),
    )

    # Symbol encoder
    sep = p["symbol_encoder"]
    se = model.symbol_encoder
    se = eqx.tree_at(
        lambda m: m.word_embeddings,
        se,
        _set_embedding(se.word_embeddings, sep["word_embeddings"]),
    )
    se = eqx.tree_at(lambda m: m.pe, se, sep["pe"])
    se = eqx.tree_at(
        lambda m: m.transformer_encoder,
        se,
        _transfer_encoder_2to1(sep["transformer_encoder"], se.transformer_encoder),
    )
    model = eqx.tree_at(lambda m: m.symbol_encoder, model, se)

    # Fusion
    fp = p["fusion"]
    fu = model.fusion
    fu = eqx.tree_at(
        lambda m: m.type_embeddings,
        fu,
        _set_embedding(fu.type_embeddings, fp["type_embeddings"]),
    )
    fu = eqx.tree_at(
        lambda m: m.transformer_encoder,
        fu,
        _transfer_encoder_2to1(fp["transformer_encoder"], fu.transformer_encoder),
    )
    model = eqx.tree_at(lambda m: m.fusion, model, fu)

    # Data decoder
    ddp = p["data_decoder"]
    dd = model.data_decoder
    dd = eqx.tree_at(lambda m: m.time_embed, dd, ddp["time_embed"])
    dd = eqx.tree_at(
        lambda m: m.patch_position_embeddings, dd, ddp["patch_position_embeddings"]
    )
    for i, layer in enumerate(dd.layers):
        dd = eqx.tree_at(
            lambda m, idx=i: m.layers[idx],
            dd,
            _transfer_decoder_layer_2to1(ddp[f"layers_{i}"], layer),
        )
    dd = eqx.tree_at(
        lambda m: m.norm, dd, _transfer_rmsnorm_scale(ddp["norm"], dd.norm)
    )
    model = eqx.tree_at(lambda m: m.data_decoder, model, dd)

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# PROSE ODE/PDE 2to1 weight transfer  (shared components)
# ═══════════════════════════════════════════════════════════════════════════════


def _transfer_torch_mha(fp, eqx_mha):
    """Transfer TorchLikeMHA: q_proj, k_proj, v_proj, out_proj."""
    new = eqx_mha
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        new = eqx.tree_at(
            lambda m, n=name: getattr(m, n),
            new,
            _transfer_dense(fp[name], getattr(new, name)),
        )
    return new


def _transfer_ffn(fp, eqx_ffn):
    """Transfer TransformerFFN."""
    new = eqx_ffn
    new = eqx.tree_at(lambda m: m.lin1, new, _transfer_dense(fp["lin1"], new.lin1))
    new = eqx.tree_at(lambda m: m.lin2, new, _transfer_dense(fp["lin2"], new.lin2))
    for i, mid in enumerate(new.mid):
        new = eqx.tree_at(
            lambda m, idx=i: m.mid[idx], new, _transfer_dense(fp[f"midlin_{i}"], mid)
        )
    return new


def _transfer_data_transformer(fp, eqx_dtm):
    """Transfer DataTransformerModel."""
    new = eqx_dtm
    new = eqx.tree_at(
        lambda m: m.layer_norm_emb,
        new,
        _set_layernorm(new.layer_norm_emb, fp["layer_norm_emb"]),
    )
    if new.position_embeddings is not None and "position_embeddings" in fp:
        new = eqx.tree_at(
            lambda m: m.position_embeddings,
            new,
            _set_embedding(new.position_embeddings, fp["position_embeddings"]),
        )
    for i in range(new.n_layers):
        new = eqx.tree_at(
            lambda m, idx=i: m.attentions[idx],
            new,
            _transfer_torch_mha(fp[f"attentions_{i}"], new.attentions[i]),
        )
        new = eqx.tree_at(
            lambda m, idx=i: m.layer_norm1[idx],
            new,
            _set_layernorm(new.layer_norm1[i], fp[f"layer_norm1_{i}"]),
        )
        new = eqx.tree_at(
            lambda m, idx=i: m.ffns[idx],
            new,
            _transfer_ffn(fp[f"ffns_{i}"], new.ffns[i]),
        )
        new = eqx.tree_at(
            lambda m, idx=i: m.layer_norm2[idx],
            new,
            _set_layernorm(new.layer_norm2[i], fp[f"layer_norm2_{i}"]),
        )
    return new


def _transfer_text_transformer(fp, eqx_ttm):
    """Transfer TextTransformerModel."""
    new = eqx_ttm
    new = eqx.tree_at(
        lambda m: m.embeddings, new, _set_embedding(new.embeddings, fp["embeddings"])
    )
    new = eqx.tree_at(
        lambda m: m.layer_norm_emb,
        new,
        _set_layernorm(new.layer_norm_emb, fp["layer_norm_emb"]),
    )
    if new.position_embeddings is not None and "position_embeddings" in fp:
        new = eqx.tree_at(
            lambda m: m.position_embeddings,
            new,
            _set_embedding(new.position_embeddings, fp["position_embeddings"]),
        )
    for i in range(new.n_layers):
        new = eqx.tree_at(
            lambda m, idx=i: m.attentions[idx],
            new,
            _transfer_torch_mha(fp[f"attentions_{i}"], new.attentions[i]),
        )
        new = eqx.tree_at(
            lambda m, idx=i: m.layer_norm1[idx],
            new,
            _set_layernorm(new.layer_norm1[i], fp[f"layer_norm1_{i}"]),
        )
        new = eqx.tree_at(
            lambda m, idx=i: m.ffns[idx],
            new,
            _transfer_ffn(fp[f"ffns_{i}"], new.ffns[i]),
        )
        new = eqx.tree_at(
            lambda m, idx=i: m.layer_norm2[idx],
            new,
            _set_layernorm(new.layer_norm2[i], fp[f"layer_norm2_{i}"]),
        )
    return new


def _transfer_fusion_transformer(fp, eqx_ftm):
    """Transfer FusionTransformerModel."""
    new = eqx_ftm
    if new.type_embeddings is not None and "type_embeddings" in fp:
        new = eqx.tree_at(
            lambda m: m.type_embeddings,
            new,
            _set_embedding(new.type_embeddings, fp["type_embeddings"]),
        )
    new = eqx.tree_at(
        lambda m: m.layer_norm_emb,
        new,
        _set_layernorm(new.layer_norm_emb, fp["layer_norm_emb"]),
    )
    for i in range(new.n_layers):
        new = eqx.tree_at(
            lambda m, idx=i: m.attentions[idx],
            new,
            _transfer_torch_mha(fp[f"attentions_{i}"], new.attentions[i]),
        )
        new = eqx.tree_at(
            lambda m, idx=i: m.layer_norm1[idx],
            new,
            _set_layernorm(new.layer_norm1[i], fp[f"layer_norm1_{i}"]),
        )
        new = eqx.tree_at(
            lambda m, idx=i: m.ffns[idx],
            new,
            _transfer_ffn(fp[f"ffns_{i}"], new.ffns[i]),
        )
        new = eqx.tree_at(
            lambda m, idx=i: m.layer_norm2[idx],
            new,
            _set_layernorm(new.layer_norm2[i], fp[f"layer_norm2_{i}"]),
        )
    return new


def _transfer_data_operator(fp, eqx_dom):
    """Transfer DataOperatorModel."""
    new = eqx_dom
    new = eqx.tree_at(
        lambda m: m.query_embedder,
        new,
        _transfer_dense(fp["query_embedder"], new.query_embedder),
    )
    new = eqx.tree_at(
        lambda m: m.layer_norm_emb,
        new,
        _set_layernorm(new.layer_norm_emb, fp["layer_norm_emb"]),
    )
    if new.position_embeddings is not None and "position_embeddings" in fp:
        new = eqx.tree_at(
            lambda m: m.position_embeddings,
            new,
            _set_embedding(new.position_embeddings, fp["position_embeddings"]),
        )
    if new.data_embedder_0 is not None and "data_embedder_0" in fp:
        new = eqx.tree_at(
            lambda m: m.data_embedder_0,
            new,
            _transfer_dense(fp["data_embedder_0"], new.data_embedder_0),
        )
        new = eqx.tree_at(
            lambda m: m.data_embedder_2,
            new,
            _transfer_dense(fp["data_embedder_2"], new.data_embedder_2),
        )
    if new.text_embedder_0 is not None and "text_embedder_0" in fp:
        new = eqx.tree_at(
            lambda m: m.text_embedder_0,
            new,
            _transfer_dense(fp["text_embedder_0"], new.text_embedder_0),
        )
        new = eqx.tree_at(
            lambda m: m.text_embedder_2,
            new,
            _transfer_dense(fp["text_embedder_2"], new.text_embedder_2),
        )
    for i in range(new.n_layers):
        new = eqx.tree_at(
            lambda m, idx=i: m.attentions[idx],
            new,
            _transfer_torch_mha(fp[f"attentions_{i}"], new.attentions[i]),
        )
        new = eqx.tree_at(
            lambda m, idx=i: m.layer_norm1[idx],
            new,
            _set_layernorm(new.layer_norm1[i], fp[f"layer_norm1_{i}"]),
        )
        if new.encoder_attn is not None:
            new = eqx.tree_at(
                lambda m, idx=i: m.encoder_attn[idx],
                new,
                _transfer_torch_mha(fp[f"encoder_attn_{i}"], new.encoder_attn[i]),
            )
            new = eqx.tree_at(
                lambda m, idx=i: m.layer_norm15[idx],
                new,
                _set_layernorm(new.layer_norm15[i], fp[f"layer_norm15_{i}"]),
            )
        new = eqx.tree_at(
            lambda m, idx=i: m.ffns[idx],
            new,
            _transfer_ffn(fp[f"ffns_{i}"], new.ffns[i]),
        )
        new = eqx.tree_at(
            lambda m, idx=i: m.layer_norm2[idx],
            new,
            _set_layernorm(new.layer_norm2[i], fp[f"layer_norm2_{i}"]),
        )
    if new.proj is not None and "proj" in fp:
        new = eqx.tree_at(lambda m: m.proj, new, _transfer_dense(fp["proj"], new.proj))
    if new.proj_0 is not None and "proj_0" in fp:
        new = eqx.tree_at(
            lambda m: m.proj_0, new, _transfer_dense(fp["proj_0"], new.proj_0)
        )
        new = eqx.tree_at(
            lambda m: m.proj_1, new, _transfer_dense(fp["proj_1"], new.proj_1)
        )
    return new


def transfer_weights_ode(flax_params, eqx_model):
    p = flax_params["params"]
    model = eqx_model
    model = eqx.tree_at(
        lambda m: m.embedder_0,
        model,
        _transfer_dense(p["embedder_0"], model.embedder_0),
    )
    model = eqx.tree_at(
        lambda m: m.embedder_2,
        model,
        _transfer_dense(p["embedder_2"], model.embedder_2),
    )
    model = eqx.tree_at(
        lambda m: m.data_encoder,
        model,
        _transfer_data_transformer(p["data_encoder"], model.data_encoder),
    )
    model = eqx.tree_at(
        lambda m: m.text_encoder,
        model,
        _transfer_text_transformer(p["text_encoder"], model.text_encoder),
    )
    model = eqx.tree_at(
        lambda m: m.fusion,
        model,
        _transfer_fusion_transformer(p["fusion"], model.fusion),
    )
    model = eqx.tree_at(
        lambda m: m.data_decoder,
        model,
        _transfer_data_operator(p["data_decoder"], model.data_decoder),
    )
    return model


def transfer_weights_pde(flax_params, eqx_model):
    p = flax_params["params"]
    model = eqx_model
    if model.normalizer is not None and "normalizer" in p:
        np_ = p["normalizer"]
        norm = model.normalizer
        norm = eqx.tree_at(lambda m: m.gamma, norm, np_["gamma"])
        norm = eqx.tree_at(lambda m: m.beta, norm, np_["beta"])
        model = eqx.tree_at(lambda m: m.normalizer, model, norm)
    model = eqx.tree_at(
        lambda m: m.embedder_0,
        model,
        _transfer_dense(p["embedder_0"], model.embedder_0),
    )
    model = eqx.tree_at(
        lambda m: m.embedder_2,
        model,
        _transfer_dense(p["embedder_2"], model.embedder_2),
    )
    model = eqx.tree_at(
        lambda m: m.data_encoder,
        model,
        _transfer_data_transformer(p["data_encoder"], model.data_encoder),
    )
    model = eqx.tree_at(
        lambda m: m.text_encoder,
        model,
        _transfer_text_transformer(p["text_encoder"], model.text_encoder),
    )
    model = eqx.tree_at(
        lambda m: m.fusion,
        model,
        _transfer_fusion_transformer(p["fusion"], model.fusion),
    )
    model = eqx.tree_at(
        lambda m: m.data_decoder,
        model,
        _transfer_data_operator(p["data_decoder"], model.data_decoder),
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Small test configs
# ═══════════════════════════════════════════════════════════════════════════════

_X_NUM = 16
_DIM = 64
_FFN = 128
_NHEAD = 4
_DATA_DIM = 2
_PATCH = 4
_PATCH_OUT = 4
_N_WORDS = 32

# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPROSE1to1:
    def _make_flax(self):
        cfg = PROSE1to1Config(
            dim_emb=_DIM,
            dim_ffn=_FFN,
            n_head=_NHEAD,
            dropout=0.0,
            norm_first=True,
            patch_num=_PATCH,
            patch_num_output=_PATCH_OUT,
            time_embed="learnable",
            embedder=EmbedderConfig(
                type="conv",
                dim=_DIM,
                patch_num=_PATCH,
                patch_num_output=_PATCH_OUT,
                time_embed="learnable",
                max_time_len=10,
                conv_dim=8,
            ),
            data_encoder=DataEncoderConfig(
                n_layer=2,
                dim_emb=_DIM,
                dim_ffn=_FFN,
                n_head=_NHEAD,
                dropout=0.0,
                norm_first=True,
                norm="rms",
            ),
            data_decoder=DataDecoderConfig(
                n_layer=2,
                dim_emb=_DIM,
                dim_ffn=_FFN,
                n_head=_NHEAD,
                dropout=0.0,
                norm_first=True,
                patch_num_output=_PATCH_OUT,
                time_embed="learnable",
                max_time_len=10,
                final_ln=True,
                norm="rms",
            ),
        )
        return prose_fd_1to1(
            config=cfg,
            x_num=_X_NUM,
            max_output_dim=_DATA_DIM,
            input_len=2,
            output_len=2,
        )

    def _make_eqx(self):
        return EqxPROSE1to1(
            dim_emb=_DIM,
            dim_ffn=_FFN,
            n_head=_NHEAD,
            n_enc_layers=2,
            n_dec_layers=2,
            patch_num=_PATCH,
            patch_num_output=_PATCH_OUT,
            x_num=_X_NUM,
            max_output_dim=_DATA_DIM,
            norm_type="rms",
            norm_first=True,
            final_ln=True,
            time_embed_type="learnable",
            max_time_len=10,
            conv_dim=8,
            key=jax.random.PRNGKey(0),
        )

    def test_init(self):
        _, params = self._make_flax()
        assert "params" in params

    def test_eqx_init(self):
        model = self._make_eqx()
        assert isinstance(model, EqxPROSE1to1)

    def test_transfer(self):
        _, params = self._make_flax()
        eqx_model = self._make_eqx()
        eqx_model = transfer_weights_1to1(params, eqx_model)
        assert isinstance(eqx_model, EqxPROSE1to1)

    def test_forward_equivalence(self):
        flax_model, params = self._make_flax()
        eqx_model = transfer_weights_1to1(params, self._make_eqx())
        rng = jax.random.PRNGKey(42)
        data = jax.random.normal(rng, (1, 2, _X_NUM, _X_NUM, _DATA_DIM))
        tin = jax.random.uniform(jax.random.PRNGKey(1), (1, 2, 1))
        tout = jax.random.uniform(jax.random.PRNGKey(2), (1, 2, 1))

        flax_out = flax_model.apply(params, data, tin, tout, deterministic=True)
        eqx_out = eqx_model(data, tin, tout, deterministic=True)

        diff = float(jnp.max(jnp.abs(flax_out - eqx_out)))
        print(f"PROSE1to1 max diff: {diff:.2e}")
        assert diff < 1e-4, f"Max diff {diff:.2e} exceeds tolerance"


class TestPROSE2to1:
    def _make_flax(self):
        cfg = Prose2to1Config(
            dim_emb=_DIM,
            dim_ffn=_FFN,
            n_head=_NHEAD,
            patch_num=_PATCH,
            patch_num_output=_PATCH_OUT,
            data_encoder_layers=1,
            symbol_encoder_layers=1,
            fusion_layers=1,
            data_decoder_layers=1,
        )
        from jax_prose.prose_fd_2to1 import PROSE2to1 as Flax2to1

        model = Flax2to1(
            n_words=_N_WORDS, x_num=_X_NUM, max_output_dim=_DATA_DIM, cfg=cfg
        )
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((1, 2, _X_NUM, _X_NUM, _DATA_DIM))
        tin = jnp.zeros((1, 2, 1))
        tout = jnp.zeros((1, 2, 1))
        sym = jnp.zeros((1, 8), dtype=jnp.int32)
        sym_mask = jnp.zeros((1, 8), dtype=bool)
        params = model.init({"params": rng}, x, tin, tout, sym, sym_mask)
        return model, params

    def _make_eqx(self):
        return EqxPROSE2to1(
            n_words=_N_WORDS,
            x_num=_X_NUM,
            max_output_dim=_DATA_DIM,
            dim_emb=_DIM,
            dim_ffn=_FFN,
            n_head=_NHEAD,
            patch_num=_PATCH,
            patch_num_output=_PATCH_OUT,
            data_encoder_layers=1,
            symbol_encoder_layers=1,
            fusion_layers=1,
            data_decoder_layers=1,
            key=jax.random.PRNGKey(0),
        )

    def test_init(self):
        _, params = self._make_flax()
        assert "params" in params

    def test_eqx_init(self):
        model = self._make_eqx()
        assert isinstance(model, EqxPROSE2to1)

    def test_transfer(self):
        _, params = self._make_flax()
        eqx_model = transfer_weights_2to1(params, self._make_eqx())
        assert isinstance(eqx_model, EqxPROSE2to1)

    def test_forward_equivalence(self):
        flax_model, params = self._make_flax()
        eqx_model = transfer_weights_2to1(params, self._make_eqx())
        rng = jax.random.PRNGKey(42)
        data = jax.random.normal(rng, (1, 2, _X_NUM, _X_NUM, _DATA_DIM))
        tin = jax.random.uniform(jax.random.PRNGKey(1), (1, 2, 1))
        tout = jax.random.uniform(jax.random.PRNGKey(2), (1, 2, 1))
        sym = jax.random.randint(jax.random.PRNGKey(3), (1, 8), 0, _N_WORDS)
        sym_mask = jnp.zeros((1, 8), dtype=bool)

        flax_out = flax_model.apply(params, data, tin, tout, sym, sym_mask)
        eqx_out = eqx_model(data, tin, tout, sym, sym_mask)

        diff = float(jnp.max(jnp.abs(flax_out - eqx_out)))
        print(f"PROSE2to1 max diff: {diff:.2e}")
        assert diff < 1e-4, f"Max diff {diff:.2e} exceeds tolerance"


class TestPROSEODE2to1:
    def _make_flax(self):
        cfg = ProseTextData2to1Config(
            emb_dim=_DIM,
            n_text_enc_layers=1,
            n_data_enc_layers=1,
            n_data_dec_layers=1,
            n_fusion_layers=1,
            n_text_heads=_NHEAD,
            n_data_heads=_NHEAD,
            n_fusion_heads=_NHEAD,
            n_text_hidden_layers=1,
            n_data_hidden_layers=1,
            n_fusion_hidden_layers=1,
        )
        return prose_ode_2to1(
            n_words=_N_WORDS,
            pad_index=0,
            max_output_dimension=3,
            input_len=10,
            output_len=5,
            text_len=8,
            cfg=cfg,
        )

    def _make_eqx(self):
        return EqxPROSEODE2to1(
            n_words=_N_WORDS,
            pad_index=0,
            max_output_dimension=3,
            emb_dim=_DIM,
            n_text_enc_layers=1,
            n_data_enc_layers=1,
            n_data_dec_layers=1,
            n_fusion_layers=1,
            n_text_heads=_NHEAD,
            n_data_heads=_NHEAD,
            n_fusion_heads=_NHEAD,
            n_text_hidden_layers=1,
            n_data_hidden_layers=1,
            n_fusion_hidden_layers=1,
            key=jax.random.PRNGKey(0),
        )

    def test_init(self):
        _, params = self._make_flax()
        assert "params" in params

    def test_eqx_init(self):
        model = self._make_eqx()
        assert isinstance(model, EqxPROSEODE2to1)

    def test_transfer(self):
        _, params = self._make_flax()
        eqx_model = transfer_weights_ode(params, self._make_eqx())
        assert isinstance(eqx_model, EqxPROSEODE2to1)

    def test_forward_equivalence(self):
        flax_model, params = self._make_flax()
        eqx_model = transfer_weights_ode(params, self._make_eqx())
        rng = jax.random.PRNGKey(42)
        data = jax.random.normal(rng, (10, 1, 4))
        data_len = jnp.array([10], dtype=jnp.int32)
        query = jnp.linspace(0, 1, 5)
        text = jax.random.randint(jax.random.PRNGKey(3), (8, 1), 0, _N_WORDS)
        text_len = jnp.array([8], dtype=jnp.int32)

        flax_out = flax_model.apply(params, data, data_len, query, text, text_len)
        eqx_out = eqx_model(data, data_len, query, text, text_len)

        diff = float(jnp.max(jnp.abs(flax_out - eqx_out)))
        print(f"PROSEODE2to1 max diff: {diff:.2e}")
        assert diff < 1e-4, f"Max diff {diff:.2e} exceeds tolerance"


class TestPROSEPDE2to1:
    def _make_flax(self):
        cfg = ProseTextData2to1Config(
            emb_dim=_DIM,
            n_text_enc_layers=1,
            n_data_enc_layers=1,
            n_data_dec_layers=1,
            n_fusion_layers=1,
            n_text_heads=_NHEAD,
            n_data_heads=_NHEAD,
            n_fusion_heads=_NHEAD,
            n_text_hidden_layers=1,
            n_data_hidden_layers=1,
            n_fusion_hidden_layers=1,
            normalization=True,
        )
        return prose_pde_2to1(
            n_words=_N_WORDS,
            pad_index=0,
            max_output_dimension=1,
            x_patch_size=1,
            x_grid_size=4,
            input_len=10,
            output_len=5,
            text_len=8,
            cfg=cfg,
        )

    def _make_eqx(self):
        return EqxPROSEPDE2to1(
            n_words=_N_WORDS,
            pad_index=0,
            max_output_dimension=1,
            emb_dim=_DIM,
            n_text_enc_layers=1,
            n_data_enc_layers=1,
            n_data_dec_layers=1,
            n_fusion_layers=1,
            n_text_heads=_NHEAD,
            n_data_heads=_NHEAD,
            n_fusion_heads=_NHEAD,
            n_text_hidden_layers=1,
            n_data_hidden_layers=1,
            n_fusion_hidden_layers=1,
            x_grid_size=4,
            normalization=True,
            key=jax.random.PRNGKey(0),
        )

    def test_init(self):
        _, params = self._make_flax()
        assert "params" in params

    def test_eqx_init(self):
        model = self._make_eqx()
        assert isinstance(model, EqxPROSEPDE2to1)

    def test_transfer(self):
        _, params = self._make_flax()
        eqx_model = transfer_weights_pde(params, self._make_eqx())
        assert isinstance(eqx_model, EqxPROSEPDE2to1)

    def test_forward_equivalence(self):
        flax_model, params = self._make_flax()
        eqx_model = transfer_weights_pde(params, self._make_eqx())
        rng = jax.random.PRNGKey(42)
        in_dim = 1 + 1  # 1 + max_output_dimension * x_patch_size
        data = jax.random.normal(rng, (10, 1, in_dim))
        data_len = jnp.array([10], dtype=jnp.int32)
        query = jnp.linspace(0, 1, 5)
        text = jax.random.randint(jax.random.PRNGKey(3), (8, 1), 0, _N_WORDS)
        text_len = jnp.array([8], dtype=jnp.int32)

        flax_out = flax_model.apply(params, data, data_len, query, text, text_len)
        eqx_out = eqx_model(data, data_len, query, text, text_len)

        diff = float(jnp.max(jnp.abs(flax_out - eqx_out)))
        print(f"PROSEPDE2to1 max diff: {diff:.2e}")
        assert diff < 1e-4, f"Max diff {diff:.2e} exceeds tolerance"
