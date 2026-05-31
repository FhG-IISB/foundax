"""Transfer PyTorch Poseidon (scOT) state_dict into Equinox ScOT.

The PT model from ``scOT.model.ScOT`` (cloned ``camlab-ethz/poseidon`` repo)
has 5 top-level submodules: ``embeddings``, ``encoder``, ``decoder``,
``patch_recovery``, ``residual_blocks``. Each layer follows HuggingFace
Swinv2 conventions. The Equinox model mirrors the structure 1:1 but uses
slightly different field names (notably ``downsample_layer``/``upsample_layer``,
``output_layer``, ``key_proj``, ``proj``) and stores ConditionalLayerNorm gain/
bias linears under ``weight_dense``/``bias_dense``.

Layout (see also memory/equinox_pt_weight_transfer_gotchas.md):
    * eqx.nn.Linear.weight is (out, in) — matches PT.
    * Conv2dNHWC.conv is a wrapped eqx.nn.Conv2d with (out, in, kH, kW) weight
      and (out, 1, 1) bias.
    * ConvTranspose2dNHWC.conv is eqx.nn.ConvTranspose2d with (out, in, kH, kW)
      weight that requires PT-to-Eqx transpose + spatial flip.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np


def _to_jnp(t):
    try:
        import torch

        if isinstance(t, torch.Tensor):
            return jnp.asarray(t.detach().cpu().numpy())
    except ImportError:
        pass
    return jnp.asarray(np.asarray(t))


def _set(model, where, value):
    return eqx.tree_at(where, model, value)


def _conv_bias(arr):
    return _to_jnp(arr).reshape(-1, 1, 1)


def _ct_weight(arr):
    return _to_jnp(arr).transpose(1, 0, 2, 3)[:, :, ::-1, ::-1]


# ---------------------------------------------------------------------------
# ConditionalLayerNorm: PT `norm.{weight,bias}.{weight,bias}` (Linear(1, dim)
# each) -> JAX `norm.{weight_dense,bias_dense}.{weight,bias}`.
# ---------------------------------------------------------------------------
def _set_clayernorm(model, norm_fn, sd, prefix):
    """``prefix`` is everything up to (and including) the norm path, e.g.
    ``encoder.layers.0.downsample.norm`` — without trailing dot."""
    model = _set(
        model,
        lambda m: norm_fn(m).weight_dense.weight,
        _to_jnp(sd[f"{prefix}.weight.weight"]),
    )
    model = _set(
        model,
        lambda m: norm_fn(m).weight_dense.bias,
        _to_jnp(sd[f"{prefix}.weight.bias"]),
    )
    model = _set(
        model,
        lambda m: norm_fn(m).bias_dense.weight,
        _to_jnp(sd[f"{prefix}.bias.weight"]),
    )
    model = _set(
        model,
        lambda m: norm_fn(m).bias_dense.bias,
        _to_jnp(sd[f"{prefix}.bias.bias"]),
    )
    return model


def _set_swin_attention(model, layer_fn, sd, prefix):
    """One ScOTLayer's attention sub-tree (no LN, that's external)."""

    def _attn(m):
        return layer_fn(m).attention

    # self.{query,key,value} (in JAX: query, key_proj, value)
    model = _set(
        model,
        lambda m: _attn(m).query.weight,
        _to_jnp(sd[f"{prefix}.attention.self.query.weight"]),
    )
    model = _set(
        model,
        lambda m: _attn(m).query.bias,
        _to_jnp(sd[f"{prefix}.attention.self.query.bias"]),
    )
    model = _set(
        model,
        lambda m: _attn(m).key_proj.weight,
        _to_jnp(sd[f"{prefix}.attention.self.key.weight"]),
    )
    # PT 'key' has no bias by default in Swinv2 (use_bias=False in JAX), so skip
    model = _set(
        model,
        lambda m: _attn(m).value.weight,
        _to_jnp(sd[f"{prefix}.attention.self.value.weight"]),
    )
    model = _set(
        model,
        lambda m: _attn(m).value.bias,
        _to_jnp(sd[f"{prefix}.attention.self.value.bias"]),
    )
    # output.dense -> proj
    model = _set(
        model,
        lambda m: _attn(m).proj.weight,
        _to_jnp(sd[f"{prefix}.attention.output.dense.weight"]),
    )
    model = _set(
        model,
        lambda m: _attn(m).proj.bias,
        _to_jnp(sd[f"{prefix}.attention.output.dense.bias"]),
    )
    # logit_scale
    model = _set(
        model,
        lambda m: _attn(m).logit_scale,
        _to_jnp(sd[f"{prefix}.attention.self.logit_scale"]),
    )
    # continuous_position_bias_mlp.0 -> cpb_mlp_0; index 2 -> cpb_mlp_1
    model = _set(
        model,
        lambda m: _attn(m).relative_position_bias.cpb_mlp_0.weight,
        _to_jnp(sd[f"{prefix}.attention.self.continuous_position_bias_mlp.0.weight"]),
    )
    model = _set(
        model,
        lambda m: _attn(m).relative_position_bias.cpb_mlp_0.bias,
        _to_jnp(sd[f"{prefix}.attention.self.continuous_position_bias_mlp.0.bias"]),
    )
    model = _set(
        model,
        lambda m: _attn(m).relative_position_bias.cpb_mlp_1.weight,
        _to_jnp(sd[f"{prefix}.attention.self.continuous_position_bias_mlp.2.weight"]),
    )
    # cpb_mlp.2 has no bias in PT Swinv2 (final projection to num_heads)
    return model


def _set_scot_layer(model, layer_fn, sd, prefix):
    """Full ScOTLayer: attention + 2 conditional LNs + intermediate + output."""
    model = _set_swin_attention(model, layer_fn, sd, prefix)
    model = _set_clayernorm(
        model, lambda m: layer_fn(m).layernorm_before, sd, f"{prefix}.layernorm_before"
    )
    model = _set_clayernorm(
        model, lambda m: layer_fn(m).layernorm_after, sd, f"{prefix}.layernorm_after"
    )
    model = _set(
        model,
        lambda m: layer_fn(m).intermediate.dense.weight,
        _to_jnp(sd[f"{prefix}.intermediate.dense.weight"]),
    )
    model = _set(
        model,
        lambda m: layer_fn(m).intermediate.dense.bias,
        _to_jnp(sd[f"{prefix}.intermediate.dense.bias"]),
    )
    model = _set(
        model,
        lambda m: layer_fn(m).output_layer.dense.weight,
        _to_jnp(sd[f"{prefix}.output.dense.weight"]),
    )
    model = _set(
        model,
        lambda m: layer_fn(m).output_layer.dense.bias,
        _to_jnp(sd[f"{prefix}.output.dense.bias"]),
    )
    return model


def _set_convnext_block(model, blk_fn, sd, prefix):
    """ConvNeXtBlock (residual_blocks.N.M.*)."""
    # gamma (layer scale weight)
    if f"{prefix}.weight" in sd:
        model = _set(model, lambda m: blk_fn(m).weight, _to_jnp(sd[f"{prefix}.weight"]))
    # dwconv (Conv2dNHWC wraps eqx.nn.Conv2d under .conv)
    model = _set(
        model,
        lambda m: blk_fn(m).dwconv.conv.weight,
        _to_jnp(sd[f"{prefix}.dwconv.weight"]),
    )
    model = _set(
        model,
        lambda m: blk_fn(m).dwconv.conv.bias,
        _conv_bias(sd[f"{prefix}.dwconv.bias"]),
    )
    # norm (conditional)
    model = _set_clayernorm(model, lambda m: blk_fn(m).norm, sd, f"{prefix}.norm")
    # pwconv1 / pwconv2 (Linear)
    model = _set(
        model,
        lambda m: blk_fn(m).pwconv1.weight,
        _to_jnp(sd[f"{prefix}.pwconv1.weight"]),
    )
    model = _set(
        model, lambda m: blk_fn(m).pwconv1.bias, _to_jnp(sd[f"{prefix}.pwconv1.bias"])
    )
    model = _set(
        model,
        lambda m: blk_fn(m).pwconv2.weight,
        _to_jnp(sd[f"{prefix}.pwconv2.weight"]),
    )
    model = _set(
        model, lambda m: blk_fn(m).pwconv2.bias, _to_jnp(sd[f"{prefix}.pwconv2.bias"])
    )
    return model


def transfer_pt_to_eqx(state_dict: dict, model):
    """Return ``model`` with all params replaced from PT scOT state_dict."""
    sd = state_dict

    # ---- embeddings ----
    model = _set(
        model,
        lambda m: m.embeddings.patch_embeddings.projection.conv.weight,
        _to_jnp(sd["embeddings.patch_embeddings.projection.weight"]),
    )
    model = _set(
        model,
        lambda m: m.embeddings.patch_embeddings.projection.conv.bias,
        _conv_bias(sd["embeddings.patch_embeddings.projection.bias"]),
    )
    model = _set_clayernorm(model, lambda m: m.embeddings.norm, sd, "embeddings.norm")

    # ---- encoder ----
    for li, stage in enumerate(model.encoder.layers):
        for bi in range(len(stage.blocks)):
            prefix = f"encoder.layers.{li}.blocks.{bi}"

            def _layer_fn(m, li=li, bi=bi):
                return m.encoder.layers[li].blocks[bi]

            model = _set_scot_layer(model, _layer_fn, sd, prefix)
        # downsample (PatchMerging)
        ds_prefix = f"encoder.layers.{li}.downsample"
        if f"{ds_prefix}.reduction.weight" in sd:
            model = _set(
                model,
                lambda m, li=li: m.encoder.layers[li].downsample_layer.reduction.weight,
                _to_jnp(sd[f"{ds_prefix}.reduction.weight"]),
            )
            model = _set_clayernorm(
                model,
                lambda m, li=li: m.encoder.layers[li].downsample_layer.norm,
                sd,
                f"{ds_prefix}.norm",
            )

    # ---- decoder ----
    for li, stage in enumerate(model.decoder.layers):
        for bi in range(len(stage.blocks)):
            prefix = f"decoder.layers.{li}.blocks.{bi}"

            def _layer_fn(m, li=li, bi=bi):
                return m.decoder.layers[li].blocks[bi]

            model = _set_scot_layer(model, _layer_fn, sd, prefix)
        # upsample (PatchUnmerging)
        us_prefix = f"decoder.layers.{li}.upsample"
        if f"{us_prefix}.upsample.weight" in sd:
            model = _set(
                model,
                lambda m, li=li: m.decoder.layers[li].upsample_layer.upsample.weight,
                _to_jnp(sd[f"{us_prefix}.upsample.weight"]),
            )
            model = _set(
                model,
                lambda m, li=li: m.decoder.layers[li].upsample_layer.mixup.weight,
                _to_jnp(sd[f"{us_prefix}.mixup.weight"]),
            )
            model = _set_clayernorm(
                model,
                lambda m, li=li: m.decoder.layers[li].upsample_layer.norm,
                sd,
                f"{us_prefix}.norm",
            )

    # ---- residual_blocks (list of ResidualBlockWrapper, each wrapping
    # a `.blocks` list of ConvNeXtBlock) ----
    for i, wrapper in enumerate(model.residual_blocks):
        if wrapper.blocks is None:
            continue
        for j in range(len(wrapper.blocks)):
            prefix = f"residual_blocks.{i}.{j}"

            def _blk_fn(m, i=i, j=j):
                return m.residual_blocks[i].blocks[j]

            model = _set_convnext_block(model, _blk_fn, sd, prefix)

    # ---- patch_recovery ----
    model = _set(
        model,
        lambda m: m.patch_recovery.projection.conv.weight,
        _ct_weight(sd["patch_recovery.projection.weight"]),
    )
    model = _set(
        model,
        lambda m: m.patch_recovery.projection.conv.bias,
        _conv_bias(sd["patch_recovery.projection.bias"]),
    )
    # mixup is a Conv2dNHWC without bias
    model = _set(
        model,
        lambda m: m.patch_recovery.mixup.conv.weight,
        _to_jnp(sd["patch_recovery.mixup.weight"]),
    )

    return model
