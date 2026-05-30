"""Transfer PyTorch PROSE-FD state_dict into an Equinox PROSE2to1 model.

The PT model is built by ``PROSE_2to1`` in
``og_repos/prose/prose_fd/models/transformer_wrappers.py`` and has 5 top-level
submodules: ``embedder``, ``data_encoder``, ``symbol_encoder``, ``fusion``,
``data_decoder``. The JAX Equinox model mirrors them closely; this module maps
each PT key to its Equinox leaf.

Layout notes (see also memory/equinox_pt_weight_transfer_gotchas.md):
    * ``eqx.nn.Linear.weight`` is ``(out, in)`` — matches PT.
    * ``eqx.nn.Conv2d.weight`` is ``(out, in, kH, kW)`` — matches PT;
      bias is reshaped to ``(out, 1, 1)``.
    * ``eqx.nn.ConvTranspose2d.weight`` is ``(out, in, kH, kW)`` and must be
      spatially flipped vs PT, which stores ``(in, out, kH, kW)``.
    * RMSNormScale stores ``scale`` (not ``weight``). PT key ends in ``.weight``.
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


def _strip_prefix(k: str) -> str:
    for p in ("module._orig_mod.", "module."):
        if k.startswith(p):
            return k[len(p) :]
    return k


def _set(model, where, value):
    return eqx.tree_at(where, model, value)


def _conv_bias(arr):
    """PT conv bias (out,) -> Equinox (out, 1, 1)."""
    return _to_jnp(arr).reshape(-1, 1, 1)


def _ct_weight(arr):
    """PT ConvTranspose2d (in, out, kH, kW) -> Equinox (out, in, kH, kW) with spatial flip."""
    return _to_jnp(arr).transpose(1, 0, 2, 3)[:, :, ::-1, ::-1]


def transfer_pt_to_eqx(state_dict: dict, model):
    """Return ``model`` with all parameters replaced from PT state_dict.

    ``model`` must be a ``PROSE2to1`` instance.
    """
    sd = {_strip_prefix(k): v for k, v in state_dict.items()}

    # ---- embedder ----
    model = _set(
        model,
        lambda m: m.embedder.patch_position_embeddings,
        _to_jnp(sd["embedder.patch_position_embeddings"]),
    )
    model = _set(
        model, lambda m: m.embedder.time_embed, _to_jnp(sd["embedder.time_embed"])
    )
    # conv_proj.0 -> conv_proj_0 (Conv2d, has bias)
    model = _set(
        model,
        lambda m: m.embedder.conv_proj_0.weight,
        _to_jnp(sd["embedder.conv_proj.0.weight"]),
    )
    model = _set(
        model,
        lambda m: m.embedder.conv_proj_0.bias,
        _conv_bias(sd["embedder.conv_proj.0.bias"]),
    )
    # conv_proj.2 -> conv_proj_1 (1x1 Conv2d)
    model = _set(
        model,
        lambda m: m.embedder.conv_proj_1.weight,
        _to_jnp(sd["embedder.conv_proj.2.weight"]),
    )
    model = _set(
        model,
        lambda m: m.embedder.conv_proj_1.bias,
        _conv_bias(sd["embedder.conv_proj.2.bias"]),
    )
    # post_proj.1 -> deconv (ConvTranspose2d)
    model = _set(
        model,
        lambda m: m.embedder.deconv.weight,
        _ct_weight(sd["embedder.post_proj.1.weight"]),
    )
    model = _set(
        model,
        lambda m: m.embedder.deconv.bias,
        _conv_bias(sd["embedder.post_proj.1.bias"]),
    )
    # post_proj.3 -> post_conv_0
    model = _set(
        model,
        lambda m: m.embedder.post_conv_0.weight,
        _to_jnp(sd["embedder.post_proj.3.weight"]),
    )
    model = _set(
        model,
        lambda m: m.embedder.post_conv_0.bias,
        _conv_bias(sd["embedder.post_proj.3.bias"]),
    )
    # post_proj.5 -> post_conv_1
    model = _set(
        model,
        lambda m: m.embedder.post_conv_1.weight,
        _to_jnp(sd["embedder.post_proj.5.weight"]),
    )
    model = _set(
        model,
        lambda m: m.embedder.post_conv_1.bias,
        _conv_bias(sd["embedder.post_proj.5.bias"]),
    )

    # ---- data_encoder ----
    model = _set_encoder(
        model,
        lambda m: m.data_encoder,
        sd,
        "data_encoder.transformer_encoder",
        attn_name="self_attn",
    )

    # ---- symbol_encoder ----
    # word_embeddings.weight: (n_words, dim)
    model = _set(
        model,
        lambda m: m.symbol_encoder.word_embeddings.weight,
        _to_jnp(sd["symbol_encoder.word_embeddings.weight"]),
    )
    # PT positional_embedding.pe is (max_len, 1, dim); JAX pe is the same shape
    model = _set(
        model,
        lambda m: m.symbol_encoder.pe,
        _to_jnp(sd["symbol_encoder.positional_embedding.pe"]),
    )
    model = _set_encoder(
        model,
        lambda m: m.symbol_encoder.transformer_encoder,
        sd,
        "symbol_encoder.transformer_encoder",
        attn_name="self_attn",
    )

    # ---- fusion ----
    model = _set(
        model,
        lambda m: m.fusion.type_embeddings.weight,
        _to_jnp(sd["fusion.type_embeddings.weight"]),
    )
    model = _set_encoder(
        model,
        lambda m: m.fusion.transformer_encoder,
        sd,
        "fusion.transformer_encoder",
        attn_name="self_attn",
    )

    # ---- data_decoder ----
    model = _set(
        model,
        lambda m: m.data_decoder.time_embed,
        _to_jnp(sd["data_decoder.time_embed"]),
    )
    model = _set(
        model,
        lambda m: m.data_decoder.patch_position_embeddings,
        _to_jnp(sd["data_decoder.patch_position_embeddings"]),
    )
    model = _set_decoder(
        model, sd, "data_decoder.transformer_decoder", attn_name="multihead_attn"
    )

    return model


def _set_encoder(model, enc_path, sd, prefix, *, attn_name):
    """Assign weights for an Encoder2to1 (transformer_encoder.layers + norm)."""

    def _enc(m):
        return enc_path(m)

    # layers
    n_layers = len(_enc(model).layers)
    for i in range(n_layers):
        layer_prefix = f"{prefix}.layers.{i}"

        def _layer(m, i=i):
            return enc_path(m).layers[i]

        model = _set_attention_layer(
            model, _layer, sd, layer_prefix, attn_name=attn_name
        )

    # final norm: PT `norm.weight` -> JAX `norm.scale`
    model = _set(
        model, lambda m: enc_path(m).norm.scale, _to_jnp(sd[f"{prefix}.norm.weight"])
    )
    return model


def _set_decoder(model, sd, prefix, *, attn_name):
    """Assign weights for a DataDecoder2to1 transformer_decoder (layers + norm)."""

    def _dec_layers(m):
        return m.data_decoder.layers

    n_layers = len(_dec_layers(model))
    for i in range(n_layers):
        layer_prefix = f"{prefix}.layers.{i}"

        def _layer(m, i=i):
            return m.data_decoder.layers[i]

        model = _set_attention_layer(
            model, _layer, sd, layer_prefix, attn_name=attn_name
        )

    model = _set(
        model,
        lambda m: m.data_decoder.norm.scale,
        _to_jnp(sd[f"{prefix}.norm.weight"]),
    )
    return model


def _set_attention_layer(model, layer_fn, sd, prefix, *, attn_name):
    """Assign weights for one EncoderLayer2to1 or OperatorDecoderLayer2to1."""
    # MHA
    def _attn(m):
        return getattr(layer_fn(m), attn_name)

    for q_name in ("linear_q", "linear_k", "linear_v", "out_proj"):
        model = _set(
            model,
            lambda m, q_name=q_name: getattr(_attn(m), q_name).weight,
            _to_jnp(sd[f"{prefix}.{attn_name}.{q_name}.weight"]),
        )
        model = _set(
            model,
            lambda m, q_name=q_name: getattr(_attn(m), q_name).bias,
            _to_jnp(sd[f"{prefix}.{attn_name}.{q_name}.bias"]),
        )

    # FFN
    model = _set(
        model,
        lambda m: layer_fn(m).linear1.weight,
        _to_jnp(sd[f"{prefix}.linear1.weight"]),
    )
    model = _set(
        model,
        lambda m: layer_fn(m).linear1.bias,
        _to_jnp(sd[f"{prefix}.linear1.bias"]),
    )
    model = _set(
        model,
        lambda m: layer_fn(m).linear2.weight,
        _to_jnp(sd[f"{prefix}.linear2.weight"]),
    )
    model = _set(
        model,
        lambda m: layer_fn(m).linear2.bias,
        _to_jnp(sd[f"{prefix}.linear2.bias"]),
    )

    # Norms (PT .weight -> JAX .scale on RMSNormScale)
    model = _set(
        model,
        lambda m: layer_fn(m).norm1.scale,
        _to_jnp(sd[f"{prefix}.norm1.weight"]),
    )
    model = _set(
        model,
        lambda m: layer_fn(m).norm2.scale,
        _to_jnp(sd[f"{prefix}.norm2.weight"]),
    )

    return model
