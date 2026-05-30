"""Transfer PyTorch BCAT state_dict into an Equinox BCAT model.

PT model is built by `models.bcat.BCAT` in the cloned `felix-lyx/bcat` repo
(see `og_repos/bcat/src`). It has two top-level submodules: ``embedder`` and
``transformer``.

Layout (see also memory/equinox_pt_weight_transfer_gotchas.md):
    * eqx.nn.Linear.weight is (out, in) — matches PT.
    * eqx.nn.Conv2d.weight is (out, in, kH, kW) — matches PT; bias is reshaped
      to (out, 1, 1).
    * eqx.nn.ConvTranspose2d.weight is (out, in, kH, kW); PT is (in, out, kH, kW)
      and must be spatially flipped.
    * RMSNorm stores ``weight`` (not ``scale``) — PT key ends in ``.weight``.
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


def transfer_pt_to_eqx(state_dict: dict, model):
    """Return ``model`` with all parameters replaced from PT state_dict."""
    sd = state_dict

    # ---- embedder ----
    model = _set(
        model,
        lambda m: m.embedder.patch_position_embeddings,
        _to_jnp(sd["embedder.patch_position_embeddings"]),
    )
    if "embedder.time_embeddings" in sd:
        model = _set(
            model,
            lambda m: m.embedder.time_embeddings,
            _to_jnp(sd["embedder.time_embeddings"]),
        )

    # in_proj (Conv2d)
    model = _set(
        model, lambda m: m.embedder.in_proj.weight, _to_jnp(sd["embedder.in_proj.weight"])
    )
    model = _set(
        model, lambda m: m.embedder.in_proj.bias, _conv_bias(sd["embedder.in_proj.bias"])
    )

    # conv_proj.1 (1x1 Conv2d) -> conv_proj
    model = _set(
        model,
        lambda m: m.embedder.conv_proj.weight,
        _to_jnp(sd["embedder.conv_proj.1.weight"]),
    )
    model = _set(
        model,
        lambda m: m.embedder.conv_proj.bias,
        _conv_bias(sd["embedder.conv_proj.1.bias"]),
    )

    # post_proj.1 (ConvTranspose2d) -> post_deconv
    model = _set(
        model,
        lambda m: m.embedder.post_deconv.weight,
        _ct_weight(sd["embedder.post_proj.1.weight"]),
    )
    model = _set(
        model,
        lambda m: m.embedder.post_deconv.bias,
        _conv_bias(sd["embedder.post_proj.1.bias"]),
    )
    # post_proj.3 -> post_conv
    model = _set(
        model,
        lambda m: m.embedder.post_conv.weight,
        _to_jnp(sd["embedder.post_proj.3.weight"]),
    )
    model = _set(
        model,
        lambda m: m.embedder.post_conv.bias,
        _conv_bias(sd["embedder.post_proj.3.bias"]),
    )
    # head
    model = _set(
        model, lambda m: m.embedder.head.weight, _to_jnp(sd["embedder.head.weight"])
    )
    model = _set(
        model, lambda m: m.embedder.head.bias, _conv_bias(sd["embedder.head.bias"])
    )

    # ---- transformer ----
    n_layers = len(model.transformer.layers)
    for i in range(n_layers):
        prefix = f"transformer.layers.{i}"

        def _layer(m, i=i):
            return m.transformer.layers[i]

        # self_attn (Q, K, V, out_proj)
        for q in ("linear_q", "linear_k", "linear_v", "out_proj"):
            model = _set(
                model,
                lambda m, q=q, i=i: getattr(_layer(m).self_attn, q).weight,
                _to_jnp(sd[f"{prefix}.self_attn.{q}.weight"]),
            )
            model = _set(
                model,
                lambda m, q=q, i=i: getattr(_layer(m).self_attn, q).bias,
                _to_jnp(sd[f"{prefix}.self_attn.{q}.bias"]),
            )

        # qk_norm (optional LayerNorms)
        for n in ("q_norm", "k_norm"):
            wk = f"{prefix}.self_attn.{n}.weight"
            bk = f"{prefix}.self_attn.{n}.bias"
            if wk in sd:
                model = _set(
                    model,
                    lambda m, n=n, i=i: getattr(_layer(m).self_attn, n).weight,
                    _to_jnp(sd[wk]),
                )
                model = _set(
                    model,
                    lambda m, n=n, i=i: getattr(_layer(m).self_attn, n).bias,
                    _to_jnp(sd[bk]),
                )

        # FFN (gated MLP: fc1, fc2, fc_gate)
        for fc in ("fc1", "fc2", "fc_gate"):
            wk = f"{prefix}.ffn.{fc}.weight"
            bk = f"{prefix}.ffn.{fc}.bias"
            if wk in sd:
                model = _set(
                    model,
                    lambda m, fc=fc, i=i: getattr(_layer(m).ffn, fc).weight,
                    _to_jnp(sd[wk]),
                )
                model = _set(
                    model,
                    lambda m, fc=fc, i=i: getattr(_layer(m).ffn, fc).bias,
                    _to_jnp(sd[bk]),
                )

        # norm1 / norm2 (RMSNorm: only .weight)
        model = _set(
            model,
            lambda m, i=i: _layer(m).norm1.weight,
            _to_jnp(sd[f"{prefix}.norm1.weight"]),
        )
        model = _set(
            model,
            lambda m, i=i: _layer(m).norm2.weight,
            _to_jnp(sd[f"{prefix}.norm2.weight"]),
        )

    # Final norm (RMSNorm)
    if "transformer.norm.weight" in sd:
        model = _set(
            model,
            lambda m: m.transformer.norm.weight,
            _to_jnp(sd["transformer.norm.weight"]),
        )

    return model
