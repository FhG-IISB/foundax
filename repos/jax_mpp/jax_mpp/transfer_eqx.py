"""Transfer PyTorch AViT (MPP) state_dict into an Equinox AViT model.

The existing ``convert_weights.py`` targets a Flax-style parameter tree. The
Equinox model in ``avit_eqx.py`` uses its own field layout, so this module
walks the PT state_dict and assigns each tensor to the corresponding
Equinox leaf via ``eqx.tree_at``.

Layout notes:
    * eqx.nn.Conv2d / ConvTranspose2d weight is ``(out, in, kH, kW)``.
      PyTorch Conv2d uses the same layout, ConvTranspose2d uses
      ``(in, out, kH, kW)`` — needs ``transpose(1, 0, 2, 3)``.
    * eqx.nn.Conv2d bias is ``(out, 1, 1)``; PT bias is ``(out,)``.
    * eqx.nn.LayerNorm weight/bias match PT shapes ``(dim,)``.
    * eqx.nn.Linear.weight is ``(out, in)`` — matches PT.
    * RelativePositionBias has a single ``embedding`` leaf
      (PT key ``rel_pos_bias.relative_attention_bias.weight``).
    * ``debed.out_kernel`` is ``(out_chans, q, 4, 4)`` in Equinox,
      ``(q, n_states, 4, 4)`` in PT — transpose ``(1, 0, 2, 3)``.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from jax_mpp.avit_eqx import AViT


def _to_jnp(t: Any) -> jnp.ndarray:
    try:
        import torch

        if isinstance(t, torch.Tensor):
            return jnp.asarray(t.detach().cpu().numpy())
    except ImportError:
        pass
    return jnp.asarray(np.asarray(t))


def _set(model, where, value):
    return eqx.tree_at(where, model, value)


def _set_conv_weight(model, where, pt_w, *, transpose: bool = False):
    """Assign a PT conv weight to an Equinox conv leaf.

    ``transpose=True`` for ConvTranspose2d: PT stores ``(in, out, kH, kW)`` and
    Equinox stores ``(out, in, kH, kW)``, AND Equinox's ``ConvTranspose2d`` uses
    a spatially flipped kernel convention vs PyTorch — so we also flip the
    spatial dims.
    """
    arr = _to_jnp(pt_w)
    if transpose:
        arr = jnp.transpose(arr, (1, 0, 2, 3))[:, :, ::-1, ::-1]
    return _set(model, where, arr)


def _set_conv_bias(model, where, pt_b):
    arr = _to_jnp(pt_b).reshape(-1, 1, 1)
    return _set(model, where, arr)


def _set_norm(model, where_w, where_b, sd, prefix):
    model = _set(model, where_w, _to_jnp(sd[f"{prefix}.weight"]))
    if f"{prefix}.bias" in sd:
        model = _set(model, where_b, _to_jnp(sd[f"{prefix}.bias"]))
    return model


def transfer_pt_to_eqx(state_dict: dict, model: AViT) -> AViT:
    """Return ``model`` with all parameters replaced from ``state_dict``."""
    sd = state_dict

    # ---- space_bag (SubsampledLinear) ----
    model = _set(model, lambda m: m.space_bag.weight, _to_jnp(sd["space_bag.weight"]))
    model = _set(model, lambda m: m.space_bag.bias, _to_jnp(sd["space_bag.bias"]))

    # ---- embed (hMLP_stem) ----
    model = _set_conv_weight(
        model, lambda m: m.embed.in_proj_0.weight, sd["embed.in_proj.0.weight"]
    )
    model = _set_norm(
        model,
        lambda m: m.embed.in_proj_1.weight,
        lambda m: m.embed.in_proj_1.bias,
        sd,
        "embed.in_proj.1",
    )
    model = _set_conv_weight(
        model, lambda m: m.embed.in_proj_3.weight, sd["embed.in_proj.3.weight"]
    )
    model = _set_norm(
        model,
        lambda m: m.embed.in_proj_4.weight,
        lambda m: m.embed.in_proj_4.bias,
        sd,
        "embed.in_proj.4",
    )
    model = _set_conv_weight(
        model, lambda m: m.embed.in_proj_6.weight, sd["embed.in_proj.6.weight"]
    )
    model = _set_norm(
        model,
        lambda m: m.embed.in_proj_7.weight,
        lambda m: m.embed.in_proj_7.bias,
        sd,
        "embed.in_proj.7",
    )

    # ---- processor blocks ----
    n_blocks = len(model.blocks)
    for i in range(n_blocks):
        # temporal (AttentionBlock — no MLP, uses InstanceNorm2d, has `gamma`)
        prefix = f"blocks.{i}.temporal"
        model = _set_attention_block(
            model, i, "temporal", sd, prefix, has_mlp=False, gamma_keys=("gamma",)
        )
        # spatial (AxialAttentionBlock — has MLP, uses RMSInstanceNorm2d,
        # has `gamma_att` + `gamma_mlp`)
        prefix = f"blocks.{i}.spatial"
        model = _set_attention_block(
            model,
            i,
            "spatial",
            sd,
            prefix,
            has_mlp=True,
            gamma_keys=("gamma_att", "gamma_mlp"),
        )

    # ---- debed (hMLP_output) ----
    model = _set_conv_weight(
        model,
        lambda m: m.debed.out_proj_0.weight,
        sd["debed.out_proj.0.weight"],
        transpose=True,  # ConvTranspose2d: PT (in, out, kH, kW) -> Eqx (out, in, kH, kW)
    )
    model = _set_norm(
        model,
        lambda m: m.debed.out_proj_1.weight,
        lambda m: m.debed.out_proj_1.bias,
        sd,
        "debed.out_proj.1",
    )
    model = _set_conv_weight(
        model,
        lambda m: m.debed.out_proj_3.weight,
        sd["debed.out_proj.3.weight"],
        transpose=True,
    )
    model = _set_norm(
        model,
        lambda m: m.debed.out_proj_4.weight,
        lambda m: m.debed.out_proj_4.bias,
        sd,
        "debed.out_proj.4",
    )
    # debed.out_kernel: PT (q, n_states, 4, 4) -> Eqx (n_states, q, 4, 4)
    out_kernel_pt = _to_jnp(sd["debed.out_kernel"])
    model = _set(
        model, lambda m: m.debed.out_kernel, jnp.transpose(out_kernel_pt, (1, 0, 2, 3))
    )
    model = _set(model, lambda m: m.debed.out_bias, _to_jnp(sd["debed.out_bias"]))

    return model


def _set_attention_block(
    model: AViT,
    block_idx: int,
    branch: str,
    sd: dict,
    prefix: str,
    *,
    has_mlp: bool,
    gamma_keys: tuple,
) -> AViT:
    """Assign weights for one AttentionBlock/AxialAttentionBlock."""

    def _branch(m):
        return getattr(m.blocks[block_idx], branch)

    # norm1 (InstanceNorm2d or RMSInstanceNorm2d — both have .weight, .bias)
    model = _set(model, lambda m: _branch(m).norm1.weight, _to_jnp(sd[f"{prefix}.norm1.weight"]))
    model = _set(model, lambda m: _branch(m).norm1.bias, _to_jnp(sd[f"{prefix}.norm1.bias"]))

    # input_head (Conv2d 1x1)
    model = _set_conv_weight(
        model,
        lambda m: _branch(m).input_head.weight,
        sd[f"{prefix}.input_head.weight"],
    )
    model = _set_conv_bias(
        model, lambda m: _branch(m).input_head.bias, sd[f"{prefix}.input_head.bias"]
    )

    # qnorm, knorm (LayerNorm)
    model = _set(
        model, lambda m: _branch(m).qnorm.weight, _to_jnp(sd[f"{prefix}.qnorm.weight"])
    )
    model = _set(
        model, lambda m: _branch(m).qnorm.bias, _to_jnp(sd[f"{prefix}.qnorm.bias"])
    )
    model = _set(
        model, lambda m: _branch(m).knorm.weight, _to_jnp(sd[f"{prefix}.knorm.weight"])
    )
    model = _set(
        model, lambda m: _branch(m).knorm.bias, _to_jnp(sd[f"{prefix}.knorm.bias"])
    )

    # rel_pos_bias (RelativePositionBias.embedding)
    rpb_key = f"{prefix}.rel_pos_bias.relative_attention_bias.weight"
    if rpb_key in sd:
        model = _set(
            model,
            lambda m: _branch(m).rel_pos_bias.embedding,
            _to_jnp(sd[rpb_key]),
        )

    # norm2
    model = _set(
        model, lambda m: _branch(m).norm2.weight, _to_jnp(sd[f"{prefix}.norm2.weight"])
    )
    model = _set(
        model, lambda m: _branch(m).norm2.bias, _to_jnp(sd[f"{prefix}.norm2.bias"])
    )

    # output_head
    model = _set_conv_weight(
        model,
        lambda m: _branch(m).output_head.weight,
        sd[f"{prefix}.output_head.weight"],
    )
    model = _set_conv_bias(
        model, lambda m: _branch(m).output_head.bias, sd[f"{prefix}.output_head.bias"]
    )

    # gamma(s)
    for gk in gamma_keys:
        full_key = f"{prefix}.{gk}"
        if full_key in sd:
            model = _set(
                model, lambda m, gk=gk: getattr(_branch(m), gk), _to_jnp(sd[full_key])
            )

    # mlp + mlp_norm (axial only)
    if has_mlp:
        model = _set(
            model,
            lambda m: _branch(m).mlp.fc1.weight,
            _to_jnp(sd[f"{prefix}.mlp.fc1.weight"]),
        )
        model = _set(
            model,
            lambda m: _branch(m).mlp.fc1.bias,
            _to_jnp(sd[f"{prefix}.mlp.fc1.bias"]),
        )
        model = _set(
            model,
            lambda m: _branch(m).mlp.fc2.weight,
            _to_jnp(sd[f"{prefix}.mlp.fc2.weight"]),
        )
        model = _set(
            model,
            lambda m: _branch(m).mlp.fc2.bias,
            _to_jnp(sd[f"{prefix}.mlp.fc2.bias"]),
        )
        model = _set(
            model,
            lambda m: _branch(m).mlp_norm.weight,
            _to_jnp(sd[f"{prefix}.mlp_norm.weight"]),
        )
        model = _set(
            model,
            lambda m: _branch(m).mlp_norm.bias,
            _to_jnp(sd[f"{prefix}.mlp_norm.bias"]),
        )

    return model
