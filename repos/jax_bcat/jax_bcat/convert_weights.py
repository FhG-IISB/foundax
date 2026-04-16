from __future__ import annotations

from typing import Any, Dict

import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict


def load_pytorch_state_dict(checkpoint_path: str) -> Dict[str, Any]:
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    out = {}
    for k, v in state_dict.items():
        nk = k
        # Strip DDP / compile wrappers
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        out[nk] = v
    return out


def convert_pytorch_to_jax_params(
    pt_state_dict: Dict[str, Any], jax_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Map a PyTorch BCAT state dict into a Flax parameter tree.

    The *jax_params* tree (from ``model.init(...)``) is used as the
    target shape reference.
    """
    pt: Dict[str, np.ndarray] = {}
    for k, v in pt_state_dict.items():
        arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        pt[k] = arr

    target_flat = flatten_dict(jax_params, sep=".")
    out_flat: Dict[str, np.ndarray] = {}

    def set_param(dst: str, src: str, transform=None) -> None:
        if src not in pt:
            raise KeyError(f"Missing source key: {src}")
        value = pt[src]
        if transform is not None:
            value = transform(value)
        expected = target_flat[dst].shape
        if value.shape != expected:
            raise ValueError(
                f"Shape mismatch for {dst}: got {value.shape}, expected {expected} (from {src})"
            )
        out_flat[dst] = value

    # -----------------------------------------------------------------------
    # Embedder
    # -----------------------------------------------------------------------

    # in_proj: Conv2d(data_dim, dim, kernel=patch_res, stride=patch_res)
    # PyTorch conv: (out, in, kH, kW) -> JAX conv: (kH, kW, in, out)
    set_param(
        "embedder.in_proj.kernel",
        "embedder.in_proj.weight",
        lambda x: np.transpose(x, (2, 3, 1, 0)),
    )
    set_param("embedder.in_proj.bias", "embedder.in_proj.bias")

    # conv_proj: Sequential(GELU, Conv2d(dim, dim, 1, 1))
    # The GELU has no params; the conv is at index 1 in PT Sequential
    set_param(
        "embedder.conv_proj.kernel",
        "embedder.conv_proj.1.weight",
        lambda x: np.transpose(x, (2, 3, 1, 0)),
    )
    set_param("embedder.conv_proj.bias", "embedder.conv_proj.1.bias")

    # Learnable time embeddings
    if "embedder.time_embeddings" in pt:
        set_param(
            "embedder.time_embeddings",
            "embedder.time_embeddings",
        )
    elif "embedder.time_proj.0.weight" in pt:
        # continuous time embeddings
        set_param(
            "embedder.time_proj_0.kernel",
            "embedder.time_proj.0.weight",
            lambda x: np.transpose(x, (1, 0)),
        )
        set_param("embedder.time_proj_0.bias", "embedder.time_proj.0.bias")
        set_param(
            "embedder.time_proj_1.kernel",
            "embedder.time_proj.2.weight",
            lambda x: np.transpose(x, (1, 0)),
        )
        set_param("embedder.time_proj_1.bias", "embedder.time_proj.2.bias")

    # Patch position embeddings
    set_param(
        "embedder.patch_position_embeddings",
        "embedder.patch_position_embeddings",
    )

    # Decoder: post_proj Sequential
    # The post_proj in PyTorch (non-deep) is:
    #   Rearrange(...),                    # index 0 — no params
    #   ConvTranspose2d(dim, conv_dim, patch_res, patch_res),  # index 1
    #   GELU(),                            # index 2
    #   Conv2d(conv_dim, conv_dim, 1, 1),  # index 3
    #   GELU(),                            # index 4
    #
    # JAX names: post_deconv, post_conv

    # ConvTranspose2d: PT (in, out, kH, kW) -> JAX ConvTranspose w/ transpose_kernel=True (kH, kW, out, in)
    set_param(
        "embedder.post_deconv.kernel",
        "embedder.post_proj.1.weight",
        lambda x: np.transpose(x, (2, 3, 1, 0)),
    )
    set_param("embedder.post_deconv.bias", "embedder.post_proj.1.bias")

    # Conv2d: PT (out, in, kH, kW) -> JAX (kH, kW, in, out)
    set_param(
        "embedder.post_conv.kernel",
        "embedder.post_proj.3.weight",
        lambda x: np.transpose(x, (2, 3, 1, 0)),
    )
    set_param("embedder.post_conv.bias", "embedder.post_proj.3.bias")

    # head: Conv2d(conv_dim, data_dim, 1, 1)
    set_param(
        "embedder.head.kernel",
        "embedder.head.weight",
        lambda x: np.transpose(x, (2, 3, 1, 0)),
    )
    set_param("embedder.head.bias", "embedder.head.bias")

    # -----------------------------------------------------------------------
    # Transformer layers
    # -----------------------------------------------------------------------
    n_layer = 0
    while f"transformer.layers.{n_layer}.self_attn.linear_q.weight" in pt:
        n_layer += 1

    for i in range(n_layer):
        pt_pfx = f"transformer.layers.{i}"
        jax_pfx = f"transformer.layers_{i}"

        # Self-attention Q/K/V/Out projections
        for proj in ("linear_q", "linear_k", "linear_v", "out_proj"):
            set_param(
                f"{jax_pfx}.self_attn.{proj}.kernel",
                f"{pt_pfx}.self_attn.{proj}.weight",
                lambda x: np.transpose(x, (1, 0)),
            )
            set_param(
                f"{jax_pfx}.self_attn.{proj}.bias",
                f"{pt_pfx}.self_attn.{proj}.bias",
            )

        # QK norm (LayerNorm on head dim)
        if f"{pt_pfx}.self_attn.q_norm.weight" in pt:
            set_param(
                f"{jax_pfx}.self_attn.q_norm.scale",
                f"{pt_pfx}.self_attn.q_norm.weight",
            )
            set_param(
                f"{jax_pfx}.self_attn.q_norm.bias",
                f"{pt_pfx}.self_attn.q_norm.bias",
            )
            set_param(
                f"{jax_pfx}.self_attn.k_norm.scale",
                f"{pt_pfx}.self_attn.k_norm.weight",
            )
            set_param(
                f"{jax_pfx}.self_attn.k_norm.bias",
                f"{pt_pfx}.self_attn.k_norm.bias",
            )

        # FFN: fc1, fc2, and optional fc_gate
        set_param(
            f"{jax_pfx}.ffn.fc1.kernel",
            f"{pt_pfx}.ffn.fc1.weight",
            lambda x: np.transpose(x, (1, 0)),
        )
        set_param(f"{jax_pfx}.ffn.fc1.bias", f"{pt_pfx}.ffn.fc1.bias")

        set_param(
            f"{jax_pfx}.ffn.fc2.kernel",
            f"{pt_pfx}.ffn.fc2.weight",
            lambda x: np.transpose(x, (1, 0)),
        )
        set_param(f"{jax_pfx}.ffn.fc2.bias", f"{pt_pfx}.ffn.fc2.bias")

        if f"{pt_pfx}.ffn.fc_gate.weight" in pt:
            set_param(
                f"{jax_pfx}.ffn.fc_gate.kernel",
                f"{pt_pfx}.ffn.fc_gate.weight",
                lambda x: np.transpose(x, (1, 0)),
            )
            set_param(f"{jax_pfx}.ffn.fc_gate.bias", f"{pt_pfx}.ffn.fc_gate.bias")

        # Norms (RMSNorm or LayerNorm)
        # In PT: norm1.weight -> In JAX RMSNorm: norm1.scale
        for norm_name in ("norm1", "norm2"):
            pt_w = f"{pt_pfx}.{norm_name}.weight"
            if pt_w in pt:
                set_param(f"{jax_pfx}.{norm_name}.scale", pt_w)
            # RMSNorm has no bias in PT; LayerNorm may have bias
            pt_b = f"{pt_pfx}.{norm_name}.bias"
            if pt_b in pt:
                set_param(f"{jax_pfx}.{norm_name}.bias", pt_b)

    # Final encoder norm (if norm_first)
    if "transformer.norm.weight" in pt:
        set_param("transformer.norm.scale", "transformer.norm.weight")
        if "transformer.norm.bias" in pt:
            set_param("transformer.norm.bias", "transformer.norm.bias")

    # -----------------------------------------------------------------------
    # Verify all parameters are mapped
    # -----------------------------------------------------------------------
    missing = sorted(set(target_flat.keys()) - set(out_flat.keys()))
    if missing:
        raise KeyError(
            f"Unmapped JAX params ({len(missing)}): {missing[:20]}{'...' if len(missing) > 20 else ''}"
        )

    return unflatten_dict({tuple(k.split(".")): v for k, v in out_flat.items()})
