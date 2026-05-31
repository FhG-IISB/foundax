"""PyTorch → JAX weight conversion for DPOTNet.

Maps the state_dict from the original PyTorch DPOT (https://github.com/thu-ml/DPOT,
models/DPOT.py) to the parameter tree expected by the Equinox re-implementation.

Shape conventions:
- Conv2d:         weight (out, in, kH, kW)  — same in both frameworks
                  bias   (out, 1, 1)         — Equinox; PyTorch uses (out,)
- ConvTranspose2d: weight (out, in, kH, kW)  — Equinox; PyTorch uses (in, out, kH, kW) → transpose
                   bias   (out, 1, 1)         — Equinox; PyTorch uses (out,)
- Linear:          weight (out, in)           — same in both frameworks
- GroupNorm:       weight/bias (channels,)    — same in both frameworks
- MLP blocks in checkpoint: Conv2d (out,in,1,1) → Linear (out,in) — squeeze spatial dims
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def load_pytorch_state_dict(checkpoint_path: str) -> Dict[str, Any]:
    """Load a PyTorch checkpoint and return the raw state_dict as numpy arrays."""
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and all(
        isinstance(v, (int, float, str, dict)) or hasattr(v, "numpy")
        for v in ckpt.values()
    ):
        state_dict = ckpt
    else:
        state_dict = ckpt

    return {k: v.cpu().numpy() if hasattr(v, "numpy") else v for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# Key mapping helpers
# ---------------------------------------------------------------------------

def _map_block_keys(state_dict: Dict, depth: int) -> Dict[str, np.ndarray]:
    """Map transformer block keys from PyTorch to JAX tree paths."""
    out = {}
    for i in range(depth):
        prefix_pt = f"blocks.{i}"
        prefix_jx = f"blocks_{i}"

        for norm in ("norm1", "norm2"):
            for attr in ("weight", "bias"):
                pt_key = f"{prefix_pt}.{norm}.{attr}"
                jx_key = f"{prefix_jx}.{norm}.{attr}"
                if pt_key in state_dict:
                    out[jx_key] = state_dict[pt_key]

        # Checkpoint uses "filter" for AFNO weights, not "afno"
        for afno_attr in ("w1", "w2", "b1", "b2"):
            pt_key = f"{prefix_pt}.filter.{afno_attr}"
            jx_key = f"{prefix_jx}.afno.{afno_attr}"
            if pt_key in state_dict:
                out[jx_key] = state_dict[pt_key]

        # Checkpoint uses mlp.0/mlp.2 (Conv2d (out,in,1,1)), but JAX uses Linear (out,in)
        for idx, dense in (("0", "mlp_dense_1"), ("2", "mlp_dense_2")):
            for attr in ("weight", "bias"):
                pt_key = f"{prefix_pt}.mlp.{idx}.{attr}"
                jx_key = f"{prefix_jx}.{dense}.{attr}"
                if pt_key in state_dict:
                    arr = state_dict[pt_key]
                    if attr == "weight" and arr.ndim == 4:
                        arr = arr[:, :, 0, 0]  # (out, in, 1, 1) → (out, in)
                    out[jx_key] = arr

    return out


def convert_pytorch_to_jax_params(
    state_dict: Dict[str, Any],
    variant: str = "Ti",
) -> Dict[str, Any]:
    """Convert a PyTorch DPOT state_dict to the nested dict expected by DPOTNet.

    The returned dict matches the Equinox attribute tree; callers load it into
    a model via ``eqx.tree_at`` using the flat-key helper in this module.

    Args:
        state_dict: Output of ``load_pytorch_state_dict``.
        variant: One of "Ti", "S", "M", "L", "H" (used to look up ``depth``).

    Returns:
        Flat dict with keys like ``"blocks_3.afno.w1"``, ``"patch_embed.conv_patch.weight"``, …
        Shapes are already adjusted for Equinox (Conv2d bias: (out,1,1), ConvTranspose2d weight
        transposed to (out,in,kH,kW)).
    """
    from .configs import DPOT_CONFIGS

    cfg = DPOT_CONFIGS[variant]
    depth = cfg.depth

    params: Dict[str, np.ndarray] = {}

    # ── Patch embedding ────────────────────────────────────────────────
    # Checkpoint uses patch_embed.proj.0 and patch_embed.proj.2 (Sequential indices)
    for attr in ("weight", "bias"):
        for pt_sub, jx_sub in (("proj.0", "conv_patch"), ("proj.2", "conv_1x1")):
            pt_key = f"patch_embed.{pt_sub}.{attr}"
            jx_key = f"patch_embed.{jx_sub}.{attr}"
            if pt_key in state_dict:
                arr = state_dict[pt_key]
                if attr == "bias":
                    arr = arr[:, None, None]  # (C,) → (C, 1, 1) for Equinox Conv2d
                params[jx_key] = arr
            elif jx_key in state_dict:
                params[jx_key] = state_dict[jx_key]

    # ── Positional embedding ───────────────────────────────────────────
    # Checkpoint: (1, C, H, W) channel-first; JAX model expects (1, H, W, C) channel-last
    if "pos_embed" in state_dict:
        pe = state_dict["pos_embed"]
        if pe.ndim == 4:
            pe = pe.transpose(0, 2, 3, 1)
        params["pos_embed"] = pe

    # ── Time aggregator ────────────────────────────────────────────────
    for attr in ("w", "gamma"):
        for pt_name in (f"time_agg_layer.{attr}", f"time_agg.{attr.upper()}", f"time_agg.{attr}"):
            if pt_name in state_dict:
                params[f"time_agg_layer.{attr}"] = state_dict[pt_name]
                break

    # ── Transformer blocks ─────────────────────────────────────────────
    params.update(_map_block_keys(state_dict, depth))

    # ── Classification head ────────────────────────────────────────────
    # PyTorch: cls_head = nn.Sequential(Linear, act, Linear, act, Linear)
    #          → indices 0, 2, 4
    for idx, jx_name in ((0, "cls_dense_1"), (2, "cls_dense_2"), (4, "cls_dense_3")):
        for attr in ("weight", "bias"):
            pt_key = f"cls_head.{idx}.{attr}"
            jx_key = f"{jx_name}.{attr}"
            if pt_key in state_dict:
                params[jx_key] = state_dict[pt_key]
            elif f"{jx_name}.{attr}" in state_dict:
                params[jx_key] = state_dict[f"{jx_name}.{attr}"]

    # ── Output head ────────────────────────────────────────────────────
    # Checkpoint uses out_layer.0/2/4 (Sequential indices)
    for idx, jx_name, layer_type in (
        ("0", "out_deconv", "deconv"),
        ("2", "out_conv_1", "conv"),
        ("4", "out_conv_2", "conv"),
    ):
        for attr in ("weight", "bias"):
            pt_key = f"out_layer.{idx}.{attr}"
            jx_key = f"{jx_name}.{attr}"
            if pt_key in state_dict:
                arr = state_dict[pt_key]
                if attr == "weight" and layer_type == "deconv":
                    # ConvTranspose2d: PyTorch (in, out, kH, kW) → Equinox (out, in, kH, kW)
                    # Equinox also flips kH/kW spatially vs PyTorch
                    arr = arr.transpose(1, 0, 2, 3)[:, :, ::-1, ::-1].copy()
                elif attr == "bias":
                    arr = arr[:, None, None]  # (C,) → (C, 1, 1)
                params[jx_key] = arr
            elif jx_key in state_dict:
                params[jx_key] = state_dict[jx_key]

    # ── Optional normalization layers ──────────────────────────────────
    for name in ("scale_feats_mu", "scale_feats_sigma"):
        for attr in ("weight", "bias"):
            key = f"{name}.{attr}"
            if key in state_dict:
                params[key] = state_dict[key]

    unmapped = set(state_dict.keys()) - _all_pt_keys(depth)
    if unmapped:
        print(f"[convert_dpot] {len(unmapped)} unmapped PyTorch keys (may be buffers):")
        for k in sorted(unmapped)[:10]:
            print(f"  {k}")

    return params


def _all_pt_keys(depth: int) -> set:
    """Return the set of all PyTorch state_dict keys we handle, for unmapped-key reporting."""
    keys = set()
    for attr in ("weight", "bias"):
        for sub in ("proj.0", "proj.2"):
            keys.add(f"patch_embed.{sub}.{attr}")
    keys.add("pos_embed")
    for attr in ("w", "W", "gamma"):
        keys.update({f"time_agg_layer.{attr}", f"time_agg.{attr}", f"time_agg.{attr.upper()}"})
    for i in range(depth):
        p = f"blocks.{i}"
        keys.update({f"{p}.norm1.weight", f"{p}.norm1.bias", f"{p}.norm2.weight", f"{p}.norm2.bias"})
        keys.update({f"{p}.filter.w1", f"{p}.filter.w2", f"{p}.filter.b1", f"{p}.filter.b2"})
        keys.update({f"{p}.mlp.0.weight", f"{p}.mlp.0.bias",
                     f"{p}.mlp.2.weight", f"{p}.mlp.2.bias"})
    for idx in (0, 2, 4):
        keys.update({f"cls_head.{idx}.weight", f"cls_head.{idx}.bias"})
    for idx in ("0", "2", "4"):
        keys.update({f"out_layer.{idx}.weight", f"out_layer.{idx}.bias"})
    for name in ("scale_feats_mu", "scale_feats_sigma"):
        keys.update({f"{name}.weight", f"{name}.bias"})
    return keys


def load_jax_params(flat_params: Dict[str, np.ndarray], model) -> Any:
    """Load flat params dict into an Equinox DPOTNet model via ``eqx.tree_at``.

    ``flat_params`` is the output of ``convert_pytorch_to_jax_params``.
    Handles ``blocks_N`` keys by indexing into ``model.blocks[N]``.
    """
    import jax.numpy as jnp
    import equinox as eqx

    def _get_leaf(m, keys):
        node = m
        for k in keys:
            if k.startswith("blocks_") and k[7:].isdigit():
                node = node.blocks[int(k[7:])]
            else:
                node = getattr(node, k)
        return node

    def _set_leaf(model, path_str: str, value: np.ndarray):
        keys = path_str.split(".")
        get_leaf = lambda m: _get_leaf(m, keys)  # noqa: E731
        return eqx.tree_at(get_leaf, model, jnp.array(value))

    for path, arr in flat_params.items():
        try:
            model = _set_leaf(model, path, arr)
        except Exception as e:
            print(f"[load_jax_params] Warning: could not set {path}: {e}")

    return model
