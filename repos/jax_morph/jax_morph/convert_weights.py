"""PyTorch → JAX weight conversion for MORPH (ViT3DRegression).

Key mapping differences vs PyTorch checkpoint:
- transformer_blocks.N.* → model.transformer_blocks[N].* (list indexing)
- conv_stack.N.* → model.patch_embedding.conv_features.conv_stack[N].* (list)
- field_attn.attn.in_proj_weight [3E,E] → q/k/v_proj.weight [E,E] each (split)
- field_attn.attn.out_proj.* → field_attn.out_proj.*
- mlp.0.* → mlp_0.*, mlp.3.* → mlp_1.*

Shapes are identical in both frameworks:
- Linear weight (out, in) — same in PyTorch and Equinox
- Conv3d weight (out, in, kD, kH, kW) — same; no bias (use_bias=False)
- LayerNorm weight/bias (dim,) — same
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union


def _arr(v):
    import numpy as np
    if hasattr(v, "numpy"):
        return v.cpu().numpy()
    return np.asarray(v)


def _get_leaf(model, path: List[Union[str, int]]):
    node = model
    for k in path:
        if isinstance(k, int):
            node = node[k]
        else:
            node = getattr(node, k)
    return node


def _set_leaf(model, path: List[Union[str, int]], value):
    import jax.numpy as jnp
    import equinox as eqx

    arr = jnp.array(value)
    return eqx.tree_at(lambda m: _get_leaf(m, path), model, arr)


def _map_key(pt_key: str) -> Optional[List[Union[str, int]]]:
    """Convert a PyTorch state_dict key to a JAX model path.

    Returns None for keys that are handled separately (field_attn MHA).
    """
    # field_attn MHA: handled outside
    if "field_attn.attn.in_proj_weight" in pt_key or "field_attn.attn.in_proj_bias" in pt_key:
        return None

    # field_attn out_proj: strip the intermediate .attn.
    pt_key = pt_key.replace("field_attn.attn.out_proj.", "field_attn.out_proj.")

    parts = pt_key.split(".")
    jax_path: List[Union[str, int]] = []
    i = 0
    while i < len(parts):
        p = parts[i]
        if p in ("transformer_blocks", "conv_stack") and i + 1 < len(parts) and parts[i + 1].isdigit():
            jax_path.append(p)
            jax_path.append(int(parts[i + 1]))
            i += 2
        elif p == "mlp" and i + 1 < len(parts):
            idx = parts[i + 1]
            if idx == "0":
                jax_path.append("mlp_0")
            elif idx == "3":
                jax_path.append("mlp_1")
            else:
                return None  # unexpected index
            i += 2
        else:
            jax_path.append(p)
            i += 1
    return jax_path


def convert_pytorch_to_jax_params(state_dict: Dict[str, Any], model: Any) -> Any:
    """Load a PyTorch MORPH state_dict into an Equinox ViT3DRegression model.

    Args:
        state_dict: PyTorch state_dict (values may be torch.Tensor or numpy arrays).
        model: An initialised Equinox ViT3DRegression instance.

    Returns:
        A new model with weights loaded from state_dict.
    """
    # Strip DataParallel prefix if present
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    embed_dim = model.patch_embedding.embed_dim

    # ── FieldCrossAttention: fused in_proj → separate q/k/v projections ──
    in_proj_w = state_dict.get("patch_embedding.field_attn.attn.in_proj_weight")
    in_proj_b = state_dict.get("patch_embedding.field_attn.attn.in_proj_bias")
    if in_proj_w is not None:
        W = _arr(in_proj_w)
        model = _set_leaf(model, ["patch_embedding", "field_attn", "q_proj", "weight"], W[:embed_dim])
        model = _set_leaf(model, ["patch_embedding", "field_attn", "k_proj", "weight"], W[embed_dim:2 * embed_dim])
        model = _set_leaf(model, ["patch_embedding", "field_attn", "v_proj", "weight"], W[2 * embed_dim:])
    if in_proj_b is not None:
        b = _arr(in_proj_b)
        model = _set_leaf(model, ["patch_embedding", "field_attn", "q_proj", "bias"], b[:embed_dim])
        model = _set_leaf(model, ["patch_embedding", "field_attn", "k_proj", "bias"], b[embed_dim:2 * embed_dim])
        model = _set_leaf(model, ["patch_embedding", "field_attn", "v_proj", "bias"], b[2 * embed_dim:])

    # ── All remaining keys ──────────────────────────────────────────────
    for pt_key, v in state_dict.items():
        jax_path = _map_key(pt_key)
        if jax_path is None:
            continue
        try:
            model = _set_leaf(model, jax_path, _arr(v))
        except Exception as e:
            print(f"[morph convert] Warning: could not set {pt_key} → {jax_path}: {e}")

    return model
