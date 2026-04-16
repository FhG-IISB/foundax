from __future__ import annotations

from typing import Any, Dict

import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict


def load_pytorch_state_dict(checkpoint_path: str) -> Dict[str, Any]:
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    out = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out


def convert_pytorch_to_jax_params(
    pt_state_dict: Dict[str, Any], jax_params: Dict[str, Any]
) -> Dict[str, Any]:
    pt = {}
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

    set_param("pos_embed", "pos_embed", lambda x: np.transpose(x, (0, 2, 3, 1)))

    set_param(
        "patch_embed.conv_patch.kernel",
        "patch_embed.proj.0.weight",
        lambda x: np.transpose(x, (2, 3, 1, 0)),
    )
    set_param("patch_embed.conv_patch.bias", "patch_embed.proj.0.bias")
    set_param(
        "patch_embed.conv_1x1.kernel",
        "patch_embed.proj.2.weight",
        lambda x: np.transpose(x, (2, 3, 1, 0)),
    )
    set_param("patch_embed.conv_1x1.bias", "patch_embed.proj.2.bias")

    set_param("time_agg_layer.w", "time_agg_layer.w")
    set_param("time_agg_layer.gamma", "time_agg_layer.gamma")

    for i in range(100):
        prefix = f"blocks.{i}."
        if f"{prefix}norm1.weight" not in pt:
            break
        set_param(f"blocks_{i}.GroupNorm_0.scale", f"{prefix}norm1.weight")
        set_param(f"blocks_{i}.GroupNorm_0.bias", f"{prefix}norm1.bias")
        set_param(f"blocks_{i}.AFNO2D_0.w1", f"{prefix}filter.w1")
        set_param(f"blocks_{i}.AFNO2D_0.b1", f"{prefix}filter.b1")
        set_param(f"blocks_{i}.AFNO2D_0.w2", f"{prefix}filter.w2")
        set_param(f"blocks_{i}.AFNO2D_0.b2", f"{prefix}filter.b2")
        set_param(f"blocks_{i}.GroupNorm_1.scale", f"{prefix}norm2.weight")
        set_param(f"blocks_{i}.GroupNorm_1.bias", f"{prefix}norm2.bias")
        set_param(
            f"blocks_{i}.mlp_dense_1.kernel",
            f"{prefix}mlp.0.weight",
            lambda x: np.transpose(x[:, :, 0, 0], (1, 0)),
        )
        set_param(f"blocks_{i}.mlp_dense_1.bias", f"{prefix}mlp.0.bias")
        set_param(
            f"blocks_{i}.mlp_dense_2.kernel",
            f"{prefix}mlp.2.weight",
            lambda x: np.transpose(x[:, :, 0, 0], (1, 0)),
        )
        set_param(f"blocks_{i}.mlp_dense_2.bias", f"{prefix}mlp.2.bias")

    set_param(
        "cls_dense_1.kernel", "cls_head.0.weight", lambda x: np.transpose(x, (1, 0))
    )
    set_param("cls_dense_1.bias", "cls_head.0.bias")
    set_param(
        "cls_dense_2.kernel", "cls_head.2.weight", lambda x: np.transpose(x, (1, 0))
    )
    set_param("cls_dense_2.bias", "cls_head.2.bias")
    set_param(
        "cls_dense_3.kernel", "cls_head.4.weight", lambda x: np.transpose(x, (1, 0))
    )
    set_param("cls_dense_3.bias", "cls_head.4.bias")

    set_param(
        "out_deconv.kernel",
        "out_layer.0.weight",
        lambda x: np.transpose(x, (2, 3, 1, 0)),
    )
    set_param("out_deconv.bias", "out_layer.0.bias")
    set_param(
        "out_conv_1.kernel",
        "out_layer.2.weight",
        lambda x: np.transpose(x, (2, 3, 1, 0)),
    )
    set_param("out_conv_1.bias", "out_layer.2.bias")
    set_param(
        "out_conv_2.kernel",
        "out_layer.4.weight",
        lambda x: np.transpose(x, (2, 3, 1, 0)),
    )
    set_param("out_conv_2.bias", "out_layer.4.bias")

    missing = sorted(set(target_flat.keys()) - set(out_flat.keys()))
    if missing:
        raise KeyError(
            f"Unmapped JAX params ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    return unflatten_dict({tuple(k.split(".")): v for k, v in out_flat.items()})


def convert_pytorch_to_jax_params_3d(
    pt_state_dict: Dict[str, Any], jax_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Convert PyTorch DPOTNet3D state_dict to JAX parameter tree."""
    pt = {}
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

    # pos_embed: NCHWD -> NHWDC
    set_param("pos_embed", "pos_embed", lambda x: np.transpose(x, (0, 2, 3, 4, 1)))

    # patch_embed: Conv3d + act + Conv3d(1x1x1)
    set_param(
        "patch_embed.conv_patch.kernel",
        "patch_embed.proj.0.weight",
        lambda x: np.transpose(x, (2, 3, 4, 1, 0)),
    )
    set_param("patch_embed.conv_patch.bias", "patch_embed.proj.0.bias")
    set_param(
        "patch_embed.conv_1x1.kernel",
        "patch_embed.proj.2.weight",
        lambda x: np.transpose(x, (2, 3, 4, 1, 0)),
    )
    set_param("patch_embed.conv_1x1.bias", "patch_embed.proj.2.bias")

    set_param("time_agg_layer.w", "time_agg_layer.w")
    set_param("time_agg_layer.gamma", "time_agg_layer.gamma")

    for i in range(100):
        prefix = f"blocks.{i}."
        if f"{prefix}norm1.weight" not in pt:
            break
        set_param(f"blocks_{i}.GroupNorm_0.scale", f"{prefix}norm1.weight")
        set_param(f"blocks_{i}.GroupNorm_0.bias", f"{prefix}norm1.bias")
        set_param(f"blocks_{i}.AFNO3D_0.w1", f"{prefix}filter.w1")
        set_param(f"blocks_{i}.AFNO3D_0.b1", f"{prefix}filter.b1")
        set_param(f"blocks_{i}.AFNO3D_0.w2", f"{prefix}filter.w2")
        set_param(f"blocks_{i}.AFNO3D_0.b2", f"{prefix}filter.b2")
        set_param(f"blocks_{i}.GroupNorm_1.scale", f"{prefix}norm2.weight")
        set_param(f"blocks_{i}.GroupNorm_1.bias", f"{prefix}norm2.bias")
        # Conv3d(1x1x1) -> Dense
        set_param(
            f"blocks_{i}.mlp_dense_1.kernel",
            f"{prefix}mlp.0.weight",
            lambda x: np.transpose(x[:, :, 0, 0, 0], (1, 0)),
        )
        set_param(f"blocks_{i}.mlp_dense_1.bias", f"{prefix}mlp.0.bias")
        set_param(
            f"blocks_{i}.mlp_dense_2.kernel",
            f"{prefix}mlp.2.weight",
            lambda x: np.transpose(x[:, :, 0, 0, 0], (1, 0)),
        )
        set_param(f"blocks_{i}.mlp_dense_2.bias", f"{prefix}mlp.2.bias")

    # out_layer: ConvTranspose3d + act + Conv3d(1x1x1) + act + Conv3d(1x1x1)
    set_param(
        "out_deconv.kernel",
        "out_layer.0.weight",
        lambda x: np.transpose(x, (2, 3, 4, 1, 0)),
    )
    set_param("out_deconv.bias", "out_layer.0.bias")
    set_param(
        "out_conv_1.kernel",
        "out_layer.2.weight",
        lambda x: np.transpose(x, (2, 3, 4, 1, 0)),
    )
    set_param("out_conv_1.bias", "out_layer.2.bias")
    set_param(
        "out_conv_2.kernel",
        "out_layer.4.weight",
        lambda x: np.transpose(x, (2, 3, 4, 1, 0)),
    )
    set_param("out_conv_2.bias", "out_layer.4.bias")

    missing = sorted(set(target_flat.keys()) - set(out_flat.keys()))
    if missing:
        raise KeyError(
            f"Unmapped JAX params ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    return unflatten_dict({tuple(k.split(".")): v for k, v in out_flat.items()})


def convert_pytorch_to_jax_params_cdpot(
    pt_state_dict: Dict[str, Any], jax_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Convert PyTorch CDPOTNet state_dict to JAX parameter tree."""
    pt = {}
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

    # pos_embed: NCHW -> NHWC
    set_param("pos_embed", "pos_embed", lambda x: np.transpose(x, (0, 2, 3, 1)))

    # CNOPatchEmbed: Conv2d + LReLu_torch + Conv2d(1x1)
    set_param(
        "patch_embed.conv_patch.kernel",
        "patch_embed.proj.0.weight",
        lambda x: np.transpose(x, (2, 3, 1, 0)),
    )
    set_param("patch_embed.conv_patch.bias", "patch_embed.proj.0.bias")
    set_param("patch_embed.act_patching.bias", "patch_embed.act_patching.bias")
    set_param(
        "patch_embed.conv_1x1.kernel",
        "patch_embed.proj.2.weight",
        lambda x: np.transpose(x, (2, 3, 1, 0)),
    )
    set_param("patch_embed.conv_1x1.bias", "patch_embed.proj.2.bias")

    set_param("time_agg_layer.w", "time_agg_layer.w")
    set_param("time_agg_layer.gamma", "time_agg_layer.gamma")

    # Blocks — same as 2D DPOTNet
    for i in range(100):
        prefix = f"blocks.{i}."
        if f"{prefix}norm1.weight" not in pt:
            break
        set_param(f"blocks_{i}.GroupNorm_0.scale", f"{prefix}norm1.weight")
        set_param(f"blocks_{i}.GroupNorm_0.bias", f"{prefix}norm1.bias")
        set_param(f"blocks_{i}.AFNO2D_0.w1", f"{prefix}filter.w1")
        set_param(f"blocks_{i}.AFNO2D_0.b1", f"{prefix}filter.b1")
        set_param(f"blocks_{i}.AFNO2D_0.w2", f"{prefix}filter.w2")
        set_param(f"blocks_{i}.AFNO2D_0.b2", f"{prefix}filter.b2")
        set_param(f"blocks_{i}.GroupNorm_1.scale", f"{prefix}norm2.weight")
        set_param(f"blocks_{i}.GroupNorm_1.bias", f"{prefix}norm2.bias")
        set_param(
            f"blocks_{i}.mlp_dense_1.kernel",
            f"{prefix}mlp.0.weight",
            lambda x: np.transpose(x[:, :, 0, 0], (1, 0)),
        )
        set_param(f"blocks_{i}.mlp_dense_1.bias", f"{prefix}mlp.0.bias")
        set_param(
            f"blocks_{i}.mlp_dense_2.kernel",
            f"{prefix}mlp.2.weight",
            lambda x: np.transpose(x[:, :, 0, 0], (1, 0)),
        )
        set_param(f"blocks_{i}.mlp_dense_2.bias", f"{prefix}mlp.2.bias")

    # cls_head
    set_param(
        "cls_dense_1.kernel", "cls_head.0.weight", lambda x: np.transpose(x, (1, 0))
    )
    set_param("cls_dense_1.bias", "cls_head.0.bias")
    set_param(
        "cls_dense_2.kernel", "cls_head.2.weight", lambda x: np.transpose(x, (1, 0))
    )
    set_param("cls_dense_2.bias", "cls_head.2.bias")
    set_param(
        "cls_dense_3.kernel", "cls_head.4.weight", lambda x: np.transpose(x, (1, 0))
    )
    set_param("cls_dense_3.bias", "cls_head.4.bias")

    # out_layer: CNOBlock(0) + Conv2d(1) + act(2) + Conv2d(3)
    # CNOBlock has: convolution + activation(LReLu_torch)
    set_param(
        "out_cno_block.conv.kernel",
        "out_layer.0.convolution.weight",
        lambda x: np.transpose(x, (2, 3, 1, 0)),
    )
    set_param("out_cno_block.conv.bias", "out_layer.0.convolution.bias")
    set_param("out_cno_block.lrelu_torch.bias", "out_layer.0.activation.bias")
    set_param(
        "out_conv_1.kernel",
        "out_layer.1.weight",
        lambda x: np.transpose(x, (2, 3, 1, 0)),
    )
    set_param("out_conv_1.bias", "out_layer.1.bias")
    set_param(
        "out_conv_2.kernel",
        "out_layer.3.weight",
        lambda x: np.transpose(x, (2, 3, 1, 0)),
    )
    set_param("out_conv_2.bias", "out_layer.3.bias")

    missing = sorted(set(target_flat.keys()) - set(out_flat.keys()))
    if missing:
        raise KeyError(
            f"Unmapped JAX params ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    return unflatten_dict({tuple(k.split(".")): v for k, v in out_flat.items()})
