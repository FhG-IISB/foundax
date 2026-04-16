import numpy as np
import torch
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import unfreeze, freeze
from flax.traverse_util import flatten_dict, unflatten_dict
from typing import Dict, Any, Tuple, Set


def jax_key_to_pytorch_key(jax_key: str, large_model: bool) -> str:
    """
    Convert a JAX parameter key to the corresponding PyTorch key.
    """
    if jax_key.startswith("params/"):
        jax_key = jax_key[7:]

    parts = jax_key.split("/")
    pt_parts = []

    is_top_level_residual = parts[0].startswith("residual_block_")
    is_decoder = len(parts) > 0 and parts[0] == "decoder"

    i = 0
    while i < len(parts):
        part = parts[i]

        if part.startswith("residual_block_") and is_top_level_residual:
            idx = part.split("_")[2]
            pt_parts.append("residual_blocks")
            pt_parts.append(idx)
            i += 1
            continue

        if part.startswith("block_") and is_top_level_residual:
            idx = part.split("_")[1]
            pt_parts.append(idx)
            i += 1
            continue

        if part.startswith("layer_"):
            idx = int(part.split("_")[1])
            if is_decoder:
                reversed_idx = 3 - idx
                pt_parts.extend(["layers", str(reversed_idx)])
            else:
                pt_parts.extend(["layers", str(idx)])
            i += 1
            continue

        if large_model:
            if part.startswith("block_") and not is_top_level_residual:
                idx = int(part.split("_")[1])
                if is_decoder:
                    layer_idx = None
                    for p in pt_parts:
                        if p == "layers":
                            layer_pos = pt_parts.index("layers")
                            if layer_pos + 1 < len(pt_parts):
                                layer_idx = int(pt_parts[layer_pos + 1])
                            break

                    depths = (8, 8, 8, 8)
                    if layer_idx is not None:
                        num_blocks = depths[3 - layer_idx]
                    else:
                        num_blocks = 8

                    reversed_block_idx = num_blocks - 1 - idx
                    pt_parts.extend(["blocks", str(reversed_block_idx)])
                else:
                    pt_parts.extend(["blocks", str(idx)])
                i += 1
                continue
        else:
            if part.startswith("block_") and not is_top_level_residual:
                idx = int(part.split("_")[1])
                if is_decoder:
                    depth = 4
                    reversed_block_idx = depth - 1 - idx
                    pt_parts.extend(["blocks", str(reversed_block_idx)])
                else:
                    pt_parts.extend(["blocks", str(idx)])
                i += 1
                continue

        if part == "attention":
            if i + 1 < len(parts):
                next_part = parts[i + 1]
                if next_part in ["query", "key", "value", "logit_scale"]:
                    pt_parts.extend(["attention", "self"])
                elif next_part == "proj":
                    pt_parts.extend(["attention", "output", "dense"])
                    i += 2
                    continue
                elif next_part == "relative_position_bias":
                    pt_parts.extend(["attention", "self"])
                    i += 2
                    if i < len(parts):
                        cpb_part = parts[i]
                        if cpb_part.startswith("cpb_mlp_"):
                            mlp_idx = int(cpb_part.split("_")[2])
                            pt_mlp_idx = 0 if mlp_idx == 0 else 2
                            pt_parts.extend(
                                ["continuous_position_bias_mlp", str(pt_mlp_idx)]
                            )
                            i += 1
                    continue
                else:
                    pt_parts.append("attention")
            else:
                pt_parts.append("attention")
            i += 1
            continue

        if part == "upsample":
            pt_parts.append("upsample")
            i += 1
            continue

        if part == "downsample":
            pt_parts.append("downsample")
            i += 1
            continue

        if part == "norm":
            pt_parts.append("norm")
            i += 1
            continue

        if (
            part == "weight"
            and i + 1 < len(parts)
            and parts[i + 1] in ["kernel", "bias"]
        ):
            pt_parts.append("weight")
            next_part = parts[i + 1]
            pt_parts.append("weight" if next_part == "kernel" else "bias")
            i += 2
            continue

        if part == "bias" and i + 1 < len(parts) and parts[i + 1] in ["kernel", "bias"]:
            pt_parts.append("bias")
            next_part = parts[i + 1]
            pt_parts.append("weight" if next_part == "kernel" else "bias")
            i += 2
            continue

        if part == "kernel":
            pt_parts.append("weight")
            i += 1
            continue

        if part == "bias":
            pt_parts.append("bias")
            i += 1
            continue

        if part in [
            "encoder",
            "decoder",
            "embeddings",
            "patch_embeddings",
            "projection",
            "intermediate",
            "output",
            "dense",
            "reduction",
            "dwconv",
            "pwconv1",
            "pwconv2",
            "query",
            "key",
            "value",
            "logit_scale",
            "patch_recovery",
            "mixup",
        ]:
            pt_parts.append(part)
            i += 1
            continue

        if part.startswith("layernorm_"):
            pt_parts.append(part)
            i += 1
            continue

        if part == "weight" and (
            i + 1 >= len(parts) or parts[i + 1] not in ["kernel", "bias"]
        ):
            pt_parts.append("weight")
            i += 1
            continue

        pt_parts.append(part)
        i += 1

    return ".".join(pt_parts)


def transpose_weight(
    jax_key: str, pt_np: np.ndarray, jax_shape: Tuple[int, ...]
) -> np.ndarray:
    """Apply necessary transpositions to convert PyTorch weights to JAX format."""
    if pt_np.ndim == 2:
        if pt_np.shape == jax_shape[::-1]:
            return pt_np.T
        elif pt_np.shape == jax_shape:
            return pt_np

    elif pt_np.ndim == 4:
        if "dwconv" in jax_key:
            return pt_np.transpose(2, 3, 1, 0)

        if "patch_embeddings" in jax_key and "projection" in jax_key:
            return pt_np.transpose(2, 3, 1, 0)

        if "patch_recovery" in jax_key and "projection" in jax_key:
            kernel = pt_np.transpose(2, 3, 0, 1)
            return kernel[::-1, ::-1, :, :]

        if "patch_recovery" in jax_key and "mixup" in jax_key:
            return pt_np.transpose(2, 3, 1, 0)

    return pt_np


def convert_pytorch_to_jax(
    pt_state_dict: Dict[str, torch.Tensor],
    jax_params: Dict[str, Any],
    verbose: bool = False,
    large_model: bool = True,
) -> Dict[str, Any]:
    """Convert PyTorch state dict to JAX/Flax parameters."""
    flat_jax = flatten_dict(unfreeze(jax_params), sep="/")
    new_flat_jax = {}
    used_pt_keys: Set[str] = set()

    matched = 0
    unmatched_jax = []
    shape_mismatches = []
    duplicate_pt_keys = []

    for jax_key_str, jax_array in flat_jax.items():
        pt_key = jax_key_to_pytorch_key(jax_key_str, large_model)

        if verbose:
            pt_shape_str = (
                str(tuple(pt_state_dict[pt_key].shape))
                if pt_key in pt_state_dict
                else "N/A"
            )
            print(
                f"{jax_key_str:80s} {str(tuple(jax_array.shape)):20s} -> {pt_key:80s} {pt_shape_str}"
            )

        if pt_key not in pt_state_dict:
            unmatched_jax.append((jax_key_str, pt_key))
            new_flat_jax[jax_key_str] = jax_array
            continue

        if pt_key in used_pt_keys:
            duplicate_pt_keys.append((jax_key_str, pt_key))
            new_flat_jax[jax_key_str] = jax_array
            continue

        used_pt_keys.add(pt_key)

        pt_tensor = pt_state_dict[pt_key]
        # Some PyTorch builds raise "RuntimeError: Numpy is not available" when
        # calling `tensor.numpy()` (e.g. numpy missing for that build). Avoid
        # that by trying `.numpy()` and falling back to `.tolist()` -> `np.array()`
        # which does not require PyTorch to import NumPy internally.
        try:
            pt_np = pt_tensor.cpu().numpy()
        except Exception:
            pt_list = pt_tensor.detach().cpu().tolist()
            pt_np = np.array(pt_list)
        pt_np = transpose_weight(jax_key_str, pt_np, jax_array.shape)

        if pt_np.shape != jax_array.shape:
            shape_mismatches.append(
                (jax_key_str, pt_key, pt_tensor.shape, pt_np.shape, jax_array.shape)
            )
            new_flat_jax[jax_key_str] = jax_array
            continue

        new_flat_jax[jax_key_str] = jnp.array(pt_np)
        matched += 1

    all_pt_keys = set(pt_state_dict.keys())
    unused_pt_keys = all_pt_keys - used_pt_keys

    print(f"\n{'='*80}")
    print(f"CONVERSION SUMMARY")
    print(f"{'='*80}")
    print(f"  JAX parameters:       {len(flat_jax)}")
    print(f"  PyTorch parameters:   {len(pt_state_dict)}")
    print(f"  Successfully matched: {matched}")
    print(f"  Unmatched JAX keys:   {len(unmatched_jax)}")
    print(f"  Shape mismatches:     {len(shape_mismatches)}")
    print(f"  Duplicate PT keys:    {len(duplicate_pt_keys)}")
    print(f"  Unused PT keys:       {len(unused_pt_keys)}")

    if len(unused_pt_keys) > 0:
        with open("unused_pytorch_keys.txt", "w") as f:
            f.write(f"Unused PyTorch Keys ({len(unused_pt_keys)})\n")
            f.write("=" * 80 + "\n\n")
            for pt_k in sorted(unused_pt_keys):
                shape = tuple(pt_state_dict[pt_k].shape)
                f.write(f"{pt_k}: {shape}\n")
        print(f"  Written to: unused_pytorch_keys.txt")

    if len(unmatched_jax) > 0:
        with open("unmatched_jax_keys.txt", "w") as f:
            f.write(f"Unmatched JAX Keys ({len(unmatched_jax)})\n")
            f.write("=" * 80 + "\n\n")
            for jax_k, attempted_pt_k in sorted(unmatched_jax):
                shape = tuple(flat_jax[jax_k].shape)
                f.write(f"JAX key:      {jax_k}\n")
                f.write(f"  Shape:      {shape}\n")
                f.write(f"  Tried PT:   {attempted_pt_k}\n\n")
        print(f"  Written to: unmatched_jax_keys.txt")

    if shape_mismatches:
        with open("shape_mismatches.txt", "w") as f:
            f.write(f"Shape Mismatches ({len(shape_mismatches)})\n")
            f.write("=" * 80 + "\n\n")
            for (
                jax_k,
                pt_k,
                pt_orig_shape,
                pt_transposed_shape,
                jax_shape,
            ) in shape_mismatches:
                f.write(f"JAX key:           {jax_k}\n")
                f.write(f"PT key:            {pt_k}\n")
                f.write(f"PT original shape: {pt_orig_shape}\n")
                f.write(f"PT transposed:     {pt_transposed_shape}\n")
                f.write(f"JAX expected:      {jax_shape}\n\n")
        print(f"  Written to: shape_mismatches.txt")

    if verbose and unused_pt_keys:
        print(f"\n--- Unused PyTorch keys ({len(unused_pt_keys)}) ---")
        for pt_k in sorted(unused_pt_keys):
            print(f"  {pt_k}: {tuple(pt_state_dict[pt_k].shape)}")

    if matched == len(flat_jax) and len(unused_pt_keys) == 0:
        print(f"\n  SUCCESS: All parameters matched perfectly!")

    new_params = unflatten_dict(
        {tuple(k.split("/")): v for k, v in new_flat_jax.items()}
    )
    return freeze(new_params)


def get_model_config(model_name: str):
    """
    Get the ScOTConfig for a given model name.

    Args:
        model_name: One of 'poseidonT', 'poseidonB', or 'poseidonL'

    Returns:
        Tuple of (config, large_model_flag)
    """
    from jax_poseidon import ScOTConfig

    if model_name == "poseidonT":
        large_model = False
        config = ScOTConfig(
            image_size=128,
            patch_size=4,
            num_channels=4,
            num_out_channels=4,
            embed_dim=48,
            depths=(4, 4, 4, 4),
            num_heads=(3, 6, 12, 24),
            skip_connections=(2, 2, 2, 0),
            window_size=16,
            mlp_ratio=4.0,
            qkv_bias=True,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            drop_path_rate=0.0,
            hidden_act="gelu",
            use_absolute_embeddings=False,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            p=1,
            channel_slice_list_normalized_loss=[0, 1, 3, 4],
            residual_model="convnext",
            use_conditioning=True,
            learn_residual=False,
            pretrained_window_sizes=(0, 0, 0, 0),
            chunk_size_feed_forward=0,
            output_attentions=False,
            output_hidden_states=False,
            use_return_dict=True,
        )
    elif model_name == "poseidonB":
        large_model = True
        config = ScOTConfig(
            image_size=128,
            patch_size=4,
            num_channels=4,
            num_out_channels=4,
            embed_dim=96,
            depths=(8, 8, 8, 8),
            num_heads=(3, 6, 12, 24),
            skip_connections=(2, 2, 2, 0),
            window_size=16,
            mlp_ratio=4.0,
            qkv_bias=True,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            drop_path_rate=0.0,
            hidden_act="gelu",
            use_absolute_embeddings=False,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            p=1,
            channel_slice_list_normalized_loss=[0, 1, 3, 4],
            residual_model="convnext",
            use_conditioning=True,
            learn_residual=False,
            pretrained_window_sizes=(0, 0, 0, 0),
            chunk_size_feed_forward=0,
            output_attentions=False,
            output_hidden_states=False,
            use_return_dict=True,
        )
    elif model_name == "poseidonL":
        large_model = True
        config = ScOTConfig(
            image_size=128,
            patch_size=4,
            num_channels=4,
            num_out_channels=4,
            embed_dim=192,
            depths=(8, 8, 8, 8),
            num_heads=(3, 6, 12, 24),
            skip_connections=(2, 2, 2, 0),
            window_size=16,
            mlp_ratio=4.0,
            qkv_bias=True,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            drop_path_rate=0.0,
            hidden_act="gelu",
            use_absolute_embeddings=False,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            p=1,
            channel_slice_list_normalized_loss=[0, 1, 3, 4],
            residual_model="convnext",
            use_conditioning=True,
            learn_residual=False,
            pretrained_window_sizes=(0, 0, 0, 0),
            chunk_size_feed_forward=0,
            output_attentions=False,
            output_hidden_states=False,
            use_return_dict=True,
        )
    else:
        raise ValueError(
            f"Unknown model: {model_name}. Expected poseidonT, poseidonB, or poseidonL"
        )

    return config, large_model


def convert_model(
    model_path: str,
    save: bool = False,
    verbose: bool = False,
):
    """
    Main conversion function that loads a PyTorch model and converts it to JAX.

    Args:
        model_path: Path to the PyTorch model checkpoint
        save: Whether to save the converted model to a .msgpack file
        verbose: Whether to print detailed conversion information

    Returns:
        Tuple of (jax_model, jax_params, pt_model, config)
    """
    from scOT.model import ScOT as pytorch_ScOT
    from jax_poseidon import ScOT

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load PyTorch model
    print("Loading PyTorch model...")
    pt_model = pytorch_ScOT.from_pretrained(model_path).to(device)
    pt_model.eval()

    num_params = sum(p.numel() for p in pt_model.parameters() if p.requires_grad)
    print(f"PyTorch model parameters: {num_params:,}")

    pt_state_dict = pt_model.state_dict()

    # Get config based on model name
    model_name = model_path.split("/")[-1]
    config, large_model = get_model_config(model_name)

    # Initialize JAX model
    print("\nInitializing JAX model...")
    rng = jax.random.PRNGKey(0)
    jax_model = ScOT(config, False, False)
    jax_params_init = jax_model.init(
        {"params": rng, "dropout": rng},
        pixel_values=jnp.ones((1, 128, 128, 4)),
        time=jnp.zeros((1,)),
        deterministic=False,
    )

    # Count JAX params
    flat_jax = flatten_dict(unfreeze(jax_params_init), sep="/")
    jax_num_params = sum(np.prod(v.shape) for v in flat_jax.values())
    print(f"JAX model parameters: {jax_num_params:,}")

    # Convert parameters
    jax_params = convert_pytorch_to_jax(
        pt_state_dict, jax_params_init, verbose, large_model
    )

    # Save if requested
    if save:
        from flax.serialization import to_bytes

        output_path = f"{model_name}.msgpack"
        with open(output_path, "wb") as f:
            f.write(to_bytes(jax_params))
        print(f"Saved JAX weights to {output_path}")

    return jax_model, jax_params, pt_model, config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert PyTorch model to JAX")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the PyTorch model checkpoint (e.g., /path/to/poseidonT)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the converted JAX model to a .msgpack file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed conversion information",
    )
    args = parser.parse_args()

    jax_model, jax_params, pt_model, config = convert_model(
        args.model_path,
        save=args.save,
        verbose=args.verbose,
    )

    print("\nConversion complete!")
