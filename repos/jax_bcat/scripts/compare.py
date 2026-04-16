#!/usr/bin/env python3
"""Compare PyTorch BCAT and JAX BCAT forward passes."""
from __future__ import annotations

import argparse
import sys
import os

import numpy as np
import torch

import jax
import jax.numpy as jnp

from jax_bcat import bcat_default, load_pytorch_state_dict, convert_pytorch_to_jax_params


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PyTorch and JAX BCAT outputs")
    parser.add_argument("--checkpoint", default=None, help="Path to PyTorch checkpoint (omit for random weights)")
    parser.add_argument(
        "--bcat-root",
        default=os.environ.get("BCAT_ROOT", "./ogrepo/bcat"),
        help="Path to original BCAT repository",
    )
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--x-num", type=int, default=128)
    parser.add_argument("--data-dim", type=int, default=4)
    parser.add_argument("--t-num", type=int, default=20)
    parser.add_argument("--input-len", type=int, default=10)
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(args.bcat_root, "src"))

    from omegaconf import OmegaConf

    # Load the default BCAT config
    cfg_path = os.path.join(args.bcat_root, "src", "configs", "model", "bcat.yaml")
    model_config = OmegaConf.load(cfg_path)
    # Resolve interpolations manually
    model_config.patch_num_output = model_config.patch_num
    model_config.embedder.dim = model_config.dim_emb
    model_config.embedder.patch_num = model_config.patch_num
    model_config.embedder.patch_num_output = model_config.patch_num_output

    from models.bcat import BCAT as BCAT_PT

    rng = np.random.default_rng(args.seed)
    data_np = rng.normal(size=(1, args.t_num, args.x_num, args.x_num, args.data_dim)).astype(np.float32)
    times_np = np.arange(args.t_num, dtype=np.float32).reshape(1, args.t_num, 1)

    # Build PyTorch model
    model_config.kv_cache = 0  # disable KV cache for comparison
    pt_model = BCAT_PT(model_config, args.x_num, args.data_dim, max_data_len=args.t_num)
    if args.checkpoint is not None:
        pt_state = load_pytorch_state_dict(args.checkpoint)
        pt_model.load_state_dict(pt_state, strict=True)
    pt_model.eval()
    pt_state = dict(pt_model.state_dict())

    data_pt = torch.from_numpy(data_np)
    times_pt = torch.from_numpy(times_np)

    with torch.no_grad():
        # Manually replicate fwd() to avoid sdpa_kernel(CUDNN_ATTENTION)
        # which requires a CUDA backend.
        from models.bcat import block_lower_triangular_mask as pt_mask_fn

        d_in = data_pt[:, :-1]
        t_in = times_pt[:, :-1]
        enc = pt_model.embedder.encode(d_in, t_in)
        data_len = enc.size(1)
        mask = pt_mask_fn(
            pt_model.seq_len_per_step, args.t_num, use_float=True,
        )[:data_len, :data_len]
        enc = pt_model.transformer(enc, mask)
        input_seq_len = (args.input_len - 1) * pt_model.seq_len_per_step
        y_pt = pt_model.embedder.decode(enc[:, input_seq_len:])
    y_pt_np = y_pt.numpy()

    # Build JAX model
    model = bcat_default()
    variables = model.init(
        jax.random.PRNGKey(0),
        jnp.asarray(data_np),
        jnp.asarray(times_np),
        input_len=args.input_len,
    )
    jax_params = convert_pytorch_to_jax_params(pt_state, variables["params"])

    y_jax = model.apply(
        {"params": jax_params},
        jnp.asarray(data_np),
        jnp.asarray(times_np),
        input_len=args.input_len,
    )
    y_jax_np = np.asarray(y_jax)

    d = max_abs_diff(y_pt_np, y_jax_np)
    print(f"Output shapes  PT: {y_pt_np.shape}  JAX: {y_jax_np.shape}")
    print(f"max_abs_diff:  {d:.6e}")

    if d >= args.threshold:
        print(f"FAIL: {d:.3e} >= {args.threshold:.3e}")
        raise SystemExit(1)

    print(f"PASS: {d:.3e} < {args.threshold:.3e}")


if __name__ == "__main__":
    main()
