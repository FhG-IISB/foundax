#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.serialization import from_bytes

from jax_prose.prose_fd_2to1 import PROSE2to1


def _strip_prefix(k: str) -> str:
    for p in ("module._orig_mod.", "module."):
        if k.startswith(p):
            return k[len(p) :]
    return k


def main():
    ap = argparse.ArgumentParser(description="Compare PyTorch PROSE-FD and JAX outputs")
    ap.add_argument(
        "--prose-root",
        type=Path,
        default=Path("/home/users/armbrust/projects/prose/prose_fd"),
    )
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--msgpack", type=Path, default=None)
    ap.add_argument("--n-words", type=int, default=512)
    ap.add_argument("--x-num", type=int, default=128)
    ap.add_argument("--max-output-dim", type=int, default=4)
    ap.add_argument("--input-len", type=int, default=10)
    ap.add_argument("--output-len", type=int, default=10)
    ap.add_argument("--symbol-len", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sys.path.insert(0, str(args.prose_root))
    from omegaconf import OmegaConf
    from models.transformer_wrappers import PROSE_2to1
    from symbol_utils.environment import SymbolicEnvironment

    model_cfg = OmegaConf.load(args.prose_root / "configs/model/prose_2to1.yaml")
    symbol_cfg = OmegaConf.load(args.prose_root / "configs/symbol/symbol.yaml")
    symbol_env = SymbolicEnvironment(symbol_cfg)

    use_weights = (args.checkpoint is not None) or (args.msgpack is not None)
    if (args.checkpoint is None) != (args.msgpack is None):
        raise ValueError("Provide both --checkpoint and --msgpack, or neither.")

    pt_model = PROSE_2to1(
        model_cfg, symbol_env, args.x_num, args.max_output_dim, args.output_len
    )
    if use_weights:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        state = {_strip_prefix(k): v for k, v in state.items()}
        pt_model.load_state_dict(state, strict=True)
        n_words = state["symbol_encoder.word_embeddings.weight"].shape[0]
    else:
        state = None
        n_words = args.n_words
    pt_model.eval()

    rng = np.random.default_rng(args.seed)
    data_input = rng.normal(
        size=(1, args.input_len, args.x_num, args.x_num, args.max_output_dim)
    ).astype(np.float32)
    input_times = np.linspace(0.0, 1.0, args.input_len, dtype=np.float32)[None, :, None]
    output_times = np.linspace(1.1, 2.0, args.output_len, dtype=np.float32)[
        None, :, None
    ]

    symbol_input = rng.integers(
        low=0, high=n_words, size=(1, args.symbol_len), dtype=np.int64
    )
    symbol_mask = np.zeros((1, args.symbol_len), dtype=bool)

    with torch.no_grad():
        y_pt = (
            pt_model.fwd(
                data_input=torch.from_numpy(data_input),
                input_times=torch.from_numpy(input_times),
                output_times=torch.from_numpy(output_times),
                symbol_input=torch.from_numpy(symbol_input),
                symbol_padding_mask=torch.from_numpy(symbol_mask),
            )
            .cpu()
            .numpy()
        )

    j_model = PROSE2to1(
        n_words=n_words, x_num=args.x_num, max_output_dim=args.max_output_dim
    )
    j_rng = jax.random.PRNGKey(0)
    init_vars = j_model.init(
        {"params": j_rng},
        jnp.asarray(data_input),
        jnp.asarray(input_times),
        jnp.asarray(output_times),
        jnp.asarray(symbol_input.astype(np.int32)),
        jnp.asarray(symbol_mask),
    )
    params = (
        from_bytes(init_vars["params"], args.msgpack.read_bytes())
        if use_weights
        else init_vars["params"]
    )

    y_jax = j_model.apply(
        {"params": params},
        jnp.asarray(data_input),
        jnp.asarray(input_times),
        jnp.asarray(output_times),
        jnp.asarray(symbol_input.astype(np.int32)),
        jnp.asarray(symbol_mask),
    )
    y_jax = np.asarray(y_jax)

    if use_weights:
        diff = y_jax - y_pt
        max_abs = np.max(np.abs(diff))
        rmse = np.sqrt(np.mean(diff**2))
        rel_l2 = np.linalg.norm(diff.ravel()) / (np.linalg.norm(y_pt.ravel()) + 1e-12)

        print(f"max_abs: {max_abs:.6e}")
        print(f"rmse:    {rmse:.6e}")
        print(f"rel_l2:  {rel_l2:.6e}")
    else:
        print("Smoke mode (no weights):")
        print(f"pt_shape:  {y_pt.shape}, finite={np.isfinite(y_pt).all()}")
        print(f"jax_shape: {y_jax.shape}, finite={np.isfinite(y_jax).all()}")


if __name__ == "__main__":
    main()
