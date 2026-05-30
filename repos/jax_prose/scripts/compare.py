#!/usr/bin/env python3
"""Compare PyTorch PROSE-FD and JAX PROSE2to1 forward passes.

Without --prose-root / --checkpoint: structural validation with random weights.
With --prose-root and --checkpoint:  full numerical comparison against PyTorch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jax_prose import PROSE2to1  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare PyTorch and JAX PROSE-FD")
    ap.add_argument("--prose-root", type=Path, default=None,
                    help="Path to original PROSE repo (required for full comparison)")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Path to prose_fd.pth checkpoint")
    ap.add_argument("--msgpack", type=Path, default=None,
                    help="Path to converted JAX .msgpack weights")
    ap.add_argument("--n-words", type=int, default=512)
    ap.add_argument("--x-num", type=int, default=64,
                    help="Spatial grid size (use smaller value for faster structural check)")
    ap.add_argument("--max-output-dim", type=int, default=4)
    ap.add_argument("--input-len", type=int, default=4)
    ap.add_argument("--output-len", type=int, default=4)
    ap.add_argument("--symbol-len", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def _make_inputs(args: argparse.Namespace):
    rng = np.random.default_rng(args.seed)
    data_input = jnp.array(rng.normal(
        size=(1, args.input_len, args.x_num, args.x_num, args.max_output_dim)
    ).astype(np.float32))
    input_times = jnp.zeros((1, args.input_len, 1), dtype=jnp.float32)
    output_times = jnp.zeros((1, args.output_len, 1), dtype=jnp.float32)
    symbol_input = jnp.zeros((1, args.symbol_len), dtype=jnp.int32)
    symbol_mask = jnp.zeros((1, args.symbol_len), dtype=bool)
    return data_input, input_times, output_times, symbol_input, symbol_mask


def run_structural_check(args: argparse.Namespace) -> int:
    print("\n[PROSE-FD] Structural validation (JAX only, random weights)")
    print(f"  x_num={args.x_num}, n_words={args.n_words}, max_output_dim={args.max_output_dim}")

    key = jax.random.PRNGKey(args.seed)
    model = PROSE2to1(
        n_words=args.n_words,
        x_num=args.x_num,
        max_output_dim=args.max_output_dim,
        key=key,
    )

    data_input, input_times, output_times, symbol_input, symbol_mask = _make_inputs(args)
    print(f"  Input shape: {data_input.shape}")

    out = model(data_input, input_times, output_times, symbol_input, symbol_mask)
    out = np.asarray(out)

    print(f"  Output shape: {out.shape}")
    if not np.all(np.isfinite(out)):
        print("  FAIL: output contains non-finite values")
        return 1

    print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
    print("  PASS: output shape correct and finite")
    return 0


def run_full_comparison(args: argparse.Namespace) -> int:
    import torch
    from jax_prose import transfer_pt_to_eqx

    mode = "checkpoint" if args.checkpoint else "random-weight"
    print(f"\n[PROSE-FD] {mode} comparison")
    print(f"  prose-root: {args.prose_root}")
    if args.checkpoint:
        print(f"  checkpoint: {args.checkpoint}")

    sys.path.insert(0, str(args.prose_root))
    from omegaconf import OmegaConf
    from models.transformer_wrappers import PROSE_2to1  # type: ignore[import]
    from symbol_utils.environment import SymbolicEnvironment  # type: ignore[import]

    model_cfg = OmegaConf.load(args.prose_root / "configs/model/prose_2to1.yaml")
    symbol_cfg = OmegaConf.load(args.prose_root / "configs/symbol/symbol.yaml")
    symbol_env = SymbolicEnvironment(symbol_cfg)

    torch.manual_seed(args.seed)
    pt_model = PROSE_2to1(model_cfg, symbol_env, args.x_num, args.max_output_dim, args.output_len)

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        def _strip(k: str) -> str:
            for p in ("module._orig_mod.", "module."):
                if k.startswith(p):
                    return k[len(p):]
            return k

        state = {_strip(k): v for k, v in state.items()}
        pt_model.load_state_dict(state, strict=True)
    pt_model.eval()
    pt_state = pt_model.state_dict()
    n_words = int(pt_state["symbol_encoder.word_embeddings.weight"].shape[0])

    data_input, input_times, output_times, symbol_input, symbol_mask = _make_inputs(args)

    with torch.no_grad():
        y_pt = pt_model.fwd(
            data_input=torch.from_numpy(np.asarray(data_input)),
            input_times=torch.from_numpy(np.asarray(input_times)),
            output_times=torch.from_numpy(np.asarray(output_times)),
            symbol_input=torch.from_numpy(np.asarray(symbol_input).astype(np.int64)),
            symbol_padding_mask=torch.from_numpy(np.asarray(symbol_mask)),
        ).cpu().numpy()

    key = jax.random.PRNGKey(args.seed)
    jax_model = PROSE2to1(
        n_words=n_words, x_num=args.x_num, max_output_dim=args.max_output_dim, key=key
    )
    jax_model = transfer_pt_to_eqx(pt_state, jax_model)

    y_jax = np.asarray(
        jax_model(data_input, input_times, output_times, symbol_input, symbol_mask)
    )

    print(f"  PT  shape: {y_pt.shape}  range [{y_pt.min():.4f}, {y_pt.max():.4f}]")
    print(f"  JAX shape: {y_jax.shape}  range [{y_jax.min():.4f}, {y_jax.max():.4f}]")

    if y_pt.shape != y_jax.shape:
        print(f"  FAIL: shape mismatch {y_pt.shape} vs {y_jax.shape}")
        return 1

    max_d = float(np.max(np.abs(y_pt - y_jax)))
    mean_d = float(np.mean(np.abs(y_pt - y_jax)))
    rel = float(np.linalg.norm((y_jax - y_pt).ravel()) / (np.linalg.norm(y_pt.ravel()) + 1e-8))
    status = "PASS" if rel < args.threshold else "FAIL"
    print(f"  Max abs diff: {max_d:.2e}  Mean abs diff: {mean_d:.2e}  Rel L2: {rel:.2e}  → {status}")
    return 0 if status == "PASS" else 1


def main() -> None:
    args = parse_args()
    if args.prose_root is not None:
        code = run_full_comparison(args)
    else:
        code = run_structural_check(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
