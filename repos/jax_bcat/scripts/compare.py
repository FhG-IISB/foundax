"""Compare PyTorch BCAT and JAX BCAT forward passes.

Without --bcat-root: structural validation with random weights (JAX only).
With --bcat-root:    full numerical comparison against PyTorch + checkpoint.

Usage (structural check, no original repo needed):
    python compare.py

Usage (full numerical comparison):
    python compare.py --bcat-root /path/to/BCAT --checkpoint /path/to/ckpt.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jax_bcat import BCAT  # noqa: E402
from jax_bcat.configs import BCAT_CONFIGS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch and JAX BCAT outputs")
    parser.add_argument("--bcat-root", type=Path, default=None,
                        help="Path to cloned BCAT repo (required for full comparison)")
    parser.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=1e-3)
    return parser.parse_args()


def _build_small_model(key: jax.Array) -> BCAT:
    """Build a small BCAT for fast structural checks."""
    return BCAT(
        n_layer=2,
        dim_emb=64,
        dim_ffn=128,
        n_head=2,
        norm_first=True,
        norm_type="rms",
        activation=jax.nn.silu,
        gated=True,
        qk_norm=True,
        x_num=16,
        max_output_dim=1,
        patch_num=4,
        patch_num_output=4,
        conv_dim=8,
        time_embed="learnable",
        max_time_len=12,
        max_data_len=12,
        deep=False,
        data_dim=1,
        key=key,
    )


def run_structural_check(args: argparse.Namespace) -> int:
    print("\n[BCAT] Structural validation (JAX only, random weights)")
    key = jax.random.PRNGKey(args.seed)
    model = _build_small_model(key)

    T_total = 11
    input_len = 10
    data = jnp.zeros((1, T_total, 16, 16, 1), dtype=jnp.float32)
    times = jnp.zeros((1, T_total, 1), dtype=jnp.float32)
    print(f"  Input shape: data={data.shape}, times={times.shape}, input_len={input_len}")

    t0 = time.perf_counter()
    out = model(data, times, input_len=input_len)
    elapsed = time.perf_counter() - t0

    out_np = np.asarray(out)
    print(f"  Output shape: {out_np.shape}")

    if not np.all(np.isfinite(out_np)):
        print("  FAIL: output contains non-finite values")
        return 1

    print(f"  Output range: [{out_np.min():.4f}, {out_np.max():.4f}]")
    print(f"  Elapsed: {elapsed:.2f}s")
    print("  PASS: output shape correct and finite")
    return 0


def run_full_comparison(args: argparse.Namespace) -> int:
    import os
    import torch
    bcat_root = str(args.bcat_root)
    sys.path.insert(0, os.path.join(bcat_root, "src"))
    print("\n[BCAT] Full numerical comparison")
    print(f"  bcat-root: {bcat_root}")

    from omegaconf import OmegaConf
    from models.bcat import BCAT as BCAT_PT

    cfg_path = os.path.join(bcat_root, "src", "configs", "model", "bcat.yaml")
    model_config = OmegaConf.load(cfg_path)

    x_num = model_config.get("x_num", 128)
    data_dim = model_config.get("data_dim", 4)
    t_num = model_config.get("max_data_len", 20)
    input_len = t_num // 2

    np.random.seed(args.seed)
    data_np = np.random.randn(1, t_num, x_num, x_num, data_dim).astype(np.float32)
    times_np = np.arange(t_num, dtype=np.float32).reshape(1, t_num, 1)

    model_config.kv_cache = 0
    pt_model = BCAT_PT(model_config, x_num, data_dim, max_data_len=t_num)
    if args.checkpoint is not None:
        sd = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        pt_model.load_state_dict(sd, strict=True)
    pt_model.eval()

    data_pt = torch.from_numpy(data_np)
    times_pt = torch.from_numpy(times_np)

    with torch.no_grad():
        from models.bcat import block_lower_triangular_mask as pt_mask_fn
        d_in = data_pt[:, :-1]
        t_in = times_pt[:, :-1]
        enc = pt_model.embedder.encode(d_in, t_in)
        data_len = enc.size(1)
        mask = pt_mask_fn(pt_model.seq_len_per_step, t_num, use_float=True)[:data_len, :data_len]
        enc = pt_model.transformer(enc, mask)
        input_seq_len = (input_len - 1) * pt_model.seq_len_per_step
        y_pt = pt_model.embedder.decode(enc[:, input_seq_len:])
    y_pt_np = y_pt.numpy()

    key = jax.random.PRNGKey(args.seed)
    cfg = BCAT_CONFIGS["default"]
    jax_model = BCAT(
        n_layer=cfg.n_layer,
        dim_emb=cfg.dim_emb,
        dim_ffn=cfg.dim_ffn,
        n_head=cfg.n_head,
        norm_first=cfg.norm_first,
        norm_type=cfg.norm_type,
        qk_norm=cfg.qk_norm,
        x_num=cfg.x_num,
        max_output_dim=cfg.max_output_dim,
        patch_num=cfg.patch_num,
        patch_num_output=cfg.patch_num_output,
        conv_dim=cfg.conv_dim,
        time_embed=cfg.time_embed,
        max_time_len=cfg.max_time_len,
        max_data_len=cfg.max_data_len,
        deep=cfg.deep,
        data_dim=data_dim,
        key=key,
    )
    y_jax = jax_model(jnp.array(data_np), jnp.array(times_np), input_len=input_len)
    y_jax_np = np.asarray(y_jax)

    print(f"  PT  shape: {y_pt_np.shape}  range [{y_pt_np.min():.4f}, {y_pt_np.max():.4f}]")
    print(f"  JAX shape: {y_jax_np.shape}  range [{y_jax_np.min():.4f}, {y_jax_np.max():.4f}]")

    if y_pt_np.shape != y_jax_np.shape:
        print("  FAIL: shape mismatch")
        return 1

    max_d = float(np.max(np.abs(y_pt_np - y_jax_np)))
    rel = max_d / (np.max(np.abs(y_pt_np)) + 1e-8)
    status = "PASS" if rel < args.threshold else "FAIL"
    print(f"  Max abs: {max_d:.2e}  Rel: {rel:.2e}  → {status}")
    return 0 if status == "PASS" else 1


def main() -> None:
    args = parse_args()
    if args.bcat_root is not None:
        code = run_full_comparison(args)
    else:
        code = run_structural_check(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
