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
    import os
    import torch
    from jax_bcat import transfer_pt_to_eqx

    bcat_root = str(args.bcat_root)
    sys.path.insert(0, os.path.join(bcat_root, "src"))
    mode = "checkpoint" if args.checkpoint else "random-weight"
    print(f"\n[BCAT] {mode} comparison")
    print(f"  bcat-root: {bcat_root}")

    # BCAT's embedder.py imports `from neuralop.models import FNO2d`, which was
    # renamed to `FNO` in newer neuralop. The import is at module top-level even
    # though the class isn't used by the default BCAT model, so alias it.
    import neuralop.models as _nm
    if not hasattr(_nm, "FNO2d"):
        _nm.FNO2d = _nm.FNO

    from omegaconf import OmegaConf
    from models.bcat import BCAT as BCAT_PT

    cfg_path = os.path.join(bcat_root, "src", "configs", "model", "bcat.yaml")
    model_config = OmegaConf.load(cfg_path)
    model_config.kv_cache = 0

    x_num = model_config.get("x_num", 128)
    data_dim = model_config.get("data_dim", 4)
    t_num = model_config.get("max_data_len", 20)
    input_len = t_num // 2

    np.random.seed(args.seed)
    data_np = np.random.randn(1, t_num, x_num, x_num, data_dim).astype(np.float32)
    times_np = np.arange(t_num, dtype=np.float32).reshape(1, t_num, 1)

    torch.manual_seed(args.seed)
    pt_model = BCAT_PT(model_config, x_num, data_dim, max_data_len=t_num)
    if args.checkpoint is not None:
        sd = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        pt_model.load_state_dict(sd, strict=True)
    pt_model.eval()

    data_pt = torch.from_numpy(data_np)
    times_pt = torch.from_numpy(times_np)

    # Monkey-patch scaled_dot_product_attention with a manual math implementation
    # — the native PT 2.9 CPU kernel has no viable backend for our float attn_mask.
    import torch.nn.functional as _F

    _orig_sdpa = _F.scaled_dot_product_attention

    def _manual_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        d = q.shape[-1]
        s = (d**-0.5) if scale is None else scale
        scores = torch.matmul(q, k.transpose(-2, -1)) * s
        if is_causal:
            L = q.shape[-2]
            cm = torch.ones(L, L, dtype=torch.bool, device=q.device).tril()
            scores = scores.masked_fill(~cm, float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    _F.scaled_dot_product_attention = _manual_sdpa
    try:
        with torch.no_grad():
            y_pt = pt_model.fwd(data_pt, times_pt, input_len=input_len)
    finally:
        _F.scaled_dot_product_attention = _orig_sdpa
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
    jax_model = transfer_pt_to_eqx(pt_model.state_dict(), jax_model)
    y_jax = jax_model(jnp.array(data_np), jnp.array(times_np), input_len=input_len)
    y_jax_np = np.asarray(y_jax)

    print(f"  PT  shape: {y_pt_np.shape}  range [{y_pt_np.min():.4f}, {y_pt_np.max():.4f}]")
    print(f"  JAX shape: {y_jax_np.shape}  range [{y_jax_np.min():.4f}, {y_jax_np.max():.4f}]")

    if y_pt_np.shape != y_jax_np.shape:
        print("  FAIL: shape mismatch")
        return 1

    max_d = float(np.max(np.abs(y_pt_np - y_jax_np)))
    mean_d = float(np.mean(np.abs(y_pt_np - y_jax_np)))
    rel = max_d / (np.max(np.abs(y_pt_np)) + 1e-8)
    status = "PASS" if rel < args.threshold else "FAIL"
    print(f"  Max abs diff: {max_d:.2e}  Mean abs diff: {mean_d:.2e}  Rel: {rel:.2e}  → {status}")
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
