#!/usr/bin/env python3
"""Compare PyTorch DPOT and JAX DPOTNet forward passes.

When run without --dpot-root / --checkpoint, performs a structural validation:
initialises the JAX model with random weights, runs a forward pass, and checks
output shapes and finite values.

When run with --dpot-root and --checkpoint, loads the original PyTorch model,
converts the checkpoint to JAX, runs both models on the same input, and
reports relative L2 differences.

Usage (structural check, no original repo needed):
    python compare.py --variant Ti

Usage (full numerical comparison):
    python compare.py --variant Ti \\
        --dpot-root /path/to/DPOT \\
        --checkpoint /path/to/DPOT_Ti.pth
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jax_dpot import DPOTNet, DPOT_CONFIGS  # noqa: E402
from jax_dpot.convert_weights import (  # noqa: E402
    load_pytorch_state_dict,
    convert_pytorch_to_jax_params,
    load_jax_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch and JAX DPOT forward passes"
    )
    parser.add_argument(
        "--variant",
        choices=["Ti", "S", "M", "L", "H"],
        default="Ti",
        help="Model variant to compare (default: Ti)",
    )
    parser.add_argument(
        "--dpot-root",
        type=Path,
        default=None,
        help="Path to the original DPOT repository (required for full numerical comparison)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to PyTorch .pth checkpoint (required together with --dpot-root)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="Maximum allowed relative L2 difference (default: 1e-3)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_dummy_input(cfg, batch: int = 1, seed: int = 0) -> np.ndarray:
    """Build a random input tensor matching DPOT's expected shape (B, H, W, T, C)."""
    rng = np.random.default_rng(seed)
    # Channel dim = in_channels (state) — grid coords are added internally
    return rng.standard_normal(
        (batch, cfg.img_size, cfg.img_size, cfg.in_timesteps, cfg.in_channels)
    ).astype(np.float32)


def _run_jax(model, x_np: np.ndarray) -> tuple[np.ndarray, float]:
    x_jax = jnp.array(x_np)
    t0 = time.perf_counter()
    out, _ = model(x_jax)
    out = np.asarray(out)
    elapsed = time.perf_counter() - t0
    return out, elapsed


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.flatten() - b.flatten()
    denom = np.linalg.norm(b.flatten()) + 1e-8
    return float(np.linalg.norm(diff) / denom)


# ── structural validation (JAX only) ────────────────────────────────────────


def run_structural_check(args: argparse.Namespace) -> int:
    cfg = DPOT_CONFIGS[args.variant]
    print(f"\n[DPOT-{args.variant}] Structural validation (JAX only, random weights)")
    print(f"  img_size={cfg.img_size}, embed_dim={cfg.embed_dim}, depth={cfg.depth}")

    key = jax.random.PRNGKey(args.seed)
    model = DPOTNet(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        mixing_type=cfg.mixing_type,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        in_timesteps=cfg.in_timesteps,
        out_timesteps=cfg.out_timesteps,
        n_blocks=cfg.n_blocks,
        embed_dim=cfg.embed_dim,
        out_layer_dim=cfg.out_layer_dim,
        depth=cfg.depth,
        modes=cfg.modes,
        mlp_ratio=cfg.mlp_ratio,
        n_cls=cfg.n_cls,
        normalize=cfg.normalize,
        time_agg=cfg.time_agg,
        key=key,
    )

    x_np = _make_dummy_input(cfg, seed=args.seed)
    print(f"  Input shape: {x_np.shape}")

    out, elapsed = _run_jax(model, x_np)

    expected_out = (1, cfg.img_size, cfg.img_size, cfg.out_timesteps, cfg.out_channels)
    print(f"  Output shape: {out.shape}  (expected {expected_out})")

    if out.shape != expected_out:
        print("  FAIL: output shape mismatch")
        return 1

    if not np.all(np.isfinite(out)):
        print("  FAIL: output contains non-finite values")
        return 1

    print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
    print(f"  Elapsed: {elapsed:.2f}s")
    print("  PASS: output shape correct and finite")
    return 0


# ── full numerical comparison (PyTorch + JAX) ────────────────────────────────


def run_full_comparison(args: argparse.Namespace) -> int:
    import torch

    cfg = DPOT_CONFIGS[args.variant]
    dpot_root = args.dpot_root

    print(f"\n[DPOT-{args.variant}] Full numerical comparison")
    print(f"  dpot-root:  {dpot_root}")
    print(f"  checkpoint: {args.checkpoint}")

    # ── Load PyTorch model ─────────────────────────────────────────────
    sys.path.insert(0, str(dpot_root))
    # The original DPOT model class is in models/DPOT.py or models/dpot.py
    for mod_path in ("models.DPOT", "models.dpot", "models"):
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            DPOTNet_PT = mod.DPOTNet
            print(f"  Imported PyTorch DPOTNet from {mod_path}")
            break
        except (ImportError, AttributeError):
            continue
    else:
        print("  ERROR: could not import PyTorch DPOTNet from dpot-root")
        print("  Make sure --dpot-root points to the DPOT repository root")
        return 1

    torch.manual_seed(args.seed)
    pt_model = DPOTNet_PT(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        mixing_type=cfg.mixing_type,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        in_timesteps=cfg.in_timesteps,
        out_timesteps=cfg.out_timesteps,
        n_blocks=cfg.n_blocks,
        embed_dim=cfg.embed_dim,
        out_layer_dim=cfg.out_layer_dim,
        depth=cfg.depth,
        modes=cfg.modes,
        mlp_ratio=cfg.mlp_ratio,
        n_cls=cfg.n_cls,
        normalize=cfg.normalize,
        time_agg=cfg.time_agg,
    ).eval()

    # ── Load checkpoint into PyTorch model ────────────────────────────
    print("  Loading checkpoint …")
    state_dict = load_pytorch_state_dict(str(args.checkpoint))
    pt_model.load_state_dict(
        {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
         for k, v in state_dict.items()},
        strict=False,
    )

    # ── Build JAX model and load converted weights ─────────────────────
    key = jax.random.PRNGKey(args.seed)
    jax_model = DPOTNet(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        mixing_type=cfg.mixing_type,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        in_timesteps=cfg.in_timesteps,
        out_timesteps=cfg.out_timesteps,
        n_blocks=cfg.n_blocks,
        embed_dim=cfg.embed_dim,
        out_layer_dim=cfg.out_layer_dim,
        depth=cfg.depth,
        modes=cfg.modes,
        mlp_ratio=cfg.mlp_ratio,
        n_cls=cfg.n_cls,
        normalize=cfg.normalize,
        time_agg=cfg.time_agg,
        key=key,
    )

    flat_params = convert_pytorch_to_jax_params(state_dict, variant=args.variant)
    jax_model = load_jax_params(flat_params, jax_model)

    # ── Run both on the same input ────────────────────────────────────
    x_np = _make_dummy_input(cfg, seed=args.seed)
    print(f"  Input shape: {x_np.shape}")

    # PyTorch forward — input is (B, H, W, T, C), model may expect (B, C, H, W) or (B, H, W, T, C)
    with torch.no_grad():
        x_pt = torch.from_numpy(x_np)
        try:
            pt_out, _ = pt_model(x_pt)
        except Exception:
            pt_out = pt_model(x_pt)
            if isinstance(pt_out, tuple):
                pt_out = pt_out[0]
    pt_out_np = pt_out.numpy()

    # JAX forward
    jax_out_np, elapsed = _run_jax(jax_model, x_np)

    # ── Compare ───────────────────────────────────────────────────────
    print(f"\n  PT  shape: {pt_out_np.shape}  range [{pt_out_np.min():.4f}, {pt_out_np.max():.4f}]")
    print(f"  JAX shape: {jax_out_np.shape}  range [{jax_out_np.min():.4f}, {jax_out_np.max():.4f}]")

    if pt_out_np.shape != jax_out_np.shape:
        print(f"  FAIL: output shape mismatch {pt_out_np.shape} vs {jax_out_np.shape}")
        return 1

    rel = _rel_l2(pt_out_np, jax_out_np)
    max_abs = float(np.abs(pt_out_np - jax_out_np).max())
    status = "PASS" if rel < args.threshold else "FAIL"
    print(f"\n  Relative L2: {rel:.2e}  Max abs: {max_abs:.2e}  → {status}")
    print(f"  (threshold: {args.threshold:.2e})")

    return 0 if status == "PASS" else 1


# ── entry ────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    if args.dpot_root is not None and args.checkpoint is not None:
        code = run_full_comparison(args)
    else:
        if args.dpot_root is not None or args.checkpoint is not None:
            print("WARNING: both --dpot-root and --checkpoint are required for full comparison.")
            print("         Falling back to structural check.")
        code = run_structural_check(args)

    raise SystemExit(code)


if __name__ == "__main__":
    main()
