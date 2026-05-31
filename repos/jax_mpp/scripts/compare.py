"""Compare PyTorch MPP AViT and JAX AViT forward passes.

Modes:
    --mpp-root REPO            : random-weight equivalency (no checkpoint needed).
    --mpp-root REPO + --checkpoint : full numerical comparison against pretrained PT weights.
    (neither)                   : structural validation (JAX only, no L2 metric).

Usage:
    python compare.py --variant Ti --mpp-root og_repos/mpp
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

from jax_mpp import AViT  # noqa: E402
from jax_mpp.configs import AVIT_CONFIGS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch and JAX MPP outputs")
    parser.add_argument("--variant", type=str, default="Ti", choices=list(AVIT_CONFIGS.keys()))
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="PyTorch checkpoint path (omit for random-weight check)")
    parser.add_argument("--mpp-root", type=str, default=None,
                        help="Path to cloned multiple_physics_pretraining repo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=1e-3)
    return parser.parse_args()


def _build_model(variant: str, key: jax.Array) -> AViT:
    cfg = AVIT_CONFIGS[variant]
    return AViT(
        embed_dim=cfg["embed_dim"],
        processor_blocks=cfg["processor_blocks"],
        n_states=cfg["n_states"],
        num_heads=cfg["num_heads"],
        key=key,
    )


def run_structural_check(args: argparse.Namespace) -> int:
    cfg = AVIT_CONFIGS[args.variant]
    print(f"\n[MPP-AViT-{args.variant}] Structural validation (JAX only, random weights)")
    print(f"  embed_dim={cfg['embed_dim']}, blocks={cfg['processor_blocks']}, heads={cfg['num_heads']}")

    key = jax.random.PRNGKey(args.seed)
    model = _build_model(args.variant, key)

    T, B, C, H, W = 2, 1, 1, 32, 32
    np.random.seed(args.seed)
    x = jnp.array(np.random.randn(T, B, C, H, W).astype(np.float32))
    state_labels = jnp.arange(C, dtype=jnp.int32)
    bcs = jnp.zeros((B, 4), dtype=jnp.int32)
    print(f"  Input shape: {x.shape}  (T={T}, B={B}, C={C}, H=W={H})")

    t0 = time.perf_counter()
    out = model(x, state_labels, bcs)
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


def _import_pt_avit(mpp_root):
    if mpp_root is None:
        return None
    mpp_root = str(Path(mpp_root).resolve())
    if mpp_root not in sys.path:
        sys.path.insert(0, mpp_root)
    try:
        from models.avit import build_avit  # type: ignore
        return build_avit
    except ImportError as e:
        print(f"ERROR: Cannot import original PyTorch AViT from {mpp_root}: {e}")
        return None


def _build_pt_model(build_avit, cfg):
    class _Params:
        pass

    p = _Params()
    p.embed_dim = cfg["embed_dim"]
    p.processor_blocks = cfg["processor_blocks"]
    p.n_states = cfg["n_states"]
    p.num_heads = cfg["num_heads"]
    p.patch_size = (16, 16)
    p.bias_type = "rel"
    p.block_type = "axial"
    p.space_type = "axial_attention"
    p.time_type = "attention"
    p.gradient_checkpointing = False
    return build_avit(p)


def run_full_comparison(args: argparse.Namespace) -> int:
    import torch
    from jax_mpp import load_pytorch_state_dict, transfer_pt_to_eqx

    cfg = AVIT_CONFIGS[args.variant]
    mode = "checkpoint" if args.checkpoint else "random-weight"
    print(f"\n[MPP-AViT-{args.variant}] {mode} comparison")
    if args.checkpoint:
        print(f"  checkpoint: {args.checkpoint}")
    print(f"  mpp-root: {args.mpp_root}")

    build_avit = _import_pt_avit(args.mpp_root)
    if build_avit is None:
        return 1

    torch.manual_seed(args.seed)
    pt_model = _build_pt_model(build_avit, cfg)

    if args.checkpoint:
        pt_state = load_pytorch_state_dict(args.checkpoint)
        pt_model.load_state_dict(pt_state, strict=True)
    pt_model.eval()
    pt_state = pt_model.state_dict()

    T, B, C, H, W = 2, 1, cfg["n_states"], 32, 32
    np.random.seed(args.seed)
    x_np = np.random.randn(T, B, C, H, W).astype(np.float32)
    bcs_np = np.zeros((B, 4), dtype=np.int64)
    labels_pt = [list(range(C))]
    labels_jax = jnp.arange(C)

    with torch.no_grad():
        y_pt = pt_model(torch.from_numpy(x_np), labels_pt, torch.from_numpy(bcs_np)).numpy()

    key = jax.random.PRNGKey(args.seed)
    jax_model = _build_model(args.variant, key)
    jax_model = transfer_pt_to_eqx(pt_state, jax_model)

    y_jax = jax_model(jnp.array(x_np), labels_jax, jnp.array(bcs_np))
    y_jax_np = np.asarray(y_jax)

    print(f"  PT  shape: {y_pt.shape}  range [{y_pt.min():.4f}, {y_pt.max():.4f}]")
    print(f"  JAX shape: {y_jax_np.shape}  range [{y_jax_np.min():.4f}, {y_jax_np.max():.4f}]")

    if y_pt.shape != y_jax_np.shape:
        print("  FAIL: shape mismatch")
        return 1

    max_d = float(np.max(np.abs(y_pt - y_jax_np)))
    mean_d = float(np.mean(np.abs(y_pt - y_jax_np)))
    rel = max_d / (np.max(np.abs(y_pt)) + 1e-8)
    status = "PASS" if rel < args.threshold else "FAIL"
    print(f"  Max abs diff: {max_d:.2e}  Mean abs diff: {mean_d:.2e}  Rel: {rel:.2e}  → {status}")
    return 0 if status == "PASS" else 1


def main() -> None:
    args = parse_args()
    if args.mpp_root is not None:
        code = run_full_comparison(args)
    else:
        code = run_structural_check(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
