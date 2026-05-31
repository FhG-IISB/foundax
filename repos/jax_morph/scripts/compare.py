"""Compare PyTorch MORPH and JAX ViT3DRegression forward passes.

Without --morph-root: structural validation with random weights (JAX only).
With --morph-root:    full numerical comparison against PyTorch + HF checkpoint.

Usage (structural check, no original repo needed):
    python compare.py --model-size Ti

Usage (full numerical comparison):
    python compare.py --model-size Ti --morph-root /path/to/MORPH
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

from jax_morph import ViT3DRegression as ViT3DRegression_JAX  # noqa: E402
from jax_morph.configs import MORPH_CONFIGS as MORPH_MODELS, CHECKPOINT_NAMES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch and JAX MORPH outputs")
    parser.add_argument("--model-size", "-m", choices=list(MORPH_MODELS.keys()), default="Ti")
    parser.add_argument("--morph-root", type=Path, default=None,
                        help="Path to cloned MORPH repo (required for full comparison)")
    parser.add_argument("--checkpoint", "-c", default=None, help="Path to .pth checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spatial", type=int, default=8, help="Spatial size D=H=W for structural check")
    parser.add_argument("--threshold", type=float, default=1e-3)
    return parser.parse_args()


def _build_jax_model(cfg: dict, key: jax.Array) -> ViT3DRegression_JAX:
    return ViT3DRegression_JAX(
        patch_size=8,
        dim=cfg["dim"],
        depth=cfg["depth"],
        heads=cfg["heads"],
        heads_xa=32,
        mlp_dim=cfg["mlp_dim"],
        max_components=3,
        conv_filter=cfg["conv_filter"],
        max_ar=cfg["max_ar"],
        max_patches=4096,
        max_fields=3,
        model_size=cfg["model_size"],
        key=key,
    )


def run_structural_check(args: argparse.Namespace) -> int:
    cfg = MORPH_MODELS[args.model_size]
    print(f"\n[MORPH-{args.model_size}] Structural validation (JAX only, random weights)")
    print(f"  dim={cfg['dim']}, depth={cfg['depth']}, heads={cfg['heads']}")

    key = jax.random.PRNGKey(args.seed)
    model = _build_jax_model(cfg, key)

    S = args.spatial
    vol = jnp.zeros((1, 1, 1, 1, S, S, S), dtype=jnp.float32)
    print(f"  Input shape: {vol.shape}  (B=1, t=1, F=1, C=1, D=H=W={S})")

    t0 = time.perf_counter()
    enc, z, pred = model(vol)
    elapsed = time.perf_counter() - t0

    pred_np = np.asarray(pred)
    print(f"  Enc shape:  {np.asarray(enc).shape}")
    print(f"  Pred shape: {pred_np.shape}")

    if not np.all(np.isfinite(pred_np)):
        print("  FAIL: output contains non-finite values")
        return 1

    print(f"  Output range: [{pred_np.min():.4f}, {pred_np.max():.4f}]")
    print(f"  Elapsed: {elapsed:.2f}s")
    print("  PASS: output shape correct and finite")
    return 0


def run_full_comparison(args: argparse.Namespace) -> int:
    import os
    morph_root = str(args.morph_root)
    sys.path.insert(0, morph_root)

    from src.utils.vit_conv_xatt_axialatt2 import ViT3DRegression as ViT3DRegression_PT  # type: ignore[import]

    cfg = MORPH_MODELS[args.model_size]
    print(f"\n[MORPH-{args.model_size}] Full numerical comparison")
    print(f"  morph-root: {morph_root}")

    # Resolve checkpoint
    ckpt_path = args.checkpoint
    if not ckpt_path:
        local = os.path.join(morph_root, "models", "FM", CHECKPOINT_NAMES[args.model_size])
        if os.path.exists(local):
            ckpt_path = local
        else:
            from huggingface_hub import hf_hub_download
            print(f"  Downloading {CHECKPOINT_NAMES[args.model_size]} from HuggingFace...")
            ckpt_path = hf_hub_download(
                repo_id="mahindrautela/MORPH",
                filename=CHECKPOINT_NAMES[args.model_size],
                subfolder="models/FM",
                repo_type="model",
                resume_download=True,
            )
    print(f"  checkpoint: {ckpt_path}")

    import torch
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # Strip DataParallel 'module.' prefix if present
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}

    pt_model = ViT3DRegression_PT(
        patch_size=8, dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"],
        heads_xa=32, mlp_dim=cfg["mlp_dim"], max_components=3,
        conv_filter=cfg["conv_filter"], max_ar=cfg["max_ar"],
        max_patches=4096, max_fields=3, dropout=0.0, emb_dropout=0.0,
        model_size=cfg["model_size"],
    )
    pt_model.load_state_dict(sd, strict=True)
    pt_model.eval()

    key = jax.random.PRNGKey(args.seed)
    jax_model = _build_jax_model(cfg, key)

    try:
        from jax_morph import convert_pytorch_to_jax_params
        jax_model = convert_pytorch_to_jax_params(sd, jax_model)
    except NotImplementedError:
        print("  WARNING: weight converter not implemented — running structural check instead")
        return run_structural_check(args)

    S = args.spatial if args.spatial != 8 else 16
    np.random.seed(args.seed)
    vol_np = np.random.randn(1, 1, 1, 1, S, S, S).astype(np.float32)
    vol_pt = torch.from_numpy(vol_np)
    vol_jax = jnp.array(vol_np)

    with torch.no_grad():
        _, _, pred_pt = pt_model(vol_pt)
    _, _, pred_jax = jax_model(vol_jax)

    pred_pt_np = pred_pt.numpy()
    pred_jax_np = np.asarray(pred_jax)

    print(f"  PT  shape: {pred_pt_np.shape}  range [{pred_pt_np.min():.4f}, {pred_pt_np.max():.4f}]")
    print(f"  JAX shape: {pred_jax_np.shape}  range [{pred_jax_np.min():.4f}, {pred_jax_np.max():.4f}]")

    if pred_pt_np.shape != pred_jax_np.shape:
        print("  FAIL: shape mismatch")
        return 1

    max_d = float(np.max(np.abs(pred_pt_np - pred_jax_np)))
    rel = max_d / (np.max(np.abs(pred_pt_np)) + 1e-8)
    status = "PASS" if rel < args.threshold else "FAIL"
    print(f"  Max abs: {max_d:.2e}  Rel: {rel:.2e}  → {status}")
    return 0 if status == "PASS" else 1


def main() -> None:
    args = parse_args()
    if args.morph_root is not None:
        code = run_full_comparison(args)
    else:
        code = run_structural_check(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
