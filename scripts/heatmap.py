#!/usr/bin/env python3
"""
Cross-model JAX ↔ PyTorch agreement heatmap.

Runs (or loads cached) forward and gradient comparisons for each foundation
model and renders a single triangular heatmap:

  • Lower triangle  → max |forward_jax − forward_pt|
  • Upper triangle  → max |grad_jax − grad_pt|
  • Diagonal        → annotated with model name

Only same-model comparisons are valid (different architectures/input shapes
make cross-model entries meaningless), so off-diagonal cells are masked.

Usage
-----
  # Run all comparisons live (requires each model's PyTorch dep + checkpoint):
  python scripts/heatmap.py --run

  # Load cached results from a previous run:
  python scripts/heatmap.py --results results.json

  # Just run a subset of models:
  python scripts/heatmap.py --run --models morph poseidon walrus
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------------------------
# Per-model comparison runners
# ---------------------------------------------------------------------------
# Each function returns (forward_max_abs_diff, gradient_max_abs_diff).
# They are intentionally self-contained so a missing dependency for one model
# doesn't prevent the others from running.
# ---------------------------------------------------------------------------


def _compare_morph(
    projects_root: Path,
    model_size: str = "Ti",
    seed: int = 42,
    spatial: int = 16,
) -> Tuple[float, float]:
    import torch, jax, jax.numpy as jnp

    jax.config.update("jax_platform_name", "cpu")
    sys.path.insert(0, str(projects_root / "jax_morph"))

    from jax_morph import ViT3DRegression as ViT3DRegression_JAX
    from jax_morph import load_pytorch_state_dict, convert_pytorch_to_jax_params
    from jax_morph.configs import MORPH_CONFIGS, CHECKPOINT_NAMES

    import os

    MORPH_ROOT = os.environ.get("MORPH_ROOT", os.path.expanduser("~/MORPH"))
    sys.path.insert(0, MORPH_ROOT)
    from src.utils.vit_conv_xatt_axialatt2 import ViT3DRegression as ViT3DRegression_PT

    cfg = MORPH_CONFIGS[model_size]
    ckpt_path = os.path.join(MORPH_ROOT, "models", "FM", CHECKPOINT_NAMES[model_size])

    np.random.seed(seed)
    S = spatial
    vol_np = np.random.randn(1, 1, 1, 1, S, S, S).astype(np.float32)

    # --- PyTorch ---
    pt_model = ViT3DRegression_PT(
        patch_size=8, dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"],
        heads_xa=32, mlp_dim=cfg["mlp_dim"], max_components=3,
        conv_filter=cfg["conv_filter"], max_ar=cfg["max_ar"],
        max_patches=4096, max_fields=3, dropout=0.0, emb_dropout=0.0,
        model_size=cfg["model_size"],
    )
    sd = load_pytorch_state_dict(ckpt_path)
    pt_model.load_state_dict(sd, strict=True)
    pt_model.eval()

    vol_pt = torch.from_numpy(vol_np)

    # Forward
    with torch.no_grad():
        _, _, pred_pt = pt_model(vol_pt)
    pred_pt_np = pred_pt.detach().cpu().numpy()

    # Gradient
    vol_pt_g = torch.from_numpy(vol_np).requires_grad_(True)
    _, _, pred_pt_g = pt_model(vol_pt_g)
    pred_pt_g.sum().backward()
    grad_pt = vol_pt_g.grad.detach().cpu().numpy()

    # --- JAX ---
    jax_model = ViT3DRegression_JAX(
        patch_size=8, dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"],
        heads_xa=32, mlp_dim=cfg["mlp_dim"], max_components=3,
        conv_filter=cfg["conv_filter"], max_ar=cfg["max_ar"],
        max_patches=4096, max_fields=3, dropout=0.0, emb_dropout=0.0,
        model_size=cfg["model_size"],
    )
    vol_jax = jnp.array(vol_np)
    rng = jax.random.PRNGKey(0)
    jax_params = jax_model.init(rng, vol_jax, deterministic=True)
    jax_params = convert_pytorch_to_jax_params(sd, jax_params, heads_xa=32)

    # Forward
    _, _, pred_jax = jax_model.apply(jax_params, vol_jax, deterministic=True)
    pred_jax_np = np.array(pred_jax)

    # Gradient
    def _fwd(x):
        _, _, pred = jax_model.apply(jax_params, x, deterministic=True)
        return jnp.sum(pred)
    grad_jax = np.array(jax.grad(_fwd)(vol_jax))

    fwd_diff = float(np.max(np.abs(pred_pt_np - pred_jax_np)))
    grad_diff = float(np.max(np.abs(grad_pt - grad_jax)))
    return fwd_diff, grad_diff


def _compare_mpp(
    projects_root: Path,
    variant: str = "Ti",
    seed: int = 42,
    resolution: int = 128,
) -> Tuple[float, float]:
    import torch, jax, jax.numpy as jnp

    jax.config.update("jax_platform_name", "cpu")
    sys.path.insert(0, str(projects_root / "jax_mpp"))
    from jax_mpp import load_pytorch_state_dict, convert_pytorch_to_jax_params
    from jax_mpp.configs import AVIT_CONFIGS, _make_model

    import os
    MPP_ROOT = os.environ.get("MPP_ROOT", os.path.expanduser("~/multiple_physics_pretraining"))
    sys.path.insert(0, MPP_ROOT)
    from models.avit import build_avit

    cfg = AVIT_CONFIGS[variant]
    ckpt_dir = os.environ.get("MPP_CHECKPOINT", os.path.join(MPP_ROOT, f"MPP_AViT_{variant}"))

    np.random.seed(seed)
    n_states = cfg["n_states"]
    T, B, C, H, W = 4, 1, n_states, resolution, resolution
    x_np = np.random.randn(T, B, C, H, W).astype(np.float32)
    bcs_np = np.zeros((B, 4), dtype=np.int64)
    labels_pt = [list(range(C))]
    labels_jax = jnp.arange(C)

    pt_state = load_pytorch_state_dict(ckpt_dir)

    # --- PyTorch ---
    class Params:
        pass
    params = Params()
    params.embed_dim = cfg["embed_dim"]
    params.processor_blocks = cfg["processor_blocks"]
    params.n_states = n_states
    params.num_heads = cfg["num_heads"]
    params.patch_size = (16, 16)
    params.bias_type = "rel"
    params.block_type = "axial"
    params.space_type = "axial_attention"
    params.time_type = "attention"
    params.gradient_checkpointing = False

    pt_model = build_avit(params)
    pt_model.load_state_dict(pt_state, strict=True)
    pt_model.eval()

    x_pt = torch.from_numpy(x_np)
    bcs_pt = torch.from_numpy(bcs_np)

    with torch.no_grad():
        y_pt = pt_model(x_pt, labels_pt, bcs_pt).numpy()

    # Gradient
    x_pt_g = torch.from_numpy(x_np).requires_grad_(True)
    bcs_pt_g = torch.from_numpy(bcs_np)
    y_pt_g = pt_model(x_pt_g, labels_pt, bcs_pt_g)
    y_pt_g.sum().backward()
    grad_pt = x_pt_g.grad.detach().cpu().numpy()

    # --- JAX ---
    jax_params = convert_pytorch_to_jax_params(pt_state, verbose=False)
    jax_model = _make_model(variant)

    x_jax = jnp.array(x_np)
    bcs_jax = jnp.array(bcs_np)

    y_jax = np.array(
        jax_model.apply({"params": jax_params}, x_jax, labels_jax, bcs_jax, deterministic=True)
    )

    def _fwd(x):
        return jnp.sum(
            jax_model.apply({"params": jax_params}, x, labels_jax, bcs_jax, deterministic=True)
        )
    grad_jax = np.array(jax.grad(_fwd)(x_jax))

    fwd_diff = float(np.max(np.abs(y_pt - y_jax)))
    grad_diff = float(np.max(np.abs(grad_pt - grad_jax)))
    return fwd_diff, grad_diff


def _compare_poseidon(
    projects_root: Path,
    seed: int = 42,
    resolution: int = 128,
    num_channels: int = 4,
) -> Tuple[float, float]:
    import torch, jax, jax.numpy as jnp

    jax.config.update("jax_platform_name", "cpu")
    sys.path.insert(0, str(projects_root / "jax_poseidon"))
    from jax_poseidon import PoseidonModel as PoseidonJAX
    from jax_poseidon import load_pytorch_state_dict, convert_pytorch_to_jax_params
    from jax_poseidon.configs import POSEIDON_CONFIGS

    import os
    POSEIDON_ROOT = os.environ.get("POSEIDON_ROOT", os.path.expanduser("~/poseidon"))
    sys.path.insert(0, POSEIDON_ROOT)

    from transformers import AutoModelForImageSegmentation

    cfg = POSEIDON_CONFIGS.get("T", POSEIDON_CONFIGS[list(POSEIDON_CONFIGS.keys())[0]])
    ckpt = os.environ.get("POSEIDON_CHECKPOINT", os.path.join(POSEIDON_ROOT, "checkpoints", "poseidon-T"))

    np.random.seed(seed)
    H = W = resolution
    x_np = np.random.randn(1, H, W, num_channels).astype(np.float32)
    time_np = np.array([0.5], dtype=np.float32)

    # --- PyTorch ---
    pt_model = AutoModelForImageSegmentation.from_pretrained(ckpt)
    pt_model.eval()

    x_pt = torch.from_numpy(x_np).permute(0, 3, 1, 2)
    time_pt = torch.from_numpy(time_np)

    with torch.no_grad():
        y_pt = pt_model(pixel_values=x_pt, time=time_pt).output
    y_pt_np = y_pt.cpu().numpy().transpose(0, 2, 3, 1)

    x_pt_g = torch.from_numpy(x_np).permute(0, 3, 1, 2).requires_grad_(True)
    y_pt_g = pt_model(pixel_values=x_pt_g, time=time_pt).output
    y_pt_g.sum().backward()
    grad_pt = x_pt_g.grad.detach().cpu().numpy().transpose(0, 2, 3, 1)

    # --- JAX ---
    jax_model = PoseidonJAX(**cfg)
    x_jax = jnp.array(x_np)
    time_jax = jnp.array(time_np)
    rng = jax.random.PRNGKey(0)

    jax_params = jax_model.init(rng, pixel_values=x_jax, time=time_jax, deterministic=True)
    pt_sd = load_pytorch_state_dict(ckpt)
    jax_params = convert_pytorch_to_jax_params(pt_sd, jax_params)

    y_jax = np.array(
        jax_model.apply(jax_params, pixel_values=x_jax, time=time_jax, deterministic=True).output
    )

    def _fwd(x):
        return jnp.sum(
            jax_model.apply(jax_params, pixel_values=x, time=time_jax, deterministic=True).output
        )
    grad_jax = np.array(jax.grad(_fwd)(x_jax))

    fwd_diff = float(np.max(np.abs(y_pt_np - y_jax)))
    grad_diff = float(np.max(np.abs(grad_pt - grad_jax)))
    return fwd_diff, grad_diff


def _compare_prose(
    projects_root: Path,
    seed: int = 42,
) -> Tuple[float, float]:
    import torch, jax, jax.numpy as jnp

    jax.config.update("jax_platform_name", "cpu")
    sys.path.insert(0, str(projects_root / "jax_prose"))
    from jax_prose.prose_fd_2to1 import PROSE2to1 as PROSE_JAX
    from jax_prose import load_pytorch_state_dict, convert_pytorch_to_jax_params
    from jax_prose.configs import PROSE_CONFIGS

    import os
    PROSE_ROOT = os.environ.get("PROSE_ROOT", os.path.expanduser("~/PROSE"))
    sys.path.insert(0, PROSE_ROOT)
    from models.transformer_wrappers import PROSE_2to1 as PROSE_PT

    cfg = PROSE_CONFIGS.get("fd", PROSE_CONFIGS[list(PROSE_CONFIGS.keys())[0]])
    ckpt = os.environ.get("PROSE_CHECKPOINT")

    np.random.seed(seed)
    input_len, x_num, max_output_dim = 10, 128, 4
    symbol_len = 48
    data_np = np.random.randn(1, input_len, x_num, x_num, max_output_dim).astype(np.float32)
    symbols_np = np.random.randint(0, 100, (1, symbol_len)).astype(np.int64)
    t_in = np.array([[0.0, 1.0]], dtype=np.float32)
    t_out = np.array([[1.1, 2.0]], dtype=np.float32)

    # --- PyTorch ---
    pt_model = PROSE_PT(**cfg)
    if ckpt:
        sd = load_pytorch_state_dict(ckpt)
        pt_model.load_state_dict(sd, strict=True)
    pt_model.eval()

    data_pt = torch.from_numpy(data_np)
    sym_pt = torch.from_numpy(symbols_np)
    tin_pt = torch.from_numpy(t_in)
    tout_pt = torch.from_numpy(t_out)

    with torch.no_grad():
        y_pt = pt_model(data_pt, sym_pt, tin_pt, tout_pt).detach().cpu().numpy()

    data_pt_g = torch.from_numpy(data_np).requires_grad_(True)
    y_pt_g = pt_model(data_pt_g, sym_pt, tin_pt, tout_pt)
    y_pt_g.sum().backward()
    grad_pt = data_pt_g.grad.detach().cpu().numpy()

    # --- JAX ---
    jax_model = PROSE_JAX(**cfg)
    data_jax = jnp.array(data_np)
    sym_jax = jnp.array(symbols_np)
    tin_jax = jnp.array(t_in)
    tout_jax = jnp.array(t_out)
    rng = jax.random.PRNGKey(0)

    jax_params = jax_model.init(rng, data_jax, sym_jax, tin_jax, tout_jax, deterministic=True)
    if ckpt:
        sd = load_pytorch_state_dict(ckpt)
        jax_params = convert_pytorch_to_jax_params(sd, jax_params)

    y_jax = np.array(
        jax_model.apply(jax_params, data_jax, sym_jax, tin_jax, tout_jax, deterministic=True)
    )

    def _fwd(d):
        return jnp.sum(
            jax_model.apply(jax_params, d, sym_jax, tin_jax, tout_jax, deterministic=True)
        )
    grad_jax = np.array(jax.grad(_fwd)(data_jax))

    fwd_diff = float(np.max(np.abs(y_pt - y_jax)))
    grad_diff = float(np.max(np.abs(grad_pt - grad_jax)))
    return fwd_diff, grad_diff


def _compare_walrus(
    projects_root: Path,
    seed: int = 42,
) -> Tuple[float, float]:
    import torch, jax, jax.numpy as jnp

    jax.config.update("jax_platform_name", "cpu")
    sys.path.insert(0, str(projects_root / "jax_walrus"))
    from jax_walrus import IsotropicModel as WalrusJAX
    from jax_walrus import load_pytorch_state_dict, convert_pytorch_to_jax_params
    from jax_walrus.configs import WALRUS_CONFIGS

    import os
    WALRUS_ROOT = os.environ.get("WALRUS_ROOT", os.path.expanduser("~/WALRUS"))
    sys.path.insert(0, WALRUS_ROOT)

    cfg = WALRUS_CONFIGS.get("default", WALRUS_CONFIGS[list(WALRUS_CONFIGS.keys())[0]])
    ckpt = os.environ.get("WALRUS_CHECKPOINT")

    np.random.seed(seed)
    T, B, C, H, W, D = 2, 1, 5, 64, 64, 16
    x_np = np.random.randn(B, T, H, W, D, C).astype(np.float32)

    # --- PyTorch (import upstream) ---
    from walrus_model import build_walrus as build_walrus_pt  # type: ignore

    pt_model = build_walrus_pt(**cfg)
    if ckpt:
        sd = load_pytorch_state_dict(ckpt)
        pt_model.load_state_dict(sd, strict=True)
    pt_model.eval()

    # PT expects (T, B, C, H, W, D) → transpose from (B, T, H, W, D, C)
    x_pt_layout = np.transpose(x_np, (1, 0, 5, 2, 3, 4))
    x_pt = torch.from_numpy(x_pt_layout)

    with torch.no_grad():
        y_pt = pt_model(x_pt).detach().cpu().numpy()

    x_pt_g = torch.from_numpy(x_pt_layout).requires_grad_(True)
    y_pt_g = pt_model(x_pt_g)
    y_pt_g.sum().backward()
    grad_pt = x_pt_g.grad.detach().cpu().numpy()

    # --- JAX ---
    jax_model = WalrusJAX(**cfg)
    x_jax = jnp.array(x_np)
    rng = jax.random.PRNGKey(0)

    jax_params = jax_model.init(rng, x_jax, deterministic=True)
    if ckpt:
        sd = load_pytorch_state_dict(ckpt)
        jax_params = convert_pytorch_to_jax_params(sd, jax_params)

    y_jax_raw = np.array(jax_model.apply(jax_params, x_jax, deterministic=True))
    # Convert JAX output (B,T,H,W,D,C) → PT layout (T,B,C,H,W,D) for comparison
    y_jax = np.transpose(y_jax_raw, (1, 0, 5, 2, 3, 4))

    def _fwd(x):
        return jnp.sum(jax_model.apply(jax_params, x, deterministic=True))
    grad_jax_raw = np.array(jax.grad(_fwd)(x_jax))
    # Same layout transpose for gradient comparison
    grad_jax = np.transpose(grad_jax_raw, (1, 0, 5, 2, 3, 4))

    fwd_diff = float(np.max(np.abs(y_pt - y_jax)))
    grad_diff = float(np.max(np.abs(grad_pt - grad_jax)))
    return fwd_diff, grad_diff


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

RUNNERS = {
    "morph":      _compare_morph,
    "mpp":        _compare_mpp,
    "poseidon":   _compare_poseidon,
    "prose":      _compare_prose,
    "walrus":     _compare_walrus,
}

ALL_MODELS = list(RUNNERS.keys())


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_comparisons(
    models: List[str],
    projects_root: Path,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Return {model: {"forward": float|None, "gradient": float|None}}."""
    results: Dict[str, Dict[str, Optional[float]]] = {}
    for name in models:
        print(f"\n{'='*60}")
        print(f"  Running comparison: {name}")
        print(f"{'='*60}")
        try:
            fwd, grad = RUNNERS[name](projects_root)
            results[name] = {"forward": fwd, "gradient": grad}
            print(f"  ✓ {name}: fwd={fwd:.3e}  grad={grad:.3e}")
        except Exception:
            traceback.print_exc()
            results[name] = {"forward": None, "gradient": None}
            print(f"  ✗ {name}: SKIPPED (missing dependency or checkpoint)")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heatmap(
    results: Dict[str, Dict[str, Optional[float]]],
    output: str = "heatmap.png",
) -> None:
    """
    Triangular heatmap: lower‐triangle = forward diff, upper = gradient diff.

    Both axes list the same model names (x = JAX, y = PyTorch).
    Only diagonal entries carry data (cross-model comparison is undefined).
    """
    models = sorted(results.keys())
    n = len(models)

    # Build matrices (NaN for off-diagonal / missing)
    fwd_mat = np.full((n, n), np.nan)
    grad_mat = np.full((n, n), np.nan)

    for idx, m in enumerate(models):
        r = results[m]
        if r["forward"] is not None:
            fwd_mat[idx, idx] = r["forward"]
        if r["gradient"] is not None:
            grad_mat[idx, idx] = r["gradient"]

    # Combine into one matrix: lower tri = forward, upper tri = gradient
    combined = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i > j:
                # Lower triangle → forward diff of model at row index
                combined[i, j] = fwd_mat[i, i]
            elif i < j:
                # Upper triangle → gradient diff of model at col index
                combined[i, j] = grad_mat[j, j]
            else:
                # Diagonal → forward diff (shown in lower-triangle color)
                combined[i, j] = fwd_mat[i, i]

    # Collect non-NaN values for color scale
    valid = combined[~np.isnan(combined)]
    if len(valid) == 0:
        print("No valid results to plot.")
        return

    vmin = valid.min()
    vmax = valid.max()
    if vmin == vmax:
        vmax = vmin + 1e-10

    # Log-scale norm for potentially wide-ranging diffs
    norm = mcolors.LogNorm(vmin=max(vmin, 1e-12), vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Masked array for NaN cells
    masked = np.ma.masked_invalid(combined)
    cmap = plt.cm.RdYlGn_r  # red = large diff, green = small diff

    im = ax.pcolormesh(
        np.arange(n + 1), np.arange(n + 1), masked,
        cmap=cmap, norm=norm, edgecolors="white", linewidth=2,
    )

    # Axes
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels([f"{m}\n(JAX)" for m in models], fontsize=10)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels([f"{m}\n(PyTorch)" for m in models], fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel("JAX model", fontsize=12, labelpad=10)
    ax.set_ylabel("PyTorch model", fontsize=12, labelpad=10)

    # Annotate cells with values
    for i in range(n):
        for j in range(n):
            val = combined[i, j]
            if not np.isnan(val):
                ax.text(
                    j + 0.5, i + 0.5, f"{val:.1e}",
                    ha="center", va="center", fontsize=8,
                    color="white" if val > np.median(valid) else "black",
                    fontweight="bold",
                )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Max absolute difference", fontsize=11)

    # Legend for triangle meaning
    ax.set_title(
        "JAX ↔ PyTorch Agreement\n"
        "Lower triangle: forward pass  |  Upper triangle: gradient (backward)",
        fontsize=12, fontweight="bold", pad=15,
    )

    # Draw diagonal line for visual separation
    ax.plot([0, n], [0, n], color="gray", linewidth=1.5, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    print(f"\nHeatmap saved to {output}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Foundation model JAX↔PyTorch agreement heatmap"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--run", action="store_true",
        help="Run comparisons live (requires model deps & checkpoints)",
    )
    mode.add_argument(
        "--results", type=Path,
        help="Load pre-computed results from JSON",
    )
    parser.add_argument(
        "--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS,
        help="Models to include (default: all)",
    )
    parser.add_argument(
        "--projects-root", type=Path,
        default=Path(__file__).resolve().parents[1] / "repos",
        help="Path to vendored jax_* repos (default: foundax/repos)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="heatmap.png",
        help="Output image path (default: heatmap.png)",
    )
    parser.add_argument(
        "--save-results", type=Path, default=None,
        help="Save results to JSON for later reuse",
    )
    args = parser.parse_args()

    if args.results:
        with open(args.results) as f:
            results = json.load(f)
        # Filter to requested models
        results = {k: v for k, v in results.items() if k in args.models}
    else:
        results = run_comparisons(args.models, args.projects_root)

    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.save_results}")

    plot_heatmap(results, output=args.output)


if __name__ == "__main__":
    main()
