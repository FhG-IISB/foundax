#!/usr/bin/env python3
"""
TimesFM: forward-pass parity between foundax's Flax wrapper and the
upstream PyTorch implementation.

Both backends load the same pretrained 200M-param checkpoint from
Hugging Face (``google/timesfm-2.5-200m-flax`` and ``-torch``), forecast
the same synthetic series, and we compare the point forecasts.

Two modes:

  1. **Full parity** (default when checkpoints are available locally OR
     ``--download`` is passed): runs both backends end-to-end and
     compares ``point_forecast`` element-wise.
  2. **Structural check** (fallback): instantiates the wrapper without
     loading any checkpoint, just verifies the public surface is sane.
     Used when checkpoints aren't downloaded and ``--download`` is not
     given (saves ~800MB on every CI run).

Usage::

    # Structural check (no download)
    python scripts/compare_timesfm.py

    # Full parity (downloads ~800MB on first run, cached afterwards)
    python scripts/compare_timesfm.py --download
"""

from __future__ import annotations

import argparse
import os
from importlib.util import find_spec
from pathlib import Path

import numpy as np


def _checkpoint_exists_locally(model_id: str) -> bool:
    """True if the HF snapshot is already cached on disk."""
    try:
        hf_home = Path(
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")
        )
        # snapshot_download stores under models--<org>--<name>/snapshots/<sha>/
        org, name = model_id.split("/")
        candidates = list(
            (hf_home / "hub").glob(f"models--{org}--{name}/snapshots/*")
        ) + list(hf_home.glob(f"models--{org}--{name}/snapshots/*"))
        return any(c.is_dir() and any(c.iterdir()) for c in candidates)
    except Exception:
        return False


def _generate_synthetic_series(seed: int) -> list[np.ndarray]:
    """Three deterministic time series of varying length and character."""
    rng = np.random.default_rng(seed)
    t1 = np.linspace(0, 50, 600)
    series1 = np.sin(t1) + 0.3 * np.cos(2.3 * t1) + 0.05 * rng.standard_normal(t1.shape)
    t2 = np.linspace(0, 30, 400)
    series2 = (
        0.5 * t2 / 30.0 + 0.4 * np.sin(0.7 * t2) + 0.1 * rng.standard_normal(t2.shape)
    )
    t3 = np.linspace(0, 100, 1024)
    series3 = np.cumsum(rng.standard_normal(t3.shape)) * 0.05 + 0.3 * np.sin(0.5 * t3)
    return [
        series1.astype(np.float32),
        series2.astype(np.float32),
        series3.astype(np.float32),
    ]


def compare_forward(seed: int) -> bool:
    """Load both backends with HF checkpoints, forecast, compare."""
    import foundax as fx

    print("Loading Flax backend (compiling — may take ~30s)...")
    flax_model = fx.timesfm.flax_200m(max_context=1024, max_horizon=128)
    print("Loading PyTorch backend...")
    torch_model = fx.timesfm.torch_200m(max_context=1024, max_horizon=128)

    series = _generate_synthetic_series(seed)
    print(f"Forecasting {len(series)} series, horizon=64")

    flax_point, flax_q = flax_model(horizon=64, inputs=series)
    torch_point, torch_q = torch_model(horizon=64, inputs=series)

    # Empirically the PT and Flax backends agree to ~1e-6 max abs on the
    # 200M model — well within float32 noise. We set the tolerance at 1e-4
    # to leave some headroom for hardware / BLAS-impl variations.
    diff_point = np.abs(np.asarray(flax_point) - np.asarray(torch_point))
    rel_l2 = np.linalg.norm(diff_point) / (
        np.linalg.norm(np.asarray(torch_point)) + 1e-12
    )
    max_abs = float(np.max(diff_point))
    mean_abs = float(np.mean(diff_point))

    tol = 1e-4
    status = "PASS" if max_abs < tol else "FAIL"
    print(
        f"  [{status}] TimesFM 2.5 200M point forecast "
        f"shape={tuple(flax_point.shape)} "
        f"max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} rel_l2={float(rel_l2):.3e}"
    )
    return status == "PASS"


def run_structural_check() -> int:
    """Check the wrapper imports + surface is callable, without downloading."""
    import foundax as fx

    print("[TimesFM] Structural check (no checkpoint download)")
    print(f"  fx.timesfm is callable:        {callable(fx.timesfm)}")
    print(f"  fx.timesfm.flax_200m exists:   {fx.timesfm.flax_200m is not None}")
    print(f"  fx.timesfm.torch_200m exists:  {fx.timesfm.torch_200m is not None}")
    # Check we can at least import the upstream classes referenced by the
    # wrapper without triggering a download.
    try:
        from timesfm import TimesFM_2p5_200M_flax, TimesFM_2p5_200M_torch  # noqa: F401

        print("  upstream timesfm classes import: ok")
    except Exception as e:
        print(f"  upstream import failed: {e}")
        return 1
    print("  PASS")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--download",
        action="store_true",
        help="Allow downloading the HF checkpoint (~800MB) for full parity test.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if find_spec("timesfm") is None:
        print(
            "timesfm package not installed — install with `pixi add --pypi --feature dev timesfm[flax,torch]`"
        )
        return 1

    flax_cached = _checkpoint_exists_locally("google/timesfm-2.5-200m-flax")
    torch_cached = _checkpoint_exists_locally("google/timesfm-2.5-200m-pytorch")

    if not (args.download or (flax_cached and torch_cached)):
        print("=" * 70)
        print("TimesFM: structural check only (run with --download for full parity)")
        print("=" * 70)
        return run_structural_check()

    print("=" * 70)
    print("TimesFM: foundax Flax wrapper vs upstream PyTorch implementation")
    print(f"  seed: {args.seed}")
    print("=" * 70)
    return 0 if compare_forward(args.seed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
