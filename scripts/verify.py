#!/usr/bin/env python3
"""Hydra-based pipeline to clone, download, convert, and compare all foundation models.

Usage (full suite):
    python scripts/verify.py

Usage (single model):
    python scripts/verify.py 'model=[walrus]'

Usage (skip clone/download if repos already present):
    python scripts/verify.py steps=[convert,compare]

Usage (walrus only, custom output dir):
    python scripts/verify.py 'model=[walrus]' outdir=/tmp/foundax_verify

Overrideable config keys (see conf/verify.yaml):
    outdir            where to write checkpoints, msgpacks, logs
    repos_dir         where to clone original PyTorch repositories
    steps             list of: clone download convert compare
    force_download    re-download even if checkpoint already exists
    force_convert     re-run conversion even if msgpack already exists
    processor_blocks  walrus-specific (default: 40)
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# ── repo root (two levels up from scripts/) ──────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


# ── step helpers ──────────────────────────────────────────────────────────────


def _run(cmd: list[str], *, cwd: Path = REPO_ROOT) -> int:
    print("  $", " ".join(shlex.quote(str(c)) for c in cmd))
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    return result.returncode


def _interpolate(args: list[str], ctx: dict[str, str]) -> list[str]:
    """Replace {key} placeholders in arg strings using ctx."""
    out = []
    for a in args:
        for k, v in ctx.items():
            a = a.replace(f"{{{k}}}", str(v))
        out.append(a)
    return out


def step_clone(model_cfg: DictConfig, repos_dir: Path) -> int:
    dest = repos_dir / model_cfg.name
    if dest.exists():
        print(f"  [clone] {model_cfg.name}: already present at {dest}")
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = model_cfg.get("github_url")
    if not url:
        print(f"  [clone] {model_cfg.name}: no github_url configured — skipping")
        return 0
    return _run(["git", "clone", "--depth=1", url, str(dest)])


def step_download(model_cfg: DictConfig, checkpoints_dir: Path, force: bool) -> int:
    hf = model_cfg.get("hf_checkpoint")
    if not hf:
        print(f"  [download] {model_cfg.name}: no HuggingFace checkpoint configured — skipping")
        return 0

    dest_dir = checkpoints_dir / model_cfg.name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / hf["filename"]

    if dest_file.exists() and not force:
        print(f"  [download] {model_cfg.name}: {dest_file.name} already present")
        return 0

    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=hf["repo_id"],
            filename=hf["filename"],
            local_dir=str(dest_dir),
            resume_download=not force,
        )
        print(f"  [download] {model_cfg.name}: saved to {path}")
        return 0
    except Exception as e:
        print(f"  [download] {model_cfg.name}: ERROR — {e}")
        return 1


def step_convert(model_cfg: DictConfig, ctx: dict[str, str], force: bool) -> int:
    name = model_cfg.name
    extra = _interpolate(list(model_cfg.get("convert_extra_args", [])), ctx)

    prose_variant = model_cfg.get("prose_variant", None)
    cmd = [sys.executable, str(SCRIPTS_DIR / "convert.py"), name]
    if prose_variant:
        cmd += ["--prose-variant", prose_variant]
    cmd += extra

    return _run(cmd)


def step_compare(model_cfg: DictConfig, ctx: dict[str, str]) -> int:
    name = model_cfg.name
    if not model_cfg.get("has_compare", True):
        print(f"  [compare] {name}: has_compare=false — skipping")
        return 0

    extra = _interpolate(list(model_cfg.get("compare_extra_args", [])), ctx)

    prose_variant = model_cfg.get("prose_variant", None)
    cmd = [sys.executable, str(SCRIPTS_DIR / "compare.py"), name]
    if prose_variant:
        cmd += ["--prose-variant", prose_variant]
    cmd += extra

    return _run(cmd)


# ── main ──────────────────────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="../conf", config_name="verify")
def main(cfg: DictConfig) -> None:
    outdir = Path(cfg.outdir)
    repos_dir = Path(cfg.repos_dir)
    checkpoints_dir = outdir / "checkpoints"
    steps = list(cfg.steps)

    outdir.mkdir(parents=True, exist_ok=True)
    repos_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("foundax-verify pipeline")
    print(f"  outdir:    {outdir}")
    print(f"  repos_dir: {repos_dir}")
    print(f"  steps:     {steps}")
    print("=" * 70)

    models = cfg.model if isinstance(cfg.model, list) else [cfg.model]

    results: dict[str, dict[str, int]] = {}

    for model_cfg in models:
        name = model_cfg.name
        print(f"\n{'─' * 70}")
        print(f"  Model: {name}")
        print(f"{'─' * 70}")

        # Template context for interpolating compare/convert extra_args
        ctx = {
            "repos_dir": str(repos_dir),
            "checkpoints_dir": str(checkpoints_dir),
            "outdir": str(outdir),
        }

        rc_map: dict[str, int] = {}

        if "clone" in steps and model_cfg.get("compare_needs_pt_repo", False):
            rc = step_clone(model_cfg, repos_dir)
            rc_map["clone"] = rc
            if rc != 0:
                print(f"  [clone] WARNING: exited {rc} — downstream steps may fail")

        if "download" in steps:
            rc = step_download(model_cfg, checkpoints_dir, force=cfg.force_download)
            rc_map["download"] = rc

        if "convert" in steps:
            rc = step_convert(model_cfg, ctx, force=cfg.force_convert)
            rc_map["convert"] = rc

        if "compare" in steps:
            rc = step_compare(model_cfg, ctx)
            rc_map["compare"] = rc

        results[name] = rc_map

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    any_fail = False
    for name, rc_map in results.items():
        worst = max(rc_map.values()) if rc_map else 0
        status = "PASS" if worst == 0 else "FAIL"
        if worst != 0:
            any_fail = True
        step_str = "  ".join(f"{s}={rc}" for s, rc in rc_map.items())
        print(f"  {name:<14} {status}  [{step_str}]")

    print()
    raise SystemExit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
