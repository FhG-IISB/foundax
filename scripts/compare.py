#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _default_projects_root() -> Path:
    return Path(__file__).resolve().parents[1] / "repos"


def _run(cmd: List[str], cwd: Path) -> int:
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(cwd), check=False).returncode


def _dispatch_script(
    projects_root: Path,
    repo_name: str,
    script_rel: str,
    extra_args: List[str],
) -> int:
    repo = projects_root / repo_name
    script = repo / script_rel
    if not repo.exists():
        print(f"ERROR: repository not found: {repo}")
        return 2
    if not script.exists():
        print(f"ERROR: script not found: {script}")
        return 2

    cmd = [sys.executable, str(script), *extra_args]
    return _run(cmd, cwd=repo)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified comparison entrypoint for JAX foundation model repos"
    )
    parser.add_argument(
        "model",
        choices=[
            "morph",
            "mpp",
            "pdeformer2",
            "poseidon",
            "prose",
            "walrus",
            "dpot",
            "bcat",
        ],
        help="Model family to compare",
    )
    parser.add_argument(
        "--projects-root",
        type=Path,
        default=_default_projects_root(),
        help="Path containing vendored jax_* repositories (default: foundax/repos)",
    )
    parser.add_argument(
        "--prose-variant",
        choices=["fd", "pde", "ode"],
        default="fd",
        help="PROSE variant when model=prose",
    )

    args, extra = parser.parse_known_args()

    if args.model == "morph":
        code = _dispatch_script(
            args.projects_root, "jax_morph", "scripts/compare.py", extra
        )
    elif args.model == "mpp":
        code = _dispatch_script(
            args.projects_root, "jax_mpp", "scripts/compare.py", extra
        )
    elif args.model == "pdeformer2":
        code = _dispatch_script(
            args.projects_root, "jax_pdeformer2", "scripts/compare.py", extra
        )
    elif args.model == "poseidon":
        code = _dispatch_script(
            args.projects_root, "jax_poseidon", "scripts/compare.py", extra
        )
    elif args.model == "prose":
        prose_script = {
            "fd": "scripts/compare.py",
            "pde": "scripts/compare_pde.py",
            "ode": "scripts/compare_ode.py",
        }[args.prose_variant]
        code = _dispatch_script(args.projects_root, "jax_prose", prose_script, extra)
    elif args.model == "walrus":
        code = _dispatch_script(
            args.projects_root, "jax_walrus", "scripts/compare.py", extra
        )
    elif args.model == "dpot":
        code = _dispatch_script(
            args.projects_root, "jax_dpot", "scripts/compare.py", extra
        )
    elif args.model == "bcat":
        print(
            "ERROR: compare is not implemented for bcat (no upstream compare script available)."
        )
        print(
            "Use convert for bcat, or add a dedicated comparison implementation in jax_bcat."
        )
        code = 2
    else:
        print(f"Unsupported model: {args.model}")
        code = 2

    raise SystemExit(code)


if __name__ == "__main__":
    main()
