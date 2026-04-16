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


def _convert_bcat(projects_root: Path, extra_args: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Direct BCAT conversion")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to BCAT PyTorch checkpoint"
    )
    parser.add_argument("--output", required=True, help="Output msgpack path")
    parser.add_argument("--input-len", type=int, default=10)
    parser.add_argument("--output-len", type=int, default=10)
    args = parser.parse_args(extra_args)

    sys.path.insert(0, str(projects_root / "jax_bcat"))

    import importlib

    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    to_bytes = importlib.import_module("flax.serialization").to_bytes
    jax_bcat = importlib.import_module("jax_bcat")

    bcat_default = jax_bcat.bcat_default
    load_pytorch_state_dict = jax_bcat.load_pytorch_state_dict
    convert_pytorch_to_jax_params = jax_bcat.convert_pytorch_to_jax_params

    model = bcat_default()
    data = jnp.zeros(
        (
            1,
            args.input_len + args.output_len,
            model.x_num,
            model.x_num,
            model.max_output_dim,
        ),
        dtype=jnp.float32,
    )
    times = jnp.zeros((1, args.input_len + args.output_len, 1), dtype=jnp.float32)

    variables = model.init(jax.random.PRNGKey(0), data, times, input_len=args.input_len)
    pt_state = load_pytorch_state_dict(args.checkpoint)
    jax_params = convert_pytorch_to_jax_params(pt_state, variables["params"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(to_bytes({"params": jax_params}))
    print(f"Wrote BCAT msgpack: {output_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified conversion entrypoint for JAX foundation model repos"
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
        help="Model family to convert",
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
            args.projects_root, "jax_morph", "scripts/convert.py", extra
        )
    elif args.model == "mpp":
        code = _dispatch_script(
            args.projects_root, "jax_mpp", "scripts/convert.py", extra
        )
    elif args.model == "pdeformer2":
        code = _dispatch_script(
            args.projects_root, "jax_pdeformer2", "scripts/convert.py", extra
        )
    elif args.model == "poseidon":
        code = _dispatch_script(
            args.projects_root, "jax_poseidon", "scripts/convert.py", extra
        )
    elif args.model == "prose":
        prose_script = {
            "fd": "scripts/convert.py",
            "pde": "scripts/convert_pde.py",
            "ode": "scripts/convert_ode.py",
        }[args.prose_variant]
        code = _dispatch_script(args.projects_root, "jax_prose", prose_script, extra)
    elif args.model == "walrus":
        code = _dispatch_script(
            args.projects_root, "jax_walrus", "scripts/convert.py", extra
        )
    elif args.model == "dpot":
        code = _dispatch_script(
            args.projects_root, "jax_dpot", "scripts/convert.py", extra
        )
    elif args.model == "bcat":
        code = _convert_bcat(args.projects_root, extra)
    else:
        print(f"Unsupported model: {args.model}")
        code = 2

    raise SystemExit(code)


if __name__ == "__main__":
    main()
