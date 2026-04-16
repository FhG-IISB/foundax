#!/usr/bin/env python
r"""Convert all MindSpore .ckpt files to JAX .msgpack format.

This script performs a two-stage conversion:
  1. Load .ckpt with MindSpore → save intermediate .npz
  2. Load .npz → convert weight names/shapes → save as Flax .msgpack

Usage (run from the pdeformer2-jax directory):
    python scripts/convert_ckpt_to_msgpack.py

The script expects these .ckpt files in the repo root:
    pdeformer2-small.ckpt
    pdeformer2-base.ckpt
    pdeformer2-fast.ckpt

Output .msgpack files are written next to the .ckpt files.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from flax import serialization
from flax.core import unfreeze

# ── project imports ──────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from jax_pdeformer2.utils import (
    convert_mindspore_to_jax,
    load_numpy_weights,
)
from jax_pdeformer2.pdeformer import (
    create_pdeformer_from_config,
    PDEFORMER_SMALL_CONFIG,
    PDEFORMER_BASE_CONFIG,
    PDEFORMER_FAST_CONFIG,
)

# ── paths ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent

# Path to Python interpreter with MindSpore installed

# Option 1: Use the same Python as the current script

MS_PYTHON = sys.executable

# Option 2: If MindSpore is in a different environment, specify the path:

# MS_PYTHON = Path("/path/to/mindspore/venv/bin/python")

# Checkpoint → config mapping

CHECKPOINTS = {
    "pdeformer2-small": PDEFORMER_SMALL_CONFIG,
    "pdeformer2-base": PDEFORMER_BASE_CONFIG,
    "pdeformer2-fast": PDEFORMER_FAST_CONFIG,
}


def ckpt_to_npz(ckpt_path: Path, npz_path: Path) -> None:
    """Use MindSpore (via subprocess) to dump a .ckpt as .npz."""
    if npz_path.exists():
        print(f"  [skip] {npz_path.name} already exists")
        return

    print(f"  Loading {ckpt_path.name} with MindSpore …")
    script = textwrap.dedent(
        f"""\
        import numpy as np
        from mindspore import load_checkpoint
        param_dict = load_checkpoint(r"{ckpt_path}")
        weights = {{}}
        for name, param in param_dict.items():
            weights[name] = param.asnumpy()
        np.savez_compressed(r"{npz_path}", **weights)
        print(f"  Saved {{len(weights)}} params → {npz_path.name}")
    """
    )
    subprocess.run(
        [str(MS_PYTHON), "-c", script],
        check=True,
    )


def _prune_to_ref(loaded: dict, ref: dict) -> dict:
    """Recursively keep only the keys present in *ref*."""
    pruned = {}
    for k, v in loaded.items():
        if k not in ref:
            continue
        if isinstance(v, dict) and isinstance(ref[k], dict):
            pruned[k] = _prune_to_ref(v, ref[k])
        else:
            pruned[k] = v
    return pruned


def npz_to_msgpack(npz_path: Path, msgpack_path: Path, config: dict) -> None:
    """Load .npz, convert to JAX format, verify shapes, save .msgpack."""
    print(f"  Converting {npz_path.name} → {msgpack_path.name} …")

    # Load raw MindSpore weights
    ms_weights = load_numpy_weights(str(npz_path))
    print(f"    {len(ms_weights)} MindSpore parameters loaded")

    # Convert names + shapes
    jax_params = convert_mindspore_to_jax(ms_weights)

    # Instantiate model to determine expected parameter tree
    model = create_pdeformer_from_config({"model": config})
    from jax_pdeformer2.utils import create_dummy_inputs

    dummy = create_dummy_inputs()
    ref_params = model.init(jax.random.PRNGKey(0), **dummy)

    # Prune checkpoint keys that are not in the model (e.g. unused
    # shift/scale hypernets in the "fast" variant)
    jax_params = _prune_to_ref(jax_params, ref_params)

    n_loaded = len(jax.tree_util.tree_leaves(jax_params))
    n_ref = len(jax.tree_util.tree_leaves(ref_params))
    assert (
        n_loaded == n_ref
    ), f"Leaf count mismatch after pruning: {n_loaded} vs {n_ref}"
    print(f"    Leaves: {n_loaded}")

    # Count total elements
    total = sum(x.size for x in jax.tree_util.tree_leaves(jax_params))
    print(f"    Total parameters: {total:,}")

    # Serialize to msgpack
    params_dict = (
        unfreeze(jax_params) if hasattr(jax_params, "unfreeze") else dict(jax_params)
    )
    raw_bytes = serialization.msgpack_serialize(params_dict)
    msgpack_path.write_bytes(raw_bytes)
    mb = len(raw_bytes) / 1e6
    print(f"    Written {msgpack_path.name}  ({mb:.1f} MB)")


# ── main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import jax  # imported here so top-level stays light

    for stem, config in CHECKPOINTS.items():
        ckpt_path = Path(f"/home/b8cl/projects/DATA/pdeformer/{stem}.ckpt")
        npz_path = Path(f"/home/b8cl/projects/DATA/pdeformer/{stem}.npz")
        msgpack_path = Path(f"/home/b8cl/projects/DATA/pdeformer/{stem}.msgpack")

        if not ckpt_path.exists():
            print(f"[skip] {ckpt_path.name} not found")
            continue

        print(f"\n{'='*60}")
        print(f" {stem}")
        print(f"{'='*60}")

        # Stage 1: .ckpt → .npz  (MindSpore subprocess)
        ckpt_to_npz(ckpt_path, npz_path)

        # Stage 2: .npz → .msgpack  (JAX / Flax)
        npz_to_msgpack(npz_path, msgpack_path, config)

    print("\n✓ All conversions complete.")
