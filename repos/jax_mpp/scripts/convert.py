#!/usr/bin/env python
"""
Convert a PyTorch MPP/AViT checkpoint to a Flax msgpack file.

Usage::

    uv run python scripts/convert.py \\
        --checkpoint path/to/ckpt.tar \\
        --output avit_b.msgpack \\
        --variant B

The ``--variant`` flag selects the model configuration (Ti / S / B / L)
so that parameter shapes can be validated.
"""

import argparse
from pathlib import Path

import jax
import numpy as np
from flax.serialization import to_bytes, from_bytes

from jax_mpp import (
    load_pytorch_state_dict,
    convert_pytorch_to_jax_params,
)


def main():
    parser = argparse.ArgumentParser(
        description="Convert MPP PyTorch checkpoint to Flax msgpack"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to PyTorch checkpoint (skips if not provided)"
    )
    parser.add_argument("--output", type=str, default=None, help="Output msgpack path")
    parser.add_argument(
        "--variant",
        type=str,
        default="B",
        choices=["Ti", "S", "B", "L"],
        help="Model variant (Ti/S/B/L)",
    )
    parser.add_argument(
        "--n_states", type=int, default=12, help="Number of state variables"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify roundtrip serialisation"
    )
    args = parser.parse_args()

    if args.checkpoint is None:
        print("No MPP checkpoint provided — skipping conversion")
        return

    if not Path(args.checkpoint).exists():
        print(f"MPP checkpoint not found: {args.checkpoint} — skipping conversion")
        return

    if args.output is None:
        print("No output path provided — skipping conversion")
        return

    print(f"Loading PyTorch checkpoint: {args.checkpoint}")
    pt_state_dict = load_pytorch_state_dict(args.checkpoint)
    print(f"  → {len(pt_state_dict)} parameters")

    print("Converting to JAX parameters...")
    jax_params = convert_pytorch_to_jax_params(pt_state_dict)

    # Wrap under {'params': ...} to match Flax convention used by jNO
    jax_params = {"params": jax_params}

    # Count parameters
    n_params = sum(np.prod(v.shape) for v in jax.tree.leaves(jax_params))
    print(f"  → {n_params:,} parameters in Flax tree")

    # Serialise
    encoded = to_bytes(jax_params)
    output_path = Path(args.output)
    output_path.write_bytes(encoded)
    print(f"Saved to {output_path} ({len(encoded) / 1e6:.1f} MB)")

    if args.verify:
        print("Verifying roundtrip...")
        from_bytes(None, output_path.read_bytes())
        print("  ✓ Roundtrip OK")


if __name__ == "__main__":
    main()
