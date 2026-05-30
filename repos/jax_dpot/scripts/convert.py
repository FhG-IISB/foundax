#!/usr/bin/env python3
"""DPOT weight conversion.

Without --checkpoint, skips gracefully (no pretrained checkpoint configured by default).
With --checkpoint and --dpot-root, converts the PyTorch .pth checkpoint to JAX msgpack.

Usage (skip / structural check only):
    python convert.py

Usage (full conversion):
    python convert.py --variant Ti --checkpoint /path/to/DPOT_Ti.pth --output /path/to/dpot_Ti.msgpack
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import jax

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jax_dpot import DPOTNet, DPOT_CONFIGS  # noqa: E402
from jax_dpot.convert_weights import (  # noqa: E402
    load_pytorch_state_dict,
    convert_pytorch_to_jax_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DPOT PyTorch weights to JAX msgpack")
    parser.add_argument(
        "--variant",
        choices=["Ti", "S", "M", "L", "H"],
        default="Ti",
        help="Model variant (default: Ti)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to PyTorch .pth checkpoint (skips conversion if not provided)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .msgpack path (default: dpot_<variant>.msgpack next to checkpoint)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.checkpoint is None:
        print("No DPOT checkpoint provided — skipping conversion")
        raise SystemExit(0)

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint} — skipping conversion")
        raise SystemExit(0)

    output = args.output or args.checkpoint.parent / f"dpot_{args.variant.lower()}.msgpack"

    cfg = DPOT_CONFIGS[args.variant]
    print(f"[DPOT-{args.variant}] Converting {args.checkpoint} → {output}")

    state_dict = load_pytorch_state_dict(str(args.checkpoint))
    print(f"  Loaded {len(state_dict)} parameters from PyTorch checkpoint")

    flat_params = convert_pytorch_to_jax_params(state_dict, variant=args.variant)
    print(f"  Converted {len(flat_params)} parameters to JAX format")

    from flax.serialization import to_bytes
    key = jax.random.PRNGKey(0)
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

    x_dummy = np.zeros(
        (1, cfg.img_size, cfg.img_size, cfg.in_timesteps, cfg.in_channels), dtype=np.float32
    )
    import jax.numpy as jnp
    ref_out, _ = model(jnp.array(x_dummy))
    print(f"  Reference output shape: {ref_out.shape}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(to_bytes({"params": flat_params}))
    print(f"  Saved to {output}")


if __name__ == "__main__":
    main()
