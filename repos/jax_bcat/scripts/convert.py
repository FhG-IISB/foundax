#!/usr/bin/env python3
"""Convert PyTorch BCAT checkpoint to JAX/Flax msgpack format."""
from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import to_bytes

from jax_bcat import bcat_default, load_pytorch_state_dict, convert_pytorch_to_jax_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BCAT PyTorch checkpoint to JAX msgpack")
    parser.add_argument("--input", required=True, help="Path to PyTorch checkpoint")
    parser.add_argument("--output", default="model.msgpack", help="Output path")
    args = parser.parse_args()

    pt_state = load_pytorch_state_dict(args.input)

    model = bcat_default()
    dummy_data = jnp.zeros((1, 20, 128, 128, 4), dtype=jnp.float32)
    dummy_times = jnp.zeros((1, 20, 1), dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), dummy_data, dummy_times, input_len=10)
    jax_params = convert_pytorch_to_jax_params(pt_state, variables["params"])

    with open(args.output, "wb") as f:
        f.write(to_bytes({"params": jax_params}))
    print(f"Saved JAX params to {args.output}")


if __name__ == "__main__":
    main()
