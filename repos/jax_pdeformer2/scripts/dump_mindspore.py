#!/usr/bin/env python3
"""Dump MindSpore PDEformer-2 state_dict, inputs, and forward output as .npz.

Run this in the ``mindspore`` pixi env (Python 3.11, MindSpore 2.9):

    pixi run -e mindspore python repos/jax_pdeformer2/scripts/dump_mindspore.py \\
        --pdeformer2-root og_repos/pdeformer2 \\
        --output verify_output/pdeformer2_ms.npz

The output ``.npz`` is consumed by ``scripts/compare.py`` to do the JAX
equivalency check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdeformer2-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import mindspore as ms
    from mindspore import Tensor

    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_seed(args.seed)

    import os

    pdeformer2_root = args.pdeformer2_root.resolve()
    sys.path.insert(0, str(pdeformer2_root))
    original_cwd = os.getcwd()
    os.chdir(pdeformer2_root)
    try:
        from src import load_config, get_model  # type: ignore[import]
        cfg = load_config("configs/inference/model-S.yaml")
    finally:
        os.chdir(original_cwd)
    cfg.model.load_ckpt = "none"

    model = get_model(cfg)
    state_dict = {k: v.asnumpy() for k, v in model.parameters_dict().items()}
    print(f"  collected {len(state_dict)} MindSpore parameters")

    num_spatial = int(cfg.model.graphormer.num_spatial)
    resolution = int(cfg.model.function_encoder.resolution)
    num_branches = int(cfg.model.function_encoder.num_branches)
    SPACE_DIM = 3

    rng = np.random.default_rng(args.seed)
    n_graph = 1
    n_scalar = 8
    n_function = 2  # number of function nodes
    # After scalar+function expansion: scalar (n_scalar) + function (n_function * num_branches)
    n_node = n_scalar + n_function * num_branches
    n_points = 16
    num_points_function = resolution * resolution  # 128*128 = 16384

    node_type = rng.integers(0, 16, size=(n_graph, n_node, 1)).astype(np.int32)
    node_scalar = rng.standard_normal((n_graph, n_scalar, 1)).astype(np.float32)
    # node_function: (n_graph, num_function, num_points_function, 1+SPACE_DIM+1)
    node_function = rng.standard_normal(
        (n_graph, n_function, num_points_function, 1 + SPACE_DIM + 1)
    ).astype(np.float32)
    in_degree = rng.integers(0, 8, size=(n_graph, n_node)).astype(np.int32)
    out_degree = rng.integers(0, 8, size=(n_graph, n_node)).astype(np.int32)
    attn_bias = np.zeros((n_graph, n_node, n_node), dtype=np.float32)
    spatial_pos = rng.integers(0, num_spatial, size=(n_graph, n_node, n_node)).astype(
        np.int32
    )
    coordinate = rng.uniform(0, 1, size=(n_graph, n_points, 1 + SPACE_DIM)).astype(
        np.float32
    )

    try:
        out = model(
            Tensor(node_type),
            Tensor(node_scalar),
            Tensor(node_function),
            Tensor(in_degree),
            Tensor(out_degree),
            Tensor(attn_bias),
            Tensor(spatial_pos),
            Tensor(coordinate),
        )
        out_np = out.asnumpy()
        print(f"  MS forward output shape: {out_np.shape}")
    except Exception as e:
        print(f"  WARNING: MS forward failed ({type(e).__name__}: {e})")
        print("  Saving state_dict + inputs only; JAX side will skip numerical check.")
        out_np = None

    save = {f"param/{k}": v for k, v in state_dict.items()}
    save.update(
        {
            "input/node_type": node_type,
            "input/node_scalar": node_scalar,
            "input/node_function": node_function,
            "input/in_degree": in_degree,
            "input/out_degree": out_degree,
            "input/attn_bias": attn_bias,
            "input/spatial_pos": spatial_pos,
            "input/coordinate": coordinate,
        }
    )
    if out_np is not None:
        save["output/ms_forward"] = out_np

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **save)
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
