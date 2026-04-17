#!/usr/bin/env python
"""MindSpore PDEformer-2 helper — run in a Python 3.12 env with MindSpore.

Usage:
    python ms_pdeformer_helper.py <output_dir> <ogrepo_path> [seed]

Creates <output_dir>/ms_params.npz  (model weights, MS naming)
        <output_dir>/ms_io.npz     (inputs + forward output + gradient)
"""
import os
import sys

import numpy as np


def main(output_dir: str, ogrepo_path: str, seed: int = 42) -> None:
    import mindspore as ms
    from mindspore import Tensor, context, ops
    from mindspore import dtype as mstype
    from omegaconf import OmegaConf

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

    sys.path.insert(0, ogrepo_path)
    from src.cell.pdeformer import PDEformer

    cfg = OmegaConf.create(
        {
            "graphormer": {
                "num_node_type": 8,
                "num_in_degree": 4,
                "num_out_degree": 4,
                "num_spatial": 4,
                "num_encoder_layers": 2,
                "embed_dim": 32,
                "ffn_embed_dim": 64,
                "num_heads": 4,
                "pre_layernorm": True,
                "dropout": 0.0,
                "attention_dropout": 0.0,
                "activation_dropout": 0.0,
            },
            "scalar_encoder": {"dim_hidden": 16, "num_layers": 2},
            "function_encoder": {
                "type": "cnn2dv3",
                "num_branches": 4,
                "resolution": 128,
                "conv2d_input_txyz": False,
                "cnn_keep_nchw": True,
            },
            "inr": {
                "type": "poly_inr",
                "num_layers": 3,
                "dim_hidden": 16,
                "poly_inr": {
                    "enable_affine": False,
                    "enable_shift": True,
                    "enable_scale": True,
                    "activation_fn": "sin",
                    "affine_act_fn": "identity",
                },
            },
            "hypernet": {"dim_hidden": 16, "num_layers": 2, "shared": False},
            "multi_inr": {"enable": False},
        }
    )

    model = PDEformer(cfg, compute_dtype=mstype.float32)
    model.set_train(False)

    # ---- deterministic inputs via numpy ----
    rng = np.random.RandomState(seed)
    NUM_SCALAR, NUM_FUNCTION = 4, 2
    NUM_BRANCHES = 4
    N_NODE = NUM_SCALAR + NUM_FUNCTION * NUM_BRANCHES  # 12
    RESOLUTION = 128
    NUM_POINTS_FUNC = RESOLUTION * RESOLUTION
    NUM_POINTS = 10
    n_graph = 1

    inputs_np = {
        "node_type": rng.randint(1, 8, (n_graph, N_NODE, 1)).astype(np.int32),
        "node_scalar": rng.randn(n_graph, NUM_SCALAR, 1).astype(np.float32),
        "node_function": rng.randn(n_graph, NUM_FUNCTION, NUM_POINTS_FUNC, 5).astype(
            np.float32
        ),
        "in_degree": rng.randint(0, 4, (n_graph, N_NODE)).astype(np.int32),
        "out_degree": rng.randint(0, 4, (n_graph, N_NODE)).astype(np.int32),
        "attn_bias": rng.randn(n_graph, N_NODE, N_NODE).astype(np.float32),
        "spatial_pos": rng.randint(0, 4, (n_graph, N_NODE, N_NODE)).astype(np.int32),
        "coordinate": rng.rand(n_graph, NUM_POINTS, 4).astype(np.float32),
    }

    ms_inputs = {k: Tensor(v) for k, v in inputs_np.items()}

    # ---- forward ----
    ms_output = model.construct(**ms_inputs).asnumpy()

    # ---- gradient w.r.t. coordinate ----
    coord_tensor = Tensor(inputs_np["coordinate"])

    def loss_fn(coord):
        return ops.sum(
            model.construct(
                ms_inputs["node_type"],
                ms_inputs["node_scalar"],
                ms_inputs["node_function"],
                ms_inputs["in_degree"],
                ms_inputs["out_degree"],
                ms_inputs["attn_bias"],
                ms_inputs["spatial_pos"],
                coord,
            )
        )

    try:
        grad_fn = ms.grad(loss_fn)
        ms_grad = grad_fn(coord_tensor).asnumpy()
    except Exception as exc:
        print(f"WARNING: gradient failed ({exc}), saving zeros", file=sys.stderr)
        ms_grad = np.zeros_like(inputs_np["coordinate"])

    # ---- save weights ----
    ms_params = {}
    for param in model.get_parameters():
        ms_params[param.name] = param.asnumpy()

    np.savez(os.path.join(output_dir, "ms_params.npz"), **ms_params)
    np.savez(
        os.path.join(output_dir, "ms_io.npz"),
        output=ms_output,
        grad=ms_grad,
        **inputs_np,
    )

    print(
        f"OK  params={len(ms_params)}  output={ms_output.shape}  "
        f"grad={ms_grad.shape}",
        flush=True,
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <output_dir> <ogrepo_path> [seed]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 42)
