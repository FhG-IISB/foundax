"""
Test script to verify JAX and MindSpore implementations produce the same output.

This script generates test inputs, runs both models, and compares results.
It needs to be run in two steps:
1. Run with MindSpore to generate reference outputs
2. Run with JAX to compare

Usage:
    # Step 1: In MindSpore environment
    python test_equivalence.py --backend mindspore --save-inputs --save-outputs

    # Step 2: In JAX environment
    python test_equivalence.py --backend jax --compare
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def create_test_inputs(seed: int = 42):
    """Create deterministic test inputs."""
    np.random.seed(seed)

    # Model dimensions (matching PDEformer-Small config)
    n_graph = 1
    num_scalar = 40
    num_function = 3
    num_branches = 4  # from function encoder
    n_node = num_scalar + num_function * num_branches  # = 52
    resolution = 128
    num_points_function = resolution**2  # 16384
    num_points = 100  # query points

    inputs = {
        "node_type": np.random.randint(1, 128, size=(n_graph, n_node, 1)).astype(
            np.int32
        ),
        "node_scalar": np.random.randn(n_graph, num_scalar, 1).astype(np.float32) * 0.1,
        "node_function": np.random.randn(
            n_graph, num_function, num_points_function, 5
        ).astype(np.float32)
        * 0.1,
        "in_degree": np.random.randint(1, 32, size=(n_graph, n_node)).astype(np.int32),
        "out_degree": np.random.randint(1, 32, size=(n_graph, n_node)).astype(np.int32),
        "attn_bias": np.zeros(
            (n_graph, n_node, n_node), dtype=np.float32
        ),  # Use zeros for simplicity
        "spatial_pos": np.random.randint(1, 16, size=(n_graph, n_node, n_node)).astype(
            np.int32
        ),
        "coordinate": np.random.rand(n_graph, num_points, 4).astype(np.float32),
    }

    return inputs


def save_inputs(inputs: dict, path: str = "test_inputs.npz"):
    """Save test inputs to file."""
    np.savez(path, **inputs)
    print(f"Saved inputs to {path}")


def load_inputs(path: str = "test_inputs.npz") -> dict:
    """Load test inputs from file."""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def run_mindspore(
    inputs: dict, save_weights: bool = True, weights_path: str = "test_weights.npz"
):
    """Run the MindSpore model and optionally save initialized weights."""
    try:
        import mindspore as ms
        from mindspore import Tensor, context
        from mindspore import dtype as mstype
    except ImportError:
        print("Error: MindSpore is not installed in this environment.")
        print("Please run this in a MindSpore environment.")
        sys.exit(1)

    # Set context
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from omegaconf import OmegaConf
    from src.cell.pdeformer import PDEformer

    # Create config matching PDEformer-Small
    config = OmegaConf.create(
        {
            "graphormer": {
                "num_node_type": 128,
                "num_in_degree": 32,
                "num_out_degree": 32,
                "num_spatial": 16,
                "num_encoder_layers": 9,
                "embed_dim": 512,
                "ffn_embed_dim": 1024,
                "num_heads": 32,
                "pre_layernorm": True,
                "dropout": 0.0,
                "attention_dropout": 0.0,
                "activation_dropout": 0.0,
                "activation_fn": "gelu",
                "encoder_normalize_before": False,
            },
            "scalar_encoder": {
                "dim_hidden": 256,
                "num_layers": 3,
            },
            "function_encoder": {
                "type": "cnn2dv3",
                "num_branches": 4,
                "resolution": 128,
                "conv2d_input_txyz": False,
                "cnn_keep_nchw": True,
            },
            "inr": {
                "type": "poly_inr",
                "num_layers": 12,
                "dim_hidden": 128,
                "poly_inr": {
                    "enable_affine": False,
                    "enable_shift": True,
                    "enable_scale": True,
                    "modify_he_init": False,
                    "affine_act_fn": "identity",
                    "activation_fn": "sin",
                },
            },
            "hypernet": {
                "dim_hidden": 512,
                "num_layers": 2,
                "shared": False,
            },
            "multi_inr": {
                "enable": False,
            },
        }
    )

    print("Creating MindSpore model...")
    model = PDEformer(config, compute_dtype=mstype.float32)
    model.set_train(False)

    # Convert inputs to MindSpore tensors
    ms_inputs = {
        "node_type": Tensor(inputs["node_type"], mstype.int32),
        "node_scalar": Tensor(inputs["node_scalar"], mstype.float32),
        "node_function": Tensor(inputs["node_function"], mstype.float32),
        "in_degree": Tensor(inputs["in_degree"], mstype.int32),
        "out_degree": Tensor(inputs["out_degree"], mstype.int32),
        "attn_bias": Tensor(inputs["attn_bias"], mstype.float32),
        "spatial_pos": Tensor(inputs["spatial_pos"], mstype.int32),
        "coordinate": Tensor(inputs["coordinate"], mstype.float32),
    }

    print("Running MindSpore forward pass...")
    output = model(
        ms_inputs["node_type"],
        ms_inputs["node_scalar"],
        ms_inputs["node_function"],
        ms_inputs["in_degree"],
        ms_inputs["out_degree"],
        ms_inputs["attn_bias"],
        ms_inputs["spatial_pos"],
        ms_inputs["coordinate"],
    )

    output_np = output.asnumpy()
    print(f"MindSpore output shape: {output_np.shape}")
    print(f"MindSpore output range: [{output_np.min():.6f}, {output_np.max():.6f}]")
    print(f"MindSpore output mean: {output_np.mean():.6f}")
    print(f"MindSpore output std: {output_np.std():.6f}")

    # Save weights for JAX
    if save_weights:
        print(f"Saving model weights to {weights_path}...")
        weights = {}
        for name, param in model.parameters_and_names():
            weights[name] = param.asnumpy()
        np.savez(weights_path, **weights)
        print(f"Saved {len(weights)} parameters")

    return output_np


def run_jax(inputs: dict, weights_path: str = "test_weights.npz"):
    """Run the JAX model with weights loaded from MindSpore."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        print("Error: JAX is not installed in this environment.")
        print("Please run this in a JAX environment.")
        sys.exit(1)

    from jax_pdeformer2 import create_pdeformer_from_config, PDEFORMER_SMALL_CONFIG
    from jax_pdeformer2.utils import convert_mindspore_to_jax

    print("Creating JAX model...")
    model = create_pdeformer_from_config(
        {"model": PDEFORMER_SMALL_CONFIG}, dtype=jnp.float32
    )

    # Convert inputs to JAX arrays
    jax_inputs = {key: jnp.array(val) for key, val in inputs.items()}

    # Load and convert weights if available
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...")
        with np.load(weights_path) as data:
            ms_weights = {key: data[key] for key in data.files}

        print("Converting weights to JAX format...")
        jax_params = convert_mindspore_to_jax(ms_weights)

        # Convert to JAX arrays
        def to_jax_arrays(d):
            if isinstance(d, dict):
                return {k: to_jax_arrays(v) for k, v in d.items()}
            elif isinstance(d, np.ndarray):
                return jnp.array(d)
            return d

        params = to_jax_arrays(jax_params)
    else:
        print("No weights file found, initializing random weights...")
        key = jax.random.PRNGKey(42)
        params = model.init(key, **jax_inputs)

    print("Running JAX forward pass...")
    output = model.apply(params, **jax_inputs)

    output_np = np.array(output)
    print(f"JAX output shape: {output_np.shape}")
    print(f"JAX output range: [{output_np.min():.6f}, {output_np.max():.6f}]")
    print(f"JAX output mean: {output_np.mean():.6f}")
    print(f"JAX output std: {output_np.std():.6f}")

    return output_np


def compare_outputs(ms_output: np.ndarray, jax_output: np.ndarray):
    """Compare MindSpore and JAX outputs."""
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    # Shape check
    if ms_output.shape != jax_output.shape:
        print(
            f"❌ Shape mismatch: MindSpore {ms_output.shape} vs JAX {jax_output.shape}"
        )
        return False
    print(f"✓ Shapes match: {ms_output.shape}")

    # Compute differences
    abs_diff = np.abs(ms_output - jax_output)
    rel_diff = abs_diff / (np.abs(ms_output) + 1e-8)

    print(f"\nAbsolute difference:")
    print(f"  Max:  {abs_diff.max():.6e}")
    print(f"  Mean: {abs_diff.mean():.6e}")
    print(f"  Std:  {abs_diff.std():.6e}")

    print(f"\nRelative difference:")
    print(f"  Max:  {rel_diff.max():.6e}")
    print(f"  Mean: {rel_diff.mean():.6e}")

    # Check if close enough
    atol = 1e-4
    rtol = 1e-3
    is_close = np.allclose(ms_output, jax_output, atol=atol, rtol=rtol)

    if is_close:
        print(f"\n✓ Outputs match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"\n❌ Outputs differ beyond tolerance (atol={atol}, rtol={rtol})")

        # Show where they differ most
        max_diff_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        print(f"\nLargest difference at index {max_diff_idx}:")
        print(f"  MindSpore: {ms_output[max_diff_idx]:.6f}")
        print(f"  JAX:       {jax_output[max_diff_idx]:.6f}")

    return is_close


def main():
    parser = argparse.ArgumentParser(description="Test JAX/MindSpore equivalence")
    parser.add_argument(
        "--backend",
        choices=["mindspore", "jax"],
        required=True,
        help="Which backend to run",
    )
    parser.add_argument(
        "--save-inputs", action="store_true", help="Save generated inputs to file"
    )
    parser.add_argument(
        "--save-outputs", action="store_true", help="Save model outputs to file"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare with saved MindSpore outputs"
    )
    parser.add_argument(
        "--inputs-path", default="test_inputs.npz", help="Path for inputs file"
    )
    parser.add_argument(
        "--weights-path", default="test_weights.npz", help="Path for weights file"
    )
    parser.add_argument(
        "--outputs-path",
        default="test_outputs_ms.npz",
        help="Path for MindSpore outputs file",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for input generation"
    )

    args = parser.parse_args()

    # Load or create inputs
    if os.path.exists(args.inputs_path) and not args.save_inputs:
        print(f"Loading inputs from {args.inputs_path}...")
        inputs = load_inputs(args.inputs_path)
    else:
        print(f"Creating test inputs with seed {args.seed}...")
        inputs = create_test_inputs(args.seed)
        if args.save_inputs:
            save_inputs(inputs, args.inputs_path)

    print(f"\nInput shapes:")
    for key, val in inputs.items():
        print(f"  {key}: {val.shape} ({val.dtype})")

    # Run the appropriate backend
    if args.backend == "mindspore":
        output = run_mindspore(
            inputs, save_weights=True, weights_path=args.weights_path
        )

        if args.save_outputs:
            np.savez(args.outputs_path, output=output)
            print(f"Saved MindSpore output to {args.outputs_path}")

    elif args.backend == "jax":
        output = run_jax(inputs, weights_path=args.weights_path)

        if args.compare and os.path.exists(args.outputs_path):
            print(f"\nLoading MindSpore outputs from {args.outputs_path}...")
            with np.load(args.outputs_path) as data:
                ms_output = data["output"]
            compare_outputs(ms_output, output)
        elif args.compare:
            print(f"\nWarning: Cannot compare - {args.outputs_path} not found")
            print("Run with --backend mindspore --save-outputs first")


if __name__ == "__main__":
    main()
