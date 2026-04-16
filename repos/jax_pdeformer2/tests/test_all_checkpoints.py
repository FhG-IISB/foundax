"""Test loading and running all available checkpoints."""

import os
import jax
import jax.numpy as jnp
import numpy as np
import sys

from jax_pdeformer2 import create_pdeformer_from_config, PDEFORMER_SMALL_CONFIG
from jax_pdeformer2.utils import (
    load_mindspore_checkpoint,
    convert_mindspore_to_jax,
    create_dummy_inputs,
)


# Model configurations for different sizes
PDEFORMER_BASE_CONFIG = {
    "graphormer": {
        "num_encoder_layers": 12,
        "embed_dim": 768,
        "ffn_embed_dim": 1536,
        "num_heads": 32,
    },
    "inr": {
        "dim_hidden": 768,  # INR hidden dimension
        "num_layers": 12,
        "poly_inr": {
            "enable_shift": True,
            "enable_scale": True,
        },
    },
    "hypernet": {
        "dim_hidden": 512,  # Hypernet hidden dimension
    },
}

PDEFORMER_FAST_CONFIG = {
    "graphormer": {
        "num_encoder_layers": 12,
        "embed_dim": 768,
        "ffn_embed_dim": 1536,
        "num_heads": 32,
    },
    "inr": {
        "dim_hidden": 256,
        "num_layers": 12,
        "poly_inr": {
            "enable_shift": False,  # No hypernets in fast model
            "enable_scale": False,  # No hypernets in fast model
        },
    },
    "hypernet": {
        "dim_hidden": 128,
    },
}


def count_parameters(params):
    """Count total number of parameters."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


import pytest


def run_checkpoint(ckpt_path, config, model_name):
    """Test loading and running a checkpoint."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"{'='*60}")

    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return False

    try:
        # Create model
        model = create_pdeformer_from_config({"model": config})
        print(f"✓ Model created successfully")

        # Load weights
        print(f"Loading weights from {os.path.basename(ckpt_path)}...")
        ms_weights = load_mindspore_checkpoint(ckpt_path)
        jax_params_dict = convert_mindspore_to_jax(ms_weights)

        # Convert numpy arrays to JAX arrays
        def to_jax(d):
            if isinstance(d, dict):
                return {k: to_jax(v) for k, v in d.items()}
            elif isinstance(d, np.ndarray):
                return jnp.array(d)
            return d

        params = to_jax(jax_params_dict)
        print(f"✓ Weights loaded successfully")

        # Count parameters
        n_params = count_parameters(params)
        print(f"✓ Model has {n_params:,} parameters ({n_params/1e6:.1f}M)")

        # Convert to frozen dict for model
        from flax.core import freeze

        params_frozen = freeze(params)

        # Create dummy inputs
        print(f"Creating test inputs...")
        num_scalar = 40
        num_function = 3
        num_branches = 4
        num_points = 100
        resolution = 128

        # Create inputs directly with numpy first to avoid JAX/numpy version conflict
        np.random.seed(42)
        n_graph = 2
        n_node = num_scalar + num_function * num_branches

        inputs = {
            "node_type": jnp.array(np.random.randint(0, 128, (n_graph, n_node, 1), dtype=np.int32)),
            "node_scalar": jnp.array(np.random.randn(n_graph, num_scalar, 1).astype(np.float32)),
            "node_function": jnp.array(np.random.randn(n_graph, num_function, resolution**2, 5).astype(np.float32)),
            "in_degree": jnp.array(np.random.randint(0, 32, (n_graph, n_node), dtype=np.int32)),
            "out_degree": jnp.array(np.random.randint(0, 32, (n_graph, n_node), dtype=np.int32)),
            "attn_bias": jnp.array(np.random.randn(n_graph, n_node, n_node).astype(np.float32)),
            "spatial_pos": jnp.array(np.random.randint(0, 16, (n_graph, n_node, n_node), dtype=np.int32)),
            "coordinate": jnp.array(np.random.rand(n_graph, num_points, 4).astype(np.float32)),
        }

        # Run forward pass
        print(f"Running forward pass...")
        output = model.apply(
            params_frozen,
            inputs["node_type"],
            inputs["node_scalar"],
            inputs["node_function"],
            inputs["in_degree"],
            inputs["out_degree"],
            inputs["attn_bias"],
            inputs["spatial_pos"],
            inputs["coordinate"],
        )

        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"  Output mean: {output.mean():.4f}")
        print(f"  Output std: {output.std():.4f}")

        # Check for NaN/Inf
        if jnp.isnan(output).any():
            print(f"⚠️  Warning: Output contains NaN values")
            return False
        if jnp.isinf(output).any():
            print(f"⚠️  Warning: Output contains Inf values")
            return False

        print(f"\n✅ {model_name} checkpoint test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ {model_name} checkpoint test FAILED")
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Test all available checkpoints."""
    base_dir = os.path.join(os.path.dirname(__file__), "..")

    checkpoints = [
        (
            os.path.join(base_dir, "pdeformer2-base.ckpt"),
            PDEFORMER_BASE_CONFIG,
            "PDEformer-2-Base",
        ),
        (
            os.path.join(base_dir, "pdeformer2-fast.ckpt"),
            PDEFORMER_FAST_CONFIG,
            "PDEformer-2-Fast",
        ),
    ]

    results = {}
    for ckpt_path, config, name in checkpoints:
        results[name] = test_checkpoint(ckpt_path, config, name)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print(f"\n🎉 All checkpoint tests passed!")
    else:
        print(f"\n⚠️  Some checkpoint tests failed")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
