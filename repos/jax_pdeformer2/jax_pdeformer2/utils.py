"""
Utility functions for loading and using PDEformer in JAX.
"""
import re
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

from .pdeformer import PDEformer, create_pdeformer_from_config, PDEFORMER_SMALL_CONFIG


def load_mindspore_checkpoint(ckpt_path: str) -> Dict[str, np.ndarray]:
    """
    Load weights from a MindSpore .ckpt file (protobuf format).
    
    Args:
        ckpt_path: Path to .ckpt file
        
    Returns:
        Dictionary mapping parameter names to numpy arrays
    """
    try:
        import mindspore
        from mindspore import load_checkpoint
        
        # Load checkpoint using MindSpore
        param_dict = load_checkpoint(ckpt_path)
        
        # Convert to numpy
        weights = {}
        for name, param in param_dict.items():
            weights[name] = param.asnumpy()
        
        return weights
        
    except ImportError:
        # Fallback: Try loading as numpy .npz if MindSpore not available
        print("Warning: MindSpore not available, trying to load as .npz...")
        return load_numpy_weights(ckpt_path)


def load_numpy_weights(npz_path: str) -> Dict[str, np.ndarray]:
    """Load weights from a numpy .npz file."""
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def map_ms_to_jax_name(ms_name: str) -> str:
    """
    Map MindSpore parameter name to JAX/Flax naming convention.
    
    MindSpore naming examples:
        pde_encoder.scalar_encoder.net.0.weight
        pde_encoder.function_encoder.net.0.weight  
        pde_encoder.graphormer.layers.0.fc1.weight
        pde_encoder.graphormer.graph_node_feature.node_encoder.embedding_table
        inr.inr.affines.0.weight
        inr.scale_hypernets.0.net.0.weight
        
    JAX naming examples:
        pde_encoder/scalar_encoder/Dense_0/kernel
        pde_encoder/function_encoder/conv1/kernel
        pde_encoder/graphormer/layers_0/fc1/kernel
        pde_encoder/graphormer/graph_node_feature/node_encoder/embedding
        inr/inr/affines_0/kernel
        inr/scale_hypernets_0/Dense_0/kernel
    """
    parts = ms_name.split(".")
    result_parts = []
    
    i = 0
    while i < len(parts):
        part = parts[i]
        
        # Skip 'net' in most cases - it's just a Sequential wrapper
        if part == "net":
            # Check if next part is a digit (layer index)
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                
                # Check context to determine layer type
                if len(result_parts) >= 2:
                    if result_parts[-1] == "function_encoder":
                        # Function encoder uses conv1, conv2, conv3
                        # MindSpore indices: 0, 2, 4 (skipping activations)
                        conv_idx = layer_idx // 2 + 1
                        result_parts.append(f"conv{conv_idx}")
                        i += 2  # Skip 'net' and layer index
                        continue
                    elif result_parts[-1] == "scalar_encoder":
                        # Scalar encoder (MLP) uses Dense_0, Dense_1, etc.
                        # MindSpore indices: 0, 2, 4 (skipping activations)
                        dense_idx = layer_idx // 2
                        result_parts.append(f"Dense_{dense_idx}")
                        i += 2
                        continue
                    elif result_parts[-1].startswith("scale_hypernets_") or \
                         result_parts[-1].startswith("shift_hypernets_") or \
                         result_parts[-1].startswith("affine_hypernets_"):
                        # Hypernets (MLP) - also skip by 2
                        dense_idx = layer_idx // 2
                        result_parts.append(f"Dense_{dense_idx}")
                        i += 2
                        continue
                        
            # Default: keep net
            result_parts.append(part)
            i += 1
            continue
        
        # Handle layer indices after 'layers', 'affines', 'dense_layers', etc.
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            result_parts.append(f"layers_{parts[i + 1]}")
            i += 2
            continue
        
        if part == "affines" and i + 1 < len(parts) and parts[i + 1].isdigit():
            result_parts.append(f"affines_{parts[i + 1]}")
            i += 2
            continue
            
        if part == "dense_layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            result_parts.append(f"dense_layers_{parts[i + 1]}")
            i += 2
            continue
            
        if part == "shift_hypernets" and i + 1 < len(parts) and parts[i + 1].isdigit():
            result_parts.append(f"shift_hypernets_{parts[i + 1]}")
            i += 2
            continue
            
        if part == "scale_hypernets" and i + 1 < len(parts) and parts[i + 1].isdigit():
            result_parts.append(f"scale_hypernets_{parts[i + 1]}")
            i += 2
            continue
            
        if part == "affine_hypernets" and i + 1 < len(parts) and parts[i + 1].isdigit():
            result_parts.append(f"affine_hypernets_{parts[i + 1]}")
            i += 2
            continue
        
        # Handle terminal parts
        if part == "weight":
            result_parts.append("kernel")
        elif part == "bias":
            result_parts.append("bias")
        elif part == "gamma":
            result_parts.append("scale")
        elif part == "beta":
            result_parts.append("bias")
        elif part == "embedding_table":
            result_parts.append("embedding")
        else:
            result_parts.append(part)
        
        i += 1
    
    return "/".join(result_parts)


def convert_weight_shape(weight: np.ndarray, ms_name: str) -> np.ndarray:
    """
    Convert weight shape from MindSpore to JAX convention.
    
    - Dense: [out, in] -> [in, out]
    - Conv2D: [out, in, H, W] -> [H, W, in, out]
    """
    if ms_name.endswith(".weight"):
        if weight.ndim == 2:
            # Dense layer: transpose
            return weight.T
        elif weight.ndim == 4:
            # Conv2D: OIHW -> HWIO
            return np.transpose(weight, (2, 3, 1, 0))
    
    # Bias, embedding, and other tensors: no change
    return weight


def build_nested_params(flat_params: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Build nested parameter dictionary from flat path-based dictionary."""
    result = {}
    
    for path, value in flat_params.items():
        parts = path.split("/")
        current = result
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    return result


def convert_mindspore_to_jax(ms_weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Convert MindSpore weights to JAX/Flax parameter format.
    
    Args:
        ms_weights: Dictionary of MindSpore parameters
        
    Returns:
        Nested dictionary suitable for Flax models
    """
    jax_flat = {}
    
    for ms_name, weight in ms_weights.items():
        # Map name
        jax_path = map_ms_to_jax_name(ms_name)
        
        # Convert weight shape
        converted = convert_weight_shape(weight, ms_name)
        
        jax_flat[jax_path] = converted
    
    # Build nested structure under 'params'
    params = build_nested_params(jax_flat)
    
    return {"params": params}


def load_pdeformer_weights(
    npz_path: str,
    config: Optional[Dict] = None,
    dtype: jnp.dtype = jnp.float32
) -> Tuple[PDEformer, Dict[str, Any]]:
    """
    Load PDEformer model with weights from a converted checkpoint.
    
    Args:
        npz_path: Path to .npz file containing converted weights
        config: Model configuration (default: PDEFORMER_SMALL_CONFIG)
        dtype: Data type for model parameters
        
    Returns:
        Tuple of (model, params) where params can be used with model.apply()
    """
    if config is None:
        config = PDEFORMER_SMALL_CONFIG
    
    # Create model
    model = create_pdeformer_from_config({"model": config}, dtype=dtype)
    
    # Load weights
    ms_weights = load_numpy_weights(npz_path)
    
    # Convert to JAX format
    jax_params = convert_mindspore_to_jax(ms_weights)
    
    # Convert to JAX arrays
    def to_jax_arrays(d):
        if isinstance(d, dict):
            return {k: to_jax_arrays(v) for k, v in d.items()}
        elif isinstance(d, np.ndarray):
            return jnp.array(d, dtype=dtype)
        return d
    
    params = freeze(to_jax_arrays(jax_params))
    
    return model, params


def create_dummy_inputs(
    n_graph: int = 1,
    num_scalar: int = 80,
    num_function: int = 6,
    num_branches: int = 4,  # from function encoder config
    num_points_function: int = 16384,
    num_points: int = 1000,
    dtype: jnp.dtype = jnp.float32,
    key: Optional[jax.random.PRNGKey] = None
) -> Dict[str, jnp.ndarray]:
    """
    Create dummy inputs for testing the PDEformer model.
    
    Note: n_node = num_scalar + num_function * num_branches
    
    Returns a dictionary with all required input tensors.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # n_node must match num_scalar + num_function * num_branches
    n_node = num_scalar + num_function * num_branches
    
    keys = jax.random.split(key, 7)
    
    return {
        "node_type": jax.random.randint(keys[0], (n_graph, n_node, 1), 0, 128, dtype=jnp.int32),
        "node_scalar": jax.random.normal(keys[1], (n_graph, num_scalar, 1), dtype=dtype),
        "node_function": jax.random.normal(keys[2], (n_graph, num_function, num_points_function, 5), dtype=dtype),
        "in_degree": jax.random.randint(keys[3], (n_graph, n_node), 0, 32, dtype=jnp.int32),
        "out_degree": jax.random.randint(keys[4], (n_graph, n_node), 0, 32, dtype=jnp.int32),
        "attn_bias": jax.random.normal(keys[5], (n_graph, n_node, n_node), dtype=dtype),
        "spatial_pos": jax.random.randint(keys[6], (n_graph, n_node, n_node), 0, 16, dtype=jnp.int32),
        "coordinate": jax.random.uniform(key, (n_graph, num_points, 4), dtype=dtype),
    }


def test_model_forward():
    """Test that the model can run a forward pass."""
    print("Creating PDEformer model...")
    model = create_pdeformer_from_config({"model": PDEFORMER_SMALL_CONFIG})
    
    print("Creating dummy inputs...")
    # Use smaller resolution for faster testing
    # num_points_function = resolution^2 = 128^2 = 16384
    inputs = create_dummy_inputs(
        num_scalar=40, 
        num_function=3, 
        num_branches=4,
        num_points_function=128**2,  # Must match function encoder resolution
        num_points=100
    )
    
    print("Initializing model...")
    key = jax.random.PRNGKey(42)
    params = model.init(key, **inputs)
    
    print("Running forward pass...")
    output = model.apply(params, **inputs)
    
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Count parameters
    def count_params(params):
        return sum(x.size for x in jax.tree_util.tree_leaves(params))
    
    num_params = count_params(params)
    print(f"Total parameters: {num_params:,}")
    
    return model, params, output


def save_params_msgpack(params: Dict[str, Any], path: str) -> None:
    """
    Save Flax model parameters to msgpack format.
    
    Args:
        params: Flax parameter dict (can be FrozenDict)
        path: Output path for .msgpack file
    """
    from flax import serialization
    
    # Convert FrozenDict to regular dict if needed
    params_dict = unfreeze(params) if hasattr(params, 'unfreeze') else dict(params)
    
    with open(path, 'wb') as f:
        f.write(serialization.msgpack_serialize(params_dict))


def load_params_msgpack(path: str, dtype: jnp.dtype = jnp.float32) -> Dict[str, Any]:
    """
    Load Flax model parameters from msgpack format.
    
    Args:
        path: Path to .msgpack file
        dtype: Data type for loaded parameters
        
    Returns:
        Frozen parameter dictionary
    """
    from flax import serialization
    
    with open(path, 'rb') as f:
        params = serialization.msgpack_restore(f.read())
    
    # Convert to JAX arrays with specified dtype
    def to_jax_arrays(d):
        if isinstance(d, dict):
            return {k: to_jax_arrays(v) for k, v in d.items()}
        elif isinstance(d, np.ndarray):
            return jnp.array(d, dtype=dtype)
        return d
    
    return freeze(to_jax_arrays(params))


def load_pdeformer_from_msgpack(
    msgpack_path: str,
    config: Optional[Dict] = None,
    dtype: jnp.dtype = jnp.float32
) -> Tuple[PDEformer, Dict[str, Any]]:
    """
    Load PDEformer model with weights from a msgpack checkpoint.
    
    Args:
        msgpack_path: Path to .msgpack file containing saved weights
        config: Model configuration (default: PDEFORMER_SMALL_CONFIG)
        dtype: Data type for model parameters
        
    Returns:
        Tuple of (model, params) where params can be used with model.apply()
    """
    if config is None:
        config = PDEFORMER_SMALL_CONFIG
    
    # Create model
    model = create_pdeformer_from_config({"model": config}, dtype=dtype)
    
    # Load params
    params = load_params_msgpack(msgpack_path, dtype=dtype)
    
    return model, params


if __name__ == "__main__":
    test_model_forward()
