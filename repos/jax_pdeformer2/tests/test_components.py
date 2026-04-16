"""
Unit tests to verify JAX implementation matches MindSpore architecture.

This script tests individual components to ensure they produce
outputs of the correct shape and follow the same logic.
"""

import numpy as np
import jax
import jax.numpy as jnp

# Test configuration (smaller for faster testing)
TEST_CONFIG = {
    "n_graph": 2,
    "n_node": 20,
    "num_scalar": 10,
    "num_function": 2,
    "num_branches": 4,
    "num_points_function": 64,  # 8x8 for testing
    "num_points": 50,
    "embed_dim": 64,
    "num_heads": 4,
    "resolution": 8,
}


def test_mlp():
    """Test MLP produces correct output shape."""
    from jax_pdeformer2.basic_block import MLP

    batch_size = TEST_CONFIG["n_graph"]
    seq_len = TEST_CONFIG["num_scalar"]
    dim_in = 1
    dim_out = TEST_CONFIG["embed_dim"]
    dim_hidden = 32
    num_layers = 3

    mlp = MLP(dim_out=dim_out, dim_hidden=dim_hidden, num_layers=num_layers)

    x = jnp.ones((batch_size, seq_len, dim_in))
    key = jax.random.PRNGKey(0)
    params = mlp.init(key, x)
    output = mlp.apply(params, x)

    expected_shape = (batch_size, seq_len, dim_out)
    assert (
        output.shape == expected_shape
    ), f"MLP: Expected {expected_shape}, got {output.shape}"
    print(f"✓ MLP: {x.shape} -> {output.shape}")
    return True


def test_graph_node_feature():
    """Test GraphNodeFeature produces correct output shape."""
    from jax_pdeformer2.graphormer import GraphNodeFeature

    n_graph = TEST_CONFIG["n_graph"]
    n_node = TEST_CONFIG["n_node"]
    embed_dim = TEST_CONFIG["embed_dim"]
    num_heads = TEST_CONFIG["num_heads"]

    layer = GraphNodeFeature(
        num_heads=num_heads,
        num_node_type=128,
        num_in_degree=32,
        num_out_degree=32,
        embed_dim=embed_dim,
    )

    node_type = jnp.ones((n_graph, n_node, 1), dtype=jnp.int32)
    in_degree = jnp.ones((n_graph, n_node), dtype=jnp.int32)
    out_degree = jnp.ones((n_graph, n_node), dtype=jnp.int32)

    key = jax.random.PRNGKey(0)
    params = layer.init(key, node_type, in_degree, out_degree)
    output = layer.apply(params, node_type, in_degree, out_degree)

    expected_shape = (n_graph, n_node, embed_dim)
    assert (
        output.shape == expected_shape
    ), f"GraphNodeFeature: Expected {expected_shape}, got {output.shape}"
    print(f"✓ GraphNodeFeature: node_type{node_type.shape} -> {output.shape}")
    return True


def test_graph_attn_bias():
    """Test GraphAttnBias produces correct output shape."""
    from jax_pdeformer2.graphormer import GraphAttnBias

    n_graph = TEST_CONFIG["n_graph"]
    n_node = TEST_CONFIG["n_node"]
    num_heads = TEST_CONFIG["num_heads"]

    layer = GraphAttnBias(
        num_heads=num_heads,
        num_spatial=16,
    )

    attn_bias = jnp.zeros((n_graph, n_node, n_node))
    spatial_pos = jnp.ones((n_graph, n_node, n_node), dtype=jnp.int32)

    key = jax.random.PRNGKey(0)
    params = layer.init(key, attn_bias, spatial_pos)
    output = layer.apply(params, attn_bias, spatial_pos)

    expected_shape = (n_graph, num_heads, n_node, n_node)
    assert (
        output.shape == expected_shape
    ), f"GraphAttnBias: Expected {expected_shape}, got {output.shape}"
    print(f"✓ GraphAttnBias: attn_bias{attn_bias.shape} -> {output.shape}")
    return True


def test_multihead_attention():
    """Test MultiheadAttention produces correct output shape."""
    from jax_pdeformer2.graphormer import MultiheadAttention

    n_graph = TEST_CONFIG["n_graph"]
    n_node = TEST_CONFIG["n_node"]
    embed_dim = TEST_CONFIG["embed_dim"]
    num_heads = TEST_CONFIG["num_heads"]

    mha = MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
    )

    # Input shape: [n_node, n_graph, embed_dim]
    x = jnp.ones((n_node, n_graph, embed_dim))
    attn_bias = jnp.zeros((n_graph, num_heads, n_node, n_node))

    key = jax.random.PRNGKey(0)
    params = mha.init(key, x, attn_bias=attn_bias)
    output = mha.apply(params, x, attn_bias=attn_bias)

    expected_shape = (n_node, n_graph, embed_dim)
    assert (
        output.shape == expected_shape
    ), f"MultiheadAttention: Expected {expected_shape}, got {output.shape}"
    print(f"✓ MultiheadAttention: {x.shape} -> {output.shape}")
    return True


def test_graphormer_encoder_layer():
    """Test GraphormerEncoderLayer produces correct output shape."""
    from jax_pdeformer2.graphormer import GraphormerEncoderLayer

    n_graph = TEST_CONFIG["n_graph"]
    n_node = TEST_CONFIG["n_node"]
    embed_dim = TEST_CONFIG["embed_dim"]
    num_heads = TEST_CONFIG["num_heads"]

    layer = GraphormerEncoderLayer(
        embed_dim=embed_dim,
        ffn_embed_dim=embed_dim * 2,
        num_heads=num_heads,
        pre_layernorm=True,
    )

    # Input shape: [n_node, n_graph, embed_dim]
    x = jnp.ones((n_node, n_graph, embed_dim))
    attn_bias = jnp.zeros((n_graph, num_heads, n_node, n_node))

    key = jax.random.PRNGKey(0)
    params = layer.init(key, x, attn_bias=attn_bias)
    output = layer.apply(params, x, attn_bias=attn_bias)

    expected_shape = (n_node, n_graph, embed_dim)
    assert (
        output.shape == expected_shape
    ), f"GraphormerEncoderLayer: Expected {expected_shape}, got {output.shape}"
    print(f"✓ GraphormerEncoderLayer: {x.shape} -> {output.shape}")
    return True


def test_conv2d_func_encoder():
    """Test Conv2dFuncEncoderV3 produces correct output shape."""
    from jax_pdeformer2.function_encoder import Conv2dFuncEncoderV3

    n_graph = TEST_CONFIG["n_graph"]
    resolution = 64  # Must be divisible by 64 for V3 encoder
    num_points = resolution**2
    dim_in = 5
    out_dim = TEST_CONFIG["embed_dim"]

    encoder = Conv2dFuncEncoderV3(
        in_dim=dim_in,
        out_dim=out_dim,
        resolution=resolution,
        input_txyz=False,
        keep_nchw=True,
    )

    x = jnp.ones((n_graph, num_points, dim_in))

    key = jax.random.PRNGKey(0)
    params = encoder.init(key, x)
    output = encoder.apply(params, x)

    # Output shape should be [n_graph, out_dim, 1, 1] for 64x64 input with 3 conv layers of stride 4
    # 64 / 4 / 4 / 4 = 1
    expected_shape = (n_graph, out_dim, 1, 1)
    assert (
        output.shape == expected_shape
    ), f"Conv2dFuncEncoderV3: Expected {expected_shape}, got {output.shape}"
    print(f"✓ Conv2dFuncEncoderV3: {x.shape} -> {output.shape}")
    return True


def test_poly_inr():
    """Test PolyINR produces correct output shape."""
    from jax_pdeformer2.inr_with_hypernet import PolyINR

    batch_size = TEST_CONFIG["n_graph"]
    num_points = TEST_CONFIG["num_points"]
    dim_in = 4
    dim_out = 1
    dim_hidden = 32
    num_layers = 4

    inr = PolyINR(
        dim_in=dim_in,
        dim_out=dim_out,
        dim_hidden=dim_hidden,
        num_layers=num_layers,
        activation_fn="sin",
    )

    x = jnp.ones((batch_size, num_points, dim_in))

    key = jax.random.PRNGKey(0)
    params = inr.init(key, x)
    output = inr.apply(params, x)

    expected_shape = (batch_size, num_points, dim_out)
    assert (
        output.shape == expected_shape
    ), f"PolyINR: Expected {expected_shape}, got {output.shape}"
    print(f"✓ PolyINR: {x.shape} -> {output.shape}")
    return True


def test_poly_inr_with_modulation():
    """Test PolyINR with modulation inputs."""
    from jax_pdeformer2.inr_with_hypernet import PolyINR

    batch_size = TEST_CONFIG["n_graph"]
    num_points = TEST_CONFIG["num_points"]
    dim_in = 4
    dim_out = 1
    dim_hidden = 32
    num_layers = 4

    inr = PolyINR(
        dim_in=dim_in,
        dim_out=dim_out,
        dim_hidden=dim_hidden,
        num_layers=num_layers,
        activation_fn="sin",
    )

    x = jnp.ones((batch_size, num_points, dim_in))
    # Modulations shape: [num_layers-1, batch_size, dim_hidden]
    scale_mod = jnp.zeros((num_layers - 1, batch_size, dim_hidden))
    shift_mod = jnp.zeros((num_layers - 1, batch_size, dim_hidden))

    key = jax.random.PRNGKey(0)
    params = inr.init(key, x, scale_modulations=scale_mod, shift_modulations=shift_mod)
    output = inr.apply(
        params, x, scale_modulations=scale_mod, shift_modulations=shift_mod
    )

    expected_shape = (batch_size, num_points, dim_out)
    assert (
        output.shape == expected_shape
    ), f"PolyINR with modulation: Expected {expected_shape}, got {output.shape}"
    print(f"✓ PolyINR with modulation: {x.shape} -> {output.shape}")
    return True


def test_pde_encoder():
    """Test PDEEncoder produces correct output shape."""
    from jax_pdeformer2.pdeformer import PDEEncoder

    n_graph = TEST_CONFIG["n_graph"]
    num_scalar = TEST_CONFIG["num_scalar"]
    num_function = TEST_CONFIG["num_function"]
    num_branches = TEST_CONFIG["num_branches"]  # 4 branches
    resolution = 128  # Must be 128 for 4 branches (128/64 = 2, 2*2 = 4)
    n_node = num_scalar + num_function * num_branches
    num_points_function = resolution**2
    embed_dim = TEST_CONFIG["embed_dim"]

    encoder = PDEEncoder(
        num_node_type=128,
        num_in_degree=32,
        num_out_degree=32,
        num_spatial=16,
        num_encoder_layers=2,  # Fewer layers for faster testing
        embed_dim=embed_dim,
        ffn_embed_dim=embed_dim * 2,
        num_heads=TEST_CONFIG["num_heads"],
        pre_layernorm=True,
        scalar_dim_hidden=32,
        scalar_num_layers=2,
        func_enc_resolution=resolution,
    )

    inputs = {
        "node_type": jnp.ones((n_graph, n_node, 1), dtype=jnp.int32),
        "node_scalar": jnp.ones((n_graph, num_scalar, 1)),
        "node_function": jnp.ones((n_graph, num_function, num_points_function, 5)),
        "in_degree": jnp.ones((n_graph, n_node), dtype=jnp.int32),
        "out_degree": jnp.ones((n_graph, n_node), dtype=jnp.int32),
        "attn_bias": jnp.zeros((n_graph, n_node, n_node)),
        "spatial_pos": jnp.ones((n_graph, n_node, n_node), dtype=jnp.int32),
    }

    key = jax.random.PRNGKey(0)
    params = encoder.init(key, **inputs)
    output = encoder.apply(params, **inputs)

    # Output shape: [n_node, n_graph, embed_dim]
    expected_shape = (n_node, n_graph, embed_dim)
    assert (
        output.shape == expected_shape
    ), f"PDEEncoder: Expected {expected_shape}, got {output.shape}"
    print(f"✓ PDEEncoder: node_type{inputs['node_type'].shape} -> {output.shape}")
    return True


def test_full_pdeformer():
    """Test full PDEformer produces correct output shape."""
    from jax_pdeformer2.pdeformer import PDEformer

    n_graph = TEST_CONFIG["n_graph"]
    num_scalar = TEST_CONFIG["num_scalar"]
    num_function = TEST_CONFIG["num_function"]
    num_branches = TEST_CONFIG["num_branches"]  # 4 branches
    resolution = 128  # Must be 128 for 4 branches (128/64 = 2, 2*2 = 4)
    n_node = num_scalar + num_function * num_branches
    num_points_function = resolution**2
    num_points = TEST_CONFIG["num_points"]
    embed_dim = TEST_CONFIG["embed_dim"]

    model = PDEformer(
        num_node_type=128,
        num_in_degree=32,
        num_out_degree=32,
        num_spatial=16,
        num_encoder_layers=2,
        embed_dim=embed_dim,
        ffn_embed_dim=embed_dim * 2,
        num_heads=TEST_CONFIG["num_heads"],
        pre_layernorm=True,
        scalar_dim_hidden=32,
        scalar_num_layers=2,
        func_enc_resolution=resolution,
        inr_dim_hidden=32,
        inr_num_layers=3,
        enable_shift=True,
        enable_scale=True,
        hyper_dim_hidden=32,
        hyper_num_layers=2,
    )

    inputs = {
        "node_type": jnp.ones((n_graph, n_node, 1), dtype=jnp.int32),
        "node_scalar": jnp.ones((n_graph, num_scalar, 1)),
        "node_function": jnp.ones((n_graph, num_function, num_points_function, 5)),
        "in_degree": jnp.ones((n_graph, n_node), dtype=jnp.int32),
        "out_degree": jnp.ones((n_graph, n_node), dtype=jnp.int32),
        "attn_bias": jnp.zeros((n_graph, n_node, n_node)),
        "spatial_pos": jnp.ones((n_graph, n_node, n_node), dtype=jnp.int32),
        "coordinate": jnp.ones((n_graph, num_points, 4)),
    }

    key = jax.random.PRNGKey(0)
    params = model.init(key, **inputs)
    output = model.apply(params, **inputs)

    # Output shape: [n_graph, num_points, 1]
    expected_shape = (n_graph, num_points, 1)
    assert (
        output.shape == expected_shape
    ), f"PDEformer: Expected {expected_shape}, got {output.shape}"
    print(f"✓ PDEformer: coordinate{inputs['coordinate'].shape} -> {output.shape}")

    # Count parameters
    def count_params(params):
        return sum(x.size for x in jax.tree_util.tree_leaves(params))

    num_params = count_params(params)
    print(f"  Total parameters: {num_params:,}")

    return True


def run_all_tests():
    """Run all component tests."""
    print("=" * 60)
    print("JAX PDEformer Component Tests")
    print("=" * 60)
    print()

    tests = [
        ("MLP", test_mlp),
        ("GraphNodeFeature", test_graph_node_feature),
        ("GraphAttnBias", test_graph_attn_bias),
        ("MultiheadAttention", test_multihead_attention),
        ("GraphormerEncoderLayer", test_graphormer_encoder_layer),
        ("Conv2dFuncEncoderV3", test_conv2d_func_encoder),
        ("PolyINR", test_poly_inr),
        ("PolyINR with modulation", test_poly_inr_with_modulation),
        ("PDEEncoder", test_pde_encoder),
        ("Full PDEformer", test_full_pdeformer),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
