import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

import foundax as fx
from foundax.architectures.transolver import (
    PhysicsAttentionIrregular,
    PhysicsAttentionStructured2D,
    PhysicsAttentionStructured3D,
    TransolverBlock,
)


KEY = jax.random.PRNGKey(0)


def test_factory_irregular_shape():
    model = fx.transolver(
        space_dim=2,
        fun_dim=1,
        out_features=3,
        hidden_dim=64,
        n_layers=2,
        n_heads=4,
        n_slices=16,
    )
    x_coords = jax.random.normal(KEY, (200, 2))
    x_func = jax.random.normal(KEY, (200, 1))
    out = model(x_coords, x_func)
    assert out.shape == (200, 3)
    assert jnp.all(jnp.isfinite(out))


def test_factory_structured_2d_shape():
    model = fx.transolver2d(
        space_dim=2,
        fun_dim=2,
        out_features=1,
        hidden_dim=64,
        n_layers=2,
        n_heads=4,
        n_slices=16,
    )
    x_coords = jax.random.normal(KEY, (32, 32, 2))
    x_func = jax.random.normal(KEY, (32, 32, 2))
    out = model(x_coords, x_func)
    assert out.shape == (32, 32, 1)
    assert jnp.all(jnp.isfinite(out))


def test_factory_structured_3d_shape():
    model = fx.transolver3d(
        space_dim=3,
        fun_dim=1,
        out_features=2,
        hidden_dim=32,
        n_layers=2,
        n_heads=4,
        n_slices=8,
    )
    x_coords = jax.random.normal(KEY, (8, 8, 8, 3))
    x_func = jax.random.normal(KEY, (8, 8, 8, 1))
    out = model(x_coords, x_func)
    assert out.shape == (8, 8, 8, 2)
    assert jnp.all(jnp.isfinite(out))


def test_physics_attention_preserves_shape():
    """All three attention variants are residual-shaped: out matches input."""
    dim = 32
    attn = PhysicsAttentionIrregular(dim, num_heads=4, dim_head=8, slice_num=8, key=KEY)
    x = jax.random.normal(KEY, (100, dim))
    assert attn(x).shape == x.shape

    attn2 = PhysicsAttentionStructured2D(
        dim, num_heads=4, dim_head=8, slice_num=8, key=KEY
    )
    x = jax.random.normal(KEY, (8, 16, dim))
    assert attn2(x).shape == x.shape

    attn3 = PhysicsAttentionStructured3D(
        dim, num_heads=4, dim_head=8, slice_num=4, key=KEY
    )
    x = jax.random.normal(KEY, (4, 8, 8, dim))
    assert attn3(x).shape == x.shape


def test_gradient_flows_through_physics_attention():
    """One forward-backward step with vmap'd batching must produce finite grads."""
    model = fx.transolver2d(
        space_dim=2,
        fun_dim=1,
        out_features=1,
        hidden_dim=32,
        n_layers=2,
        n_heads=4,
        n_slices=8,
    )
    xc = jax.random.normal(KEY, (4, 16, 16, 2))
    xf = jax.random.normal(KEY, (4, 16, 16, 1))
    yt = jax.random.normal(KEY, (4, 16, 16, 1))

    def loss(m, xc, xf, yt):
        pred = jax.vmap(m)(xc, xf)
        return jnp.mean((pred - yt) ** 2)

    grads = eqx.filter_grad(loss)(model, xc, xf, yt)
    # Slice projection, temperature, and FFN should all receive gradient.
    assert jnp.all(jnp.isfinite(grads.blocks[0].physics_attn.to_q.weight))
    assert jnp.all(jnp.isfinite(grads.blocks[0].physics_attn.temperature))
    assert jnp.all(jnp.isfinite(grads.blocks[0].ffn.pre.weight))


def test_hidden_dim_not_divisible_by_heads_raises():
    with pytest.raises(ValueError, match="divisible"):
        fx.transolver(space_dim=2, fun_dim=1, hidden_dim=63, n_heads=8)


def test_time_conditioning_changes_output():
    """time_input=True branch must use ``t`` and respond to its value."""
    model = fx.transolver(
        space_dim=2,
        fun_dim=1,
        out_features=1,
        hidden_dim=32,
        n_layers=2,
        n_heads=2,
        n_slices=8,
        time_input=True,
    )
    xc = jax.random.normal(KEY, (50, 2))
    xf = jax.random.normal(KEY, (50, 1))
    y0 = model(xc, xf, t=jnp.array(0.0))
    y1 = model(xc, xf, t=jnp.array(1.0))
    assert not jnp.allclose(y0, y1, atol=1e-6)


def test_time_passed_without_time_input_raises():
    model = fx.transolver(space_dim=2, fun_dim=1, hidden_dim=32, n_heads=2, n_layers=1)
    xc = jax.random.normal(KEY, (50, 2))
    xf = jax.random.normal(KEY, (50, 1))
    with pytest.raises(ValueError, match="time_input"):
        model(xc, xf, t=jnp.array(0.5))


def test_dropout_runs_with_key():
    model = fx.transolver(
        space_dim=2,
        fun_dim=1,
        hidden_dim=32,
        n_heads=2,
        n_layers=2,
        n_slices=8,
        dropout=0.1,
    )
    xc = jax.random.normal(KEY, (50, 2))
    xf = jax.random.normal(KEY, (50, 1))
    out = model(xc, xf, key=jax.random.PRNGKey(99))
    assert out.shape == (50, 1)
    assert jnp.all(jnp.isfinite(out))


def test_x_func_none_with_positive_fun_dim_raises():
    """Passing x_func=None with fun_dim>0 must raise a clear error, not
    a cryptic matmul shape mismatch deep in the lift."""
    model = fx.transolver(space_dim=2, fun_dim=1, hidden_dim=32, n_heads=2, n_layers=1)
    xc = jax.random.normal(KEY, (50, 2))
    with pytest.raises(ValueError, match="fun_dim"):
        model(xc, x_func=None)


def test_pipe_integration_with_transolver_block():
    """TransolverBlock alone is shape-preserving and pipeable as a block."""
    dim = 32
    attn = PhysicsAttentionIrregular(dim, num_heads=4, dim_head=8, slice_num=8, key=KEY)
    block = TransolverBlock(dim, attn, mlp_ratio=2, key=KEY)
    x = jax.random.normal(KEY, (50, dim))
    assert block(x).shape == x.shape
