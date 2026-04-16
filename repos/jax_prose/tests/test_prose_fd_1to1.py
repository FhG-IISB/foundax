import jax
import jax.numpy as jnp

from jax_prose import prose_fd_1to1


def test_forward_shape():
    model, params = prose_fd_1to1(x_num=32, max_output_dim=4, input_len=6, output_len=3)

    x = jnp.ones((2, 6, 32, 32, 4), dtype=jnp.float32)
    tin = jnp.linspace(0.0, 1.0, 6, dtype=jnp.float32)[None, :, None]
    tout = jnp.linspace(1.1, 1.5, 3, dtype=jnp.float32)[None, :, None]

    y = model.apply(params, x, tin, tout, deterministic=True)
    assert y.shape == (2, 3, 32, 32, 4)
    assert jnp.isfinite(y).all()
