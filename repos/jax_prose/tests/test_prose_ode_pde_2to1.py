import jax
import jax.numpy as jnp

from jax_prose.prose_ode_pde_2to1 import (
    PROSEODE2to1,
    PROSEPDE2to1,
    ProseTextData2to1Config,
)


def test_prose_ode_2to1_forward_shape():
    cfg = ProseTextData2to1Config()
    model = PROSEODE2to1(n_words=100, pad_index=0, max_output_dimension=3, cfg=cfg)

    in_len = 8
    out_len = 6
    txt_len = 10
    x = jnp.ones((in_len, 2, 4), dtype=jnp.float32)
    data_lengths = jnp.asarray([8, 7], dtype=jnp.int32)
    query_times = jnp.linspace(0.0, 1.0, out_len, dtype=jnp.float32)
    text_input = jnp.zeros((txt_len, 2), dtype=jnp.int32)
    text_lengths = jnp.asarray([10, 9], dtype=jnp.int32)

    params = model.init(
        {"params": jax.random.PRNGKey(0)},
        x,
        data_lengths,
        query_times,
        text_input,
        text_lengths,
    )
    y = model.apply(params, x, data_lengths, query_times, text_input, text_lengths)
    assert y.shape == (out_len, 2, 3)


def test_prose_pde_2to1_forward_shape():
    cfg = ProseTextData2to1Config(x_patch_size=2, x_grid_size=16)
    model = PROSEPDE2to1(n_words=120, pad_index=0, max_output_dimension=2, cfg=cfg)

    in_len = 5
    out_len = 4
    txt_len = 7
    in_dim = 1 + 2 * 2

    x = jnp.ones((in_len, 1, in_dim), dtype=jnp.float32)
    data_lengths = jnp.asarray([5], dtype=jnp.int32)
    query_times = jnp.linspace(0.0, 1.0, out_len, dtype=jnp.float32)
    text_input = jnp.zeros((txt_len, 1), dtype=jnp.int32)
    text_lengths = jnp.asarray([7], dtype=jnp.int32)

    params = model.init(
        {"params": jax.random.PRNGKey(1)},
        x,
        data_lengths,
        query_times,
        text_input,
        text_lengths,
    )
    y = model.apply(params, x, data_lengths, query_times, text_input, text_lengths)
    assert y.shape == (out_len, 1, 32)
