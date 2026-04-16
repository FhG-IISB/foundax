import jax
import jax.numpy as jnp

from .config import PROSE1to1Config, prose_fd_1to1_default_config
from .prose_fd import PROSE1to1
from .prose_fd_2to1 import PROSE2to1
from .prose_ode_pde_2to1 import (
    PROSEODE2to1,
    PROSEPDE2to1,
    ProseTextData2to1Config,
)


def prose_fd_1to1(
    config: PROSE1to1Config | None = None,
    x_num: int = 128,
    max_output_dim: int = 4,
    input_len: int = 10,
    output_len: int = 10,
):
    cfg = config or prose_fd_1to1_default_config()
    model = PROSE1to1(
        config=cfg, x_num=x_num, max_output_dim=max_output_dim, output_len=output_len
    )

    rng = jax.random.PRNGKey(0)
    x = jnp.ones((1, input_len, x_num, x_num, max_output_dim), dtype=jnp.float32)
    tin = jnp.zeros((1, input_len, 1), dtype=jnp.float32)
    tout = jnp.zeros((1, output_len, 1), dtype=jnp.float32)
    params = model.init(
        {"params": rng, "dropout": rng}, x, tin, tout, deterministic=True
    )
    return model, params


def prose_fd_2to1(
    n_words: int,
    x_num: int = 128,
    max_output_dim: int = 4,
    input_len: int = 10,
    output_len: int = 10,
    symbol_len: int = 48,
):
    model = PROSE2to1(
        n_words=n_words,
        x_num=x_num,
        max_output_dim=max_output_dim,
    )

    rng = jax.random.PRNGKey(0)
    x = jnp.ones((1, input_len, x_num, x_num, max_output_dim), dtype=jnp.float32)
    tin = jnp.zeros((1, input_len, 1), dtype=jnp.float32)
    tout = jnp.zeros((1, output_len, 1), dtype=jnp.float32)
    sym = jnp.zeros((1, symbol_len), dtype=jnp.int32)
    sym_mask = jnp.zeros((1, symbol_len), dtype=bool)
    params = model.init({"params": rng}, x, tin, tout, sym, sym_mask)
    return model, params


def prose_ode_2to1(
    n_words: int,
    pad_index: int,
    max_output_dimension: int = 3,
    input_len: int = 50,
    output_len: int = 50,
    text_len: int = 48,
    cfg: ProseTextData2to1Config | None = None,
):
    conf = cfg or ProseTextData2to1Config()
    model = PROSEODE2to1(
        n_words=n_words,
        pad_index=pad_index,
        max_output_dimension=max_output_dimension,
        cfg=conf,
    )

    rng = jax.random.PRNGKey(0)
    x = jnp.ones(
        (input_len, 1, 1 + max_output_dimension),
        dtype=jnp.float32,
    )
    data_lengths = jnp.asarray([input_len], dtype=jnp.int32)
    query_times = jnp.linspace(0.0, 1.0, output_len, dtype=jnp.float32)
    text = jnp.zeros((text_len, 1), dtype=jnp.int32)
    text_lengths = jnp.asarray([text_len], dtype=jnp.int32)
    params = model.init({"params": rng}, x, data_lengths, query_times, text, text_lengths)
    return model, params


def prose_pde_2to1(
    n_words: int,
    pad_index: int,
    max_output_dimension: int = 1,
    x_patch_size: int = 1,
    x_grid_size: int = 128,
    input_len: int = 10,
    output_len: int = 10,
    text_len: int = 48,
    cfg: ProseTextData2to1Config | None = None,
):
    conf = cfg or ProseTextData2to1Config()
    conf = ProseTextData2to1Config(
        **{**conf.__dict__, "x_patch_size": x_patch_size, "x_grid_size": x_grid_size}
    )
    model = PROSEPDE2to1(
        n_words=n_words,
        pad_index=pad_index,
        max_output_dimension=max_output_dimension,
        cfg=conf,
    )

    in_dim = 1 + max_output_dimension * x_patch_size
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((input_len, 1, in_dim), dtype=jnp.float32)
    data_lengths = jnp.asarray([input_len], dtype=jnp.int32)
    query_times = jnp.linspace(0.0, 1.0, output_len, dtype=jnp.float32)
    text = jnp.zeros((text_len, 1), dtype=jnp.int32)
    text_lengths = jnp.asarray([text_len], dtype=jnp.int32)
    params = model.init({"params": rng}, x, data_lengths, query_times, text, text_lengths)
    return model, params
