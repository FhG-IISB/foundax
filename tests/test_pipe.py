"""Tests for the pipe (|) composition API, combinators, and SpectralBlock layers."""

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")

import foundax as fx  # noqa: E402
from foundax.pipe import Block, Pipe, ShapeMismatchError  # noqa: E402


def _ks(n):
    return jax.random.split(jax.random.PRNGKey(42), n)


# ── SpectralBlock layers ────────────────────────────────────────────────────


def test_spectral_block_1d_shape():
    blk = fx.layers.SpectralBlock1d(4, 8, n_modes=4, key=_ks(1)[0])
    y = blk(jnp.ones((32, 4)))
    assert y.shape == (32, 8)


def test_spectral_block_2d_shape():
    blk = fx.layers.SpectralBlock2d(4, 8, n_modes=4, key=_ks(1)[0])
    y = blk(jnp.ones((16, 16, 4)))
    assert y.shape == (16, 16, 8)


def test_spectral_block_3d_shape():
    blk = fx.layers.SpectralBlock3d(4, 8, n_modes=4, key=_ks(1)[0])
    y = blk(jnp.ones((8, 8, 8, 4)))
    assert y.shape == (8, 8, 8, 8)


def test_spectral_block_same_channels():
    blk = fx.layers.SpectralBlock2d(16, 16, n_modes=8, key=_ks(1)[0])
    x = jnp.ones((12, 12, 16))
    assert blk(x).shape == x.shape


def test_spectral_block_channel_fields():
    blk = fx.layers.SpectralBlock2d(3, 32, n_modes=8, key=_ks(1)[0])
    assert blk.in_channels == 3
    assert blk.out_channels == 32


# ── block() wrapping ────────────────────────────────────────────────────────


def test_block_wraps_spectral_block():
    b = fx.block(fx.layers.SpectralBlock2d(3, 32, n_modes=8, key=_ks(1)[0]))
    assert isinstance(b, Block)
    assert b._in_channels == 3
    assert b._out_channels == 32


def test_block_custom_name():
    b = fx.block(
        fx.layers.SpectralBlock2d(3, 32, n_modes=8, key=_ks(1)[0]), name="encoder"
    )
    assert b.name == "encoder"


def test_block_is_callable():
    b = fx.block(fx.layers.SpectralBlock2d(3, 32, n_modes=8, key=_ks(1)[0]))
    y = b(jnp.ones((16, 16, 3)))
    assert y.shape == (16, 16, 32)


def test_block_wraps_mlp():
    m = fx.mlp(in_features=4, output_dim=8, hidden_dims=16, key=_ks(1)[0])
    b = fx.block(m)
    assert b._in_channels == 4  # sniffed from in_features
    assert b._out_channels == 8  # sniffed from output_dim


def test_block_unknown_channels_is_none():
    raw = eqx.nn.MLP(in_size=8, out_size=4, width_size=16, depth=2, key=_ks(1)[0])
    b = fx.block(raw)
    assert b._in_channels is None
    assert b._out_channels is None


def test_block_standalone_model():
    # Existing models still work on their own without wrapping
    fno = fx.fno2d(
        in_features=3, hidden_channels=16, n_modes=4, d_vars=1, key=_ks(1)[0]
    )
    x = jnp.ones((8, 8, 3))
    y = fno(x)
    assert y.shape[2] == 1


# ── Pipe creation and forward pass ─────────────────────────────────────────


def test_pipe_two_blocks():
    ks = _ks(2)
    b1 = fx.block(fx.layers.SpectralBlock2d(3, 16, n_modes=4, key=ks[0]))
    b2 = fx.block(fx.layers.SpectralBlock2d(16, 1, n_modes=4, key=ks[1]))
    pipe = b1 | b2
    assert isinstance(pipe, Pipe)
    assert len(pipe.blocks) == 2


def test_pipe_three_blocks_forward():
    ks = _ks(3)
    b1 = fx.block(fx.layers.SpectralBlock2d(3, 32, n_modes=4, key=ks[0]))
    b2 = fx.block(fx.layers.SpectralBlock2d(32, 32, n_modes=4, key=ks[1]))
    b3 = fx.block(fx.layers.SpectralBlock2d(32, 1, n_modes=4, key=ks[2]))
    pipe = b1 | b2 | b3
    assert len(pipe.blocks) == 3
    y = pipe(jnp.ones((16, 16, 3)))
    assert y.shape == (16, 16, 1)


def test_pipe_flattens_block_or_pipe():
    ks = _ks(3)
    b1 = fx.block(fx.layers.SpectralBlock2d(3, 16, n_modes=4, key=ks[0]))
    b2 = fx.block(fx.layers.SpectralBlock2d(16, 16, n_modes=4, key=ks[1]))
    b3 = fx.block(fx.layers.SpectralBlock2d(16, 1, n_modes=4, key=ks[2]))
    tail = b2 | b3  # Pipe([b2, b3])
    pipe = b1 | tail  # Block | Pipe  →  flat Pipe([b1, b2, b3])
    assert len(pipe.blocks) == 3


def test_pipe_flattens_pipe_or_pipe():
    ks = _ks(4)
    b = [fx.block(fx.layers.SpectralBlock2d(8, 8, n_modes=4, key=k)) for k in ks]
    left = b[0] | b[1]
    right = b[2] | b[3]
    merged = left | right
    assert isinstance(merged, Pipe)
    assert len(merged.blocks) == 4


def test_pipe_or_raises_on_non_block():
    b = fx.block(fx.layers.SpectralBlock2d(3, 32, n_modes=4, key=_ks(1)[0]))
    with pytest.raises(TypeError, match="foundax.block()"):
        _ = b | "not_a_block"


# ── Shape mismatch detection ────────────────────────────────────────────────


def test_mismatch_raises_at_chain_time():
    ks = _ks(2)
    b1 = fx.block(fx.layers.SpectralBlock2d(3, 32, n_modes=4, key=ks[0]))
    b2 = fx.block(fx.layers.SpectralBlock2d(64, 1, n_modes=4, key=ks[1]))
    with pytest.raises(ShapeMismatchError):
        _ = b1 | b2


def test_mismatch_error_shows_channel_numbers():
    ks = _ks(2)
    b1 = fx.block(fx.layers.SpectralBlock2d(3, 32, n_modes=4, key=ks[0]), name="enc")
    b2 = fx.block(fx.layers.SpectralBlock2d(64, 1, n_modes=4, key=ks[1]), name="dec")
    with pytest.raises(ShapeMismatchError) as exc:
        _ = b1 | b2
    msg = str(exc.value)
    assert "32" in msg
    assert "64" in msg
    assert "enc" in msg
    assert "dec" in msg


def test_mismatch_in_longer_pipe():
    ks = _ks(3)
    b1 = fx.block(fx.layers.SpectralBlock2d(3, 16, n_modes=4, key=ks[0]))
    b2 = fx.block(fx.layers.SpectralBlock2d(16, 16, n_modes=4, key=ks[1]))
    b3 = fx.block(fx.layers.SpectralBlock2d(32, 1, n_modes=4, key=ks[2]))
    with pytest.raises(ShapeMismatchError):
        _ = b1 | b2 | b3


def test_unknown_channels_no_error_at_chain_time():
    """When channels are not statically known, chaining succeeds silently."""
    raw = eqx.nn.MLP(in_size=32, out_size=1, width_size=16, depth=2, key=_ks(1)[0])
    b_unknown = fx.block(raw)
    b_spectral = fx.block(fx.layers.SpectralBlock2d(3, 32, n_modes=4, key=_ks(1)[0]))
    pipe = b_spectral | b_unknown
    assert isinstance(pipe, Pipe)


# ── Combinators ─────────────────────────────────────────────────────────────


def test_dot_combinator_output_shape():
    ks = _ks(2)
    branch = fx.block(fx.mlp(in_features=3, output_dim=16, hidden_dims=16, key=ks[0]))
    trunk = fx.block(fx.mlp(in_features=2, output_dim=16, hidden_dims=16, key=ks[1]))
    model = fx.dot(branch, trunk)
    u = jnp.ones((3,))  # sensor values
    y = jnp.ones((10, 2))  # 10 query coordinates in 2-D
    out = model(u, y)
    assert out.shape == (10,)


def test_add_combinator_output_shape():
    ks = _ks(2)
    a = fx.block(fx.layers.SpectralBlock2d(4, 4, n_modes=4, key=ks[0]))
    b = fx.block(fx.layers.SpectralBlock2d(4, 4, n_modes=4, key=ks[1]))
    model = fx.add(a, b)
    out = model(jnp.ones((8, 8, 4)))
    assert out.shape == (8, 8, 4)


def test_cat_combinator_output_shape():
    ks = _ks(2)
    a = fx.block(fx.layers.SpectralBlock2d(4, 16, n_modes=4, key=ks[0]))
    b = fx.block(fx.layers.SpectralBlock2d(4, 8, n_modes=4, key=ks[1]))
    model = fx.cat(a, b)
    out = model(jnp.ones((8, 8, 4)))
    assert out.shape == (8, 8, 24)  # 16 + 8


def test_combinator_is_callable():
    ks = _ks(2)
    branch = fx.block(fx.mlp(in_features=3, output_dim=16, hidden_dims=16, key=ks[0]))
    trunk = fx.block(fx.mlp(in_features=2, output_dim=16, hidden_dims=16, key=ks[1]))
    comb = fx.dot(branch, trunk)
    assert callable(comb)


# ── Equinox pytree compatibility ────────────────────────────────────────────


def test_pipe_is_jittable():
    ks = _ks(2)
    b1 = fx.block(fx.layers.SpectralBlock2d(3, 16, n_modes=4, key=ks[0]))
    b2 = fx.block(fx.layers.SpectralBlock2d(16, 1, n_modes=4, key=ks[1]))
    pipe = b1 | b2

    @jax.jit
    def forward(model, x):
        return model(x)

    y = forward(pipe, jnp.ones((8, 8, 3)))
    assert y.shape == (8, 8, 1)


def test_pipe_gradients_flow():
    ks = _ks(2)
    b1 = fx.block(fx.layers.SpectralBlock2d(3, 16, n_modes=4, key=ks[0]))
    b2 = fx.block(fx.layers.SpectralBlock2d(16, 1, n_modes=4, key=ks[1]))
    pipe = b1 | b2

    @eqx.filter_jit
    @eqx.filter_grad
    def loss(model, x):
        return jnp.mean(model(x) ** 2)

    grads = loss(pipe, jnp.ones((8, 8, 3)))
    leaves = jax.tree_util.tree_leaves(grads)
    assert any(jnp.any(jnp.isfinite(g)) for g in leaves if hasattr(g, "shape"))
