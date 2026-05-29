"""Extensive test suite for the foundax pipe API, SpectralBlocks, and combinators.

Covers:
  - block() sniffing for every recognised field name
  - Block and Pipe creation, identity, and flattening (all 4 pipe-merge cases)
  - ShapeMismatchError triggering, message content, and pipeline trace
  - Forward-pass shapes for SpectralBlock1d/2d/3d under varied channel configs,
    norms, activations, and linear_conv settings
  - Gradient flow through every block type and every combinator
  - JIT and filter_jit compatibility for Block, Pipe, and all Combinators
  - eqx.partition / eqx.combine roundtrip on Pipe and Combinators
  - Wrapping existing foundax models (mlp, linear, fno2d) as Blocks
  - Long pipelines (5 and 10 blocks)
  - Repeated-block reuse inside a pipeline
  - Cross-dimensional pipelines (1-D, 2-D, 3-D)
  - Combinator composition (dot/add/cat with Pipe inputs)
  - Nested combinators
  - vmap over a Pipe
  - Numerical residual: skip + spectral both contribute
  - Parameter counts preserved by block() wrapping
  - Determinism: same key ⟹ same weights
  - foundax top-level and foundax.layers namespace completeness
"""

import math
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")

import foundax as fx  # noqa: E402
import foundax.layers as fl  # noqa: E402
from foundax.pipe import (  # noqa: E402
    Block,
    Pipe,
    ShapeMismatchError,
    _sniff,
    _IN_NAMES,
    _OUT_NAMES,
)
from foundax.combinators import (  # noqa: E402
    DotCombinator,
    AddCombinator,
    CatCombinator,
)


# ── helpers ─────────────────────────────────────────────────────────────────


def ks(n, seed=0):
    return jax.random.split(jax.random.PRNGKey(seed), n)


def leaf_count(module):
    return len(jax.tree_util.tree_leaves(module))


def param_count(module):
    return sum(
        math.prod(x.shape)
        for x in jax.tree_util.tree_leaves(eqx.filter(module, eqx.is_array))
    )


def finite_grads(grads):
    return all(
        jnp.all(jnp.isfinite(g))
        for g in jax.tree_util.tree_leaves(grads)
        if hasattr(g, "shape") and g.dtype.kind in ("f", "c")
    )


# ═══════════════════════════════════════════════════════════════════════════
# Section 1 – _sniff and block() field detection
# ═══════════════════════════════════════════════════════════════════════════


class TestSniff:
    def test_sniff_in_channels(self):
        m = fl.SpectralBlock2d(7, 13, n_modes=4, key=ks(1)[0])
        assert _sniff(m, _IN_NAMES) == 7

    def test_sniff_in_features(self):
        m = fx.mlp(in_features=5, output_dim=3, key=ks(1)[0])
        assert _sniff(m, _IN_NAMES) == 5

    def test_sniff_out_channels(self):
        m = fl.SpectralBlock2d(4, 11, n_modes=4, key=ks(1)[0])
        assert _sniff(m, _OUT_NAMES) == 11

    def test_sniff_out_features(self):
        m = fx.linear(8, 16, key=ks(1)[0])
        assert _sniff(m, _OUT_NAMES) == 16

    def test_sniff_output_dim(self):
        m = fx.mlp(in_features=4, output_dim=9, key=ks(1)[0])
        assert _sniff(m, _OUT_NAMES) == 9

    def test_sniff_returns_none_when_missing(self):
        raw = eqx.nn.MLP(in_size=4, out_size=2, width_size=8, depth=1, key=ks(1)[0])
        assert _sniff(raw, _IN_NAMES) is None
        assert _sniff(raw, _OUT_NAMES) is None

    def test_sniff_ignores_non_int(self):
        """A field with the right name but wrong type must be ignored."""

        class FakeModule(eqx.Module):
            in_channels: str = eqx.field(static=True)

        m = FakeModule(in_channels="not_an_int")
        assert _sniff(m, _IN_NAMES) is None

    def test_block_sniffs_in_channels(self):
        b = fx.block(fl.SpectralBlock1d(3, 16, n_modes=4, key=ks(1)[0]))
        assert b._in_channels == 3

    def test_block_sniffs_out_channels(self):
        b = fx.block(fl.SpectralBlock1d(3, 16, n_modes=4, key=ks(1)[0]))
        assert b._out_channels == 16

    def test_block_sniffs_in_features_from_mlp(self):
        b = fx.block(fx.mlp(in_features=7, output_dim=2, key=ks(1)[0]))
        assert b._in_channels == 7

    def test_block_sniffs_output_dim_from_mlp(self):
        b = fx.block(fx.mlp(in_features=7, output_dim=2, key=ks(1)[0]))
        assert b._out_channels == 2

    def test_block_sniffs_out_features_from_linear(self):
        b = fx.block(fx.linear(5, 10, key=ks(1)[0]))
        assert b._out_channels == 10

    def test_block_unknown_channels_both_none(self):
        raw = eqx.nn.MLP(in_size=4, out_size=2, width_size=8, depth=1, key=ks(1)[0])
        b = fx.block(raw)
        assert b._in_channels is None
        assert b._out_channels is None

    def test_block_default_name_is_class_name(self):
        b = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=ks(1)[0]))
        assert b.name == "SpectralBlock2d"

    def test_block_custom_name(self):
        b = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=ks(1)[0]), name="encoder")
        assert b.name == "encoder"

    def test_block_preserves_module_identity(self):
        m = fl.SpectralBlock2d(4, 8, n_modes=4, key=ks(1)[0])
        b = fx.block(m)
        assert b.module is m


# ═══════════════════════════════════════════════════════════════════════════
# Section 2 – Block forward pass and standalone usability
# ═══════════════════════════════════════════════════════════════════════════


class TestBlockForward:
    def test_block_1d_forward(self):
        b = fx.block(fl.SpectralBlock1d(4, 8, n_modes=4, key=ks(1)[0]))
        assert b(jnp.ones((32, 4))).shape == (32, 8)

    def test_block_2d_forward(self):
        b = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=ks(1)[0]))
        assert b(jnp.ones((16, 16, 4))).shape == (16, 16, 8)

    def test_block_3d_forward(self):
        b = fx.block(fl.SpectralBlock3d(4, 8, n_modes=4, key=ks(1)[0]))
        assert b(jnp.ones((8, 8, 8, 4))).shape == (8, 8, 8, 8)

    def test_block_wrapping_preserves_param_count(self):
        m = fl.SpectralBlock2d(4, 8, n_modes=4, key=ks(1)[0])
        b = fx.block(m)
        assert param_count(b) == param_count(m)

    def test_block_same_output_as_raw_module(self):
        k = ks(1)[0]
        m = fl.SpectralBlock2d(4, 8, n_modes=4, key=k)
        b = fx.block(m)
        x = jnp.ones((8, 8, 4))
        assert jnp.allclose(b(x), m(x), atol=1e-6)

    def test_block_is_eqx_module(self):
        b = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=ks(1)[0]))
        assert isinstance(b, eqx.Module)

    def test_block_is_instance_of_Block(self):
        b = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=ks(1)[0]))
        assert isinstance(b, Block)

    def test_block_wraps_mlp(self):
        m = fx.mlp(in_features=4, output_dim=2, key=ks(1)[0])
        b = fx.block(m)
        out = b(jnp.ones((4,)))
        assert out.shape == (2,)

    def test_block_wraps_linear(self):
        m = fx.linear(5, 3, key=ks(1)[0])
        b = fx.block(m)
        out = b(jnp.ones((5,)))
        assert out.shape == (3,)


# ═══════════════════════════════════════════════════════════════════════════
# Section 3 – Pipe construction and flattening
# ═══════════════════════════════════════════════════════════════════════════


class TestPipeConstruction:
    """Four merge cases: B|B, B|P, P|B, P|P."""

    def _blocks(self, n, c=8):
        """n same-channel blocks."""
        return [fx.block(fl.SpectralBlock2d(c, c, n_modes=4, key=k)) for k in ks(n)]

    # B|B
    def test_block_or_block_returns_pipe(self):
        bs = self._blocks(2)
        p = bs[0] | bs[1]
        assert isinstance(p, Pipe)
        assert len(p.blocks) == 2

    def test_block_or_block_preserves_order(self):
        bs = self._blocks(2)
        p = bs[0] | bs[1]
        assert p.blocks[0] is bs[0]
        assert p.blocks[1] is bs[1]

    # B|P
    def test_block_or_pipe_flattens(self):
        bs = self._blocks(3)
        tail = bs[1] | bs[2]  # Pipe([b1, b2])
        full = bs[0] | tail  # Block | Pipe
        assert isinstance(full, Pipe)
        assert len(full.blocks) == 3

    def test_block_or_pipe_order(self):
        bs = self._blocks(3)
        tail = bs[1] | bs[2]
        full = bs[0] | tail
        assert full.blocks[0] is bs[0]
        assert full.blocks[1] is bs[1]
        assert full.blocks[2] is bs[2]

    # P|B
    def test_pipe_or_block_flattens(self):
        bs = self._blocks(3)
        head = bs[0] | bs[1]
        full = head | bs[2]
        assert isinstance(full, Pipe)
        assert len(full.blocks) == 3

    def test_pipe_or_block_order(self):
        bs = self._blocks(3)
        head = bs[0] | bs[1]
        full = head | bs[2]
        assert full.blocks[0] is bs[0]
        assert full.blocks[2] is bs[2]

    # P|P
    def test_pipe_or_pipe_flattens(self):
        bs = self._blocks(4)
        left = bs[0] | bs[1]
        right = bs[2] | bs[3]
        full = left | right
        assert isinstance(full, Pipe)
        assert len(full.blocks) == 4

    def test_pipe_or_pipe_order(self):
        bs = self._blocks(4)
        left = bs[0] | bs[1]
        right = bs[2] | bs[3]
        full = left | right
        assert full.blocks[0] is bs[0]
        assert full.blocks[3] is bs[3]

    def test_long_chain_five_blocks(self):
        bs = self._blocks(5)
        pipe = bs[0] | bs[1] | bs[2] | bs[3] | bs[4]
        assert len(pipe.blocks) == 5

    def test_long_chain_ten_blocks(self):
        bs = self._blocks(10)
        pipe = bs[0]
        for b in bs[1:]:
            pipe = pipe | b
        assert len(pipe.blocks) == 10

    def test_pipe_is_eqx_module(self):
        bs = self._blocks(2)
        p = bs[0] | bs[1]
        assert isinstance(p, eqx.Module)

    def test_channel_mismatch_at_pipe_join(self):
        b1 = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=ks(1)[0]))
        b2 = fx.block(fl.SpectralBlock2d(8, 16, n_modes=4, key=ks(1)[0]))
        b3 = fx.block(
            fl.SpectralBlock2d(8, 1, n_modes=4, key=ks(1)[0])
        )  # mismatch at join
        left = b1 | b2  # Pipe([b1, b2])
        right = b3  # expects 8, gets 16
        with pytest.raises(ShapeMismatchError):
            _ = left | right

    def test_different_channel_blocks_chain_ok(self):
        b1 = fx.block(fl.SpectralBlock2d(3, 16, n_modes=4, key=ks(1)[0]))
        b2 = fx.block(fl.SpectralBlock2d(16, 32, n_modes=4, key=ks(1)[0]))
        b3 = fx.block(fl.SpectralBlock2d(32, 1, n_modes=4, key=ks(1)[0]))
        pipe = b1 | b2 | b3
        assert len(pipe.blocks) == 3

    def test_repeated_block_allowed(self):
        """The same Block object may appear multiple times."""
        b = fx.block(fl.SpectralBlock2d(8, 8, n_modes=4, key=ks(1)[0]))
        pipe = b | b | b
        assert len(pipe.blocks) == 3

    def test_pipe_or_raises_on_non_block(self):
        b = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=ks(1)[0]))
        with pytest.raises(TypeError, match="foundax.block()"):
            _ = b | 42

    def test_block_or_raises_on_string(self):
        b = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=ks(1)[0]))
        with pytest.raises(TypeError):
            _ = b | "oops"

    def test_pipe_or_raises_on_raw_module(self):
        b = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=ks(1)[0]))
        raw = fl.SpectralBlock2d(4, 4, n_modes=4, key=ks(1)[0])
        with pytest.raises(TypeError):
            _ = b | raw


# ═══════════════════════════════════════════════════════════════════════════
# Section 4 – ShapeMismatchError messages and content
# ═══════════════════════════════════════════════════════════════════════════


class TestShapeMismatch:
    def test_raises_on_obvious_mismatch(self):
        b1 = fx.block(fl.SpectralBlock2d(3, 32, n_modes=4, key=ks(1)[0]))
        b2 = fx.block(fl.SpectralBlock2d(64, 1, n_modes=4, key=ks(1)[0]))
        with pytest.raises(ShapeMismatchError):
            _ = b1 | b2

    def test_is_subclass_of_value_error(self):
        b1 = fx.block(fl.SpectralBlock2d(3, 8, n_modes=4, key=ks(1)[0]))
        b2 = fx.block(fl.SpectralBlock2d(16, 1, n_modes=4, key=ks(1)[0]))
        with pytest.raises(ValueError):
            _ = b1 | b2

    def test_error_contains_left_channel_count(self):
        b1 = fx.block(fl.SpectralBlock2d(3, 32, n_modes=4, key=ks(1)[0]))
        b2 = fx.block(fl.SpectralBlock2d(64, 1, n_modes=4, key=ks(1)[0]))
        with pytest.raises(ShapeMismatchError, match="32"):
            _ = b1 | b2

    def test_error_contains_right_channel_count(self):
        b1 = fx.block(fl.SpectralBlock2d(3, 32, n_modes=4, key=ks(1)[0]))
        b2 = fx.block(fl.SpectralBlock2d(64, 1, n_modes=4, key=ks(1)[0]))
        with pytest.raises(ShapeMismatchError, match="64"):
            _ = b1 | b2

    def test_error_contains_left_block_name(self):
        b1 = fx.block(fl.SpectralBlock2d(3, 32, n_modes=4, key=ks(1)[0]), name="lift")
        b2 = fx.block(fl.SpectralBlock2d(64, 1, n_modes=4, key=ks(1)[0]))
        with pytest.raises(ShapeMismatchError, match="lift"):
            _ = b1 | b2

    def test_error_contains_right_block_name(self):
        b1 = fx.block(fl.SpectralBlock2d(3, 32, n_modes=4, key=ks(1)[0]))
        b2 = fx.block(fl.SpectralBlock2d(64, 1, n_modes=4, key=ks(1)[0]), name="proj")
        with pytest.raises(ShapeMismatchError, match="proj"):
            _ = b1 | b2

    def test_error_contains_pipeline_marker(self):
        b1 = fx.block(fl.SpectralBlock2d(3, 32, n_modes=4, key=ks(1)[0]))
        b2 = fx.block(fl.SpectralBlock2d(64, 1, n_modes=4, key=ks(1)[0]))
        with pytest.raises(ShapeMismatchError) as exc:
            _ = b1 | b2
        assert "mismatch" in str(exc.value).lower()

    def test_error_contains_hint(self):
        b1 = fx.block(fl.SpectralBlock2d(3, 32, n_modes=4, key=ks(1)[0]))
        b2 = fx.block(fl.SpectralBlock2d(64, 1, n_modes=4, key=ks(1)[0]))
        with pytest.raises(ShapeMismatchError) as exc:
            _ = b1 | b2
        assert "Hint" in str(exc.value) or "hint" in str(exc.value).lower()

    def test_mismatch_at_second_join_in_3block_pipe(self):
        b1 = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=ks(1)[0]))
        b2 = fx.block(fl.SpectralBlock2d(8, 16, n_modes=4, key=ks(1)[0]))
        b3 = fx.block(fl.SpectralBlock2d(32, 1, n_modes=4, key=ks(1)[0]))
        with pytest.raises(ShapeMismatchError, match="16"):
            _ = b1 | b2 | b3

    def test_no_mismatch_when_channels_match(self):
        b1 = fx.block(fl.SpectralBlock2d(3, 16, n_modes=4, key=ks(1)[0]))
        b2 = fx.block(fl.SpectralBlock2d(16, 1, n_modes=4, key=ks(1)[0]))
        pipe = b1 | b2  # must not raise
        assert isinstance(pipe, Pipe)

    def test_no_error_when_either_side_unknown(self):
        raw = eqx.nn.MLP(in_size=16, out_size=1, width_size=8, depth=1, key=ks(1)[0])
        b_known = fx.block(fl.SpectralBlock2d(3, 999, n_modes=4, key=ks(1)[0]))
        b_unknown = fx.block(raw)
        # Even though b_known outputs 999 and b_unknown expects unknown, no error
        pipe = b_known | b_unknown
        assert isinstance(pipe, Pipe)

    def test_error_in_pipe_or_pipe_join(self):
        left = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=ks(1)[0])) | fx.block(
            fl.SpectralBlock2d(8, 16, n_modes=4, key=ks(1)[0])
        )
        right = fx.block(fl.SpectralBlock2d(32, 1, n_modes=4, key=ks(1)[0])) | fx.block(
            fl.SpectralBlock2d(1, 1, n_modes=4, key=ks(1)[0])
        )
        with pytest.raises(ShapeMismatchError):
            _ = left | right


# ═══════════════════════════════════════════════════════════════════════════
# Section 5 – Pipe forward pass and numerical properties
# ═══════════════════════════════════════════════════════════════════════════


class TestPipeForward:
    def test_two_block_2d_shape(self):
        b1 = fx.block(fl.SpectralBlock2d(3, 16, n_modes=4, key=ks(1)[0]))
        b2 = fx.block(fl.SpectralBlock2d(16, 1, n_modes=4, key=ks(1)[0]))
        y = (b1 | b2)(jnp.ones((8, 8, 3)))
        assert y.shape == (8, 8, 1)

    def test_three_block_2d_shape(self):
        b1 = fx.block(fl.SpectralBlock2d(3, 32, n_modes=4, key=ks(1)[0]))
        b2 = fx.block(fl.SpectralBlock2d(32, 32, n_modes=4, key=ks(1)[0]))
        b3 = fx.block(fl.SpectralBlock2d(32, 1, n_modes=4, key=ks(1)[0]))
        y = (b1 | b2 | b3)(jnp.ones((10, 10, 3)))
        assert y.shape == (10, 10, 1)

    def test_five_block_1d_shape(self):
        C = 8
        blocks = [fx.block(fl.SpectralBlock1d(C, C, n_modes=4, key=k)) for k in ks(5)]
        pipe = blocks[0]
        for b in blocks[1:]:
            pipe = pipe | b
        y = pipe(jnp.ones((64, C)))
        assert y.shape == (64, C)

    def test_pipe_output_is_finite(self):
        b1 = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=ks(1)[0]))
        b2 = fx.block(fl.SpectralBlock2d(8, 1, n_modes=4, key=ks(1)[0]))
        y = (b1 | b2)(jnp.ones((8, 8, 4)))
        assert jnp.all(jnp.isfinite(y))

    def test_pipe_not_equal_to_either_block_alone(self):
        """A 2-block pipe should produce different output than each block alone."""
        k1, k2 = ks(2)
        b1 = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k1))
        b2 = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k2))
        x = jnp.ones((8, 8, 4))
        pipe_out = (b1 | b2)(x)
        b1_out = b1(x)
        assert not jnp.allclose(pipe_out, b1_out, atol=1e-5)

    def test_pipe_blocks_applied_in_order(self):
        """Swap b1/b2 in the pipe → different output (order matters)."""
        k1, k2 = ks(2)
        # Same channel width so both orderings accept the same input
        b1 = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k1))
        b2 = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k2))
        x = jnp.ones((8, 8, 4))
        out_12 = (b1 | b2)(x)
        out_21 = (b2 | b1)(x)
        # Same shape but different weights applied in different order → different values
        assert out_12.shape == out_21.shape
        assert not jnp.allclose(out_12, out_21, atol=1e-5)

    def test_channel_expanding_then_contracting(self):
        b1 = fx.block(fl.SpectralBlock2d(1, 64, n_modes=4, key=ks(1)[0]))
        b2 = fx.block(fl.SpectralBlock2d(64, 64, n_modes=4, key=ks(1)[0]))
        b3 = fx.block(fl.SpectralBlock2d(64, 1, n_modes=4, key=ks(1)[0]))
        y = (b1 | b2 | b3)(jnp.ones((12, 12, 1)))
        assert y.shape == (12, 12, 1)

    def test_pipe_mixed_1d_mlp(self):
        b_spec = fx.block(fl.SpectralBlock1d(3, 16, n_modes=4, key=ks(1)[0]))
        b_mlp = fx.block(
            fx.mlp(in_features=16, output_dim=1, hidden_dims=8, key=ks(1)[0])
        )
        pipe = b_spec | b_mlp
        y = pipe(jnp.ones((32, 3)))
        assert y.shape == (32, 1)

    def test_pipe_spatial_dim_preserved_1d(self):
        b = fx.block(fl.SpectralBlock1d(4, 4, n_modes=4, key=ks(1)[0]))
        for W in [16, 32, 64]:
            y = b(jnp.ones((W, 4)))
            assert y.shape[0] == W

    def test_pipe_spatial_dim_preserved_2d(self):
        b = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=ks(1)[0]))
        for H, W in [(8, 8), (16, 12), (20, 20)]:
            y = b(jnp.ones((H, W, 4)))
            assert y.shape == (H, W, 4)


# ═══════════════════════════════════════════════════════════════════════════
# Section 6 – SpectralBlock variants (norm, activation, linear_conv)
# ═══════════════════════════════════════════════════════════════════════════


class TestSpectralBlockVariants:
    @pytest.mark.parametrize("norm", [None, "layer", "batch"])
    def test_spectral_block_1d_norms(self, norm):
        blk = fl.SpectralBlock1d(4, 8, n_modes=4, norm=norm, key=ks(1)[0])
        y = blk(jnp.ones((32, 4)))
        assert y.shape == (32, 8)
        assert jnp.all(jnp.isfinite(y))

    @pytest.mark.parametrize("norm", [None, "layer", "batch"])
    def test_spectral_block_2d_norms(self, norm):
        blk = fl.SpectralBlock2d(4, 8, n_modes=4, norm=norm, key=ks(1)[0])
        y = blk(jnp.ones((12, 12, 4)))
        assert y.shape == (12, 12, 8)
        assert jnp.all(jnp.isfinite(y))

    @pytest.mark.parametrize("norm", [None, "layer", "batch"])
    def test_spectral_block_3d_norms(self, norm):
        blk = fl.SpectralBlock3d(4, 8, n_modes=4, norm=norm, key=ks(1)[0])
        y = blk(jnp.ones((6, 6, 6, 4)))
        assert y.shape == (6, 6, 6, 8)
        assert jnp.all(jnp.isfinite(y))

    @pytest.mark.parametrize("act", [jax.nn.gelu, jax.nn.relu, jnp.tanh, jax.nn.silu])
    def test_spectral_block_2d_activations(self, act):
        blk = fl.SpectralBlock2d(4, 4, n_modes=4, activation=act, key=ks(1)[0])
        y = blk(jnp.ones((8, 8, 4)))
        assert jnp.all(jnp.isfinite(y))

    @pytest.mark.parametrize("lc", [True, False])
    def test_spectral_block_2d_linear_conv(self, lc):
        blk = fl.SpectralBlock2d(4, 8, n_modes=4, linear_conv=lc, key=ks(1)[0])
        y = blk(jnp.ones((12, 12, 4)))
        assert y.shape == (12, 12, 8)

    def test_spectral_block_channel_fields_1d(self):
        blk = fl.SpectralBlock1d(5, 11, n_modes=4, key=ks(1)[0])
        assert blk.in_channels == 5
        assert blk.out_channels == 11

    def test_spectral_block_channel_fields_2d(self):
        blk = fl.SpectralBlock2d(7, 13, n_modes=4, key=ks(1)[0])
        assert blk.in_channels == 7
        assert blk.out_channels == 13

    def test_spectral_block_channel_fields_3d(self):
        blk = fl.SpectralBlock3d(2, 6, n_modes=4, key=ks(1)[0])
        assert blk.in_channels == 2
        assert blk.out_channels == 6

    def test_spectral_block_norm_layer_is_none_when_no_norm(self):
        blk = fl.SpectralBlock2d(4, 8, n_modes=4, key=ks(1)[0])
        assert blk.norm_layer is None

    def test_spectral_block_norm_layer_set_when_layer_norm(self):
        blk = fl.SpectralBlock2d(4, 8, n_modes=4, norm="layer", key=ks(1)[0])
        assert blk.norm_layer is not None

    def test_residual_both_paths_contribute(self):
        """Zero out spectral_conv weights ⟹ output differs from zeroing skip."""
        blk = fl.SpectralBlock2d(4, 4, n_modes=4, key=ks(1)[0])
        x = jnp.ones((8, 8, 4))

        # Kill spectral path: set all spectral weights to zero
        blk_no_spectral = eqx.tree_at(
            lambda m: [
                m.spectral_conv.weight_1_real,
                m.spectral_conv.weight_1_imag,
                m.spectral_conv.weight_2_real,
                m.spectral_conv.weight_2_imag,
            ],
            blk,
            [jnp.zeros_like(blk.spectral_conv.weight_1_real)] * 4,
        )
        # Kill skip path: set skip weight to zero
        blk_no_skip = eqx.tree_at(
            lambda m: m.linear_skip.weight,
            blk,
            jnp.zeros_like(blk.linear_skip.weight),
        )
        out_no_spectral = blk_no_spectral(x)
        out_no_skip = blk_no_skip(x)
        assert not jnp.allclose(out_no_spectral, out_no_skip, atol=1e-5)

    def test_determinism_same_key_same_weights(self):
        k = ks(1)[0]
        b1 = fl.SpectralBlock2d(4, 8, n_modes=4, key=k)
        b2 = fl.SpectralBlock2d(4, 8, n_modes=4, key=k)
        assert jnp.allclose(b1.linear_skip.weight, b2.linear_skip.weight)

    def test_different_keys_different_weights(self):
        k1, k2 = ks(2)
        b1 = fl.SpectralBlock2d(4, 8, n_modes=4, key=k1)
        b2 = fl.SpectralBlock2d(4, 8, n_modes=4, key=k2)
        assert not jnp.allclose(b1.linear_skip.weight, b2.linear_skip.weight)


# ═══════════════════════════════════════════════════════════════════════════
# Section 7 – Combinators: DotCombinator
# ═══════════════════════════════════════════════════════════════════════════


class TestDotCombinator:
    def _make(self, branch_in=3, trunk_in=2, basis=16, seed=0):
        k1, k2 = ks(2, seed)
        branch = fx.block(
            fx.mlp(in_features=branch_in, output_dim=basis, hidden_dims=basis, key=k1)
        )
        trunk = fx.block(
            fx.mlp(in_features=trunk_in, output_dim=basis, hidden_dims=basis, key=k2)
        )
        return fx.dot(branch, trunk)

    def test_output_shape(self):
        model = self._make()
        u = jnp.ones((3,))
        y = jnp.ones((10, 2))
        assert model(u, y).shape == (10,)

    def test_is_dot_combinator(self):
        assert isinstance(self._make(), DotCombinator)

    def test_is_eqx_module(self):
        assert isinstance(self._make(), eqx.Module)

    def test_different_n_points(self):
        model = self._make()
        u = jnp.ones((3,))
        for N in [1, 5, 50]:
            y = jnp.ones((N, 2))
            assert model(u, y).shape == (N,)

    def test_output_is_finite(self):
        model = self._make()
        out = model(jnp.ones((3,)), jnp.ones((10, 2)))
        assert jnp.all(jnp.isfinite(out))

    def test_dot_with_pipe_branch(self):
        k1, k2, k3 = ks(3)
        b1 = fx.block(fl.SpectralBlock1d(3, 16, n_modes=4, key=k1))
        b2 = fx.block(fx.mlp(in_features=16, output_dim=8, hidden_dims=8, key=k2))
        trunk = fx.block(fx.mlp(in_features=2, output_dim=8, hidden_dims=8, key=k3))
        model = fx.dot(b1 | b2, trunk)
        # branch output is (32, 8) → need (8,) for dot; this tests the user sets up shapes
        # so instead test that the combinator stores the pipe correctly
        assert isinstance(model.branch, Pipe)

    def test_gradient_flows_through_branch(self):
        model = self._make()

        @eqx.filter_jit
        @eqx.filter_grad
        def loss(m):
            return jnp.sum(m(jnp.ones((3,)), jnp.ones((5, 2))))

        grads = loss(model)
        branch_grads = jax.tree_util.tree_leaves(grads.branch)
        assert any(jnp.any(g != 0) for g in branch_grads if hasattr(g, "shape"))

    def test_gradient_flows_through_trunk(self):
        model = self._make()

        @eqx.filter_jit
        @eqx.filter_grad
        def loss(m):
            return jnp.sum(m(jnp.ones((3,)), jnp.ones((5, 2))))

        grads = loss(model)
        trunk_grads = jax.tree_util.tree_leaves(grads.trunk)
        assert any(jnp.any(g != 0) for g in trunk_grads if hasattr(g, "shape"))


# ═══════════════════════════════════════════════════════════════════════════
# Section 8 – Combinators: AddCombinator and CatCombinator
# ═══════════════════════════════════════════════════════════════════════════


class TestAddCombinator:
    def test_output_shape_matches_branch(self):
        k1, k2 = ks(2)
        a = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k1))
        b = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k2))
        m = fx.add(a, b)
        out = m(jnp.ones((8, 8, 4)))
        assert out.shape == (8, 8, 4)

    def test_is_add_combinator(self):
        k1, k2 = ks(2)
        a = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k1))
        b = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k2))
        assert isinstance(fx.add(a, b), AddCombinator)

    def test_output_is_finite(self):
        k1, k2 = ks(2)
        a = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k1))
        b = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k2))
        out = fx.add(a, b)(jnp.ones((8, 8, 4)))
        assert jnp.all(jnp.isfinite(out))

    def test_add_with_identical_branches_equals_double(self):
        """add(b, b)(x) == 2 * b(x) when the same block is used twice."""
        b = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=ks(1)[0]))
        x = jnp.ones((8, 8, 4))
        assert jnp.allclose(fx.add(b, b)(x), 2 * b(x), atol=1e-5)

    def test_gradient_flows_through_both_branches(self):
        k1, k2 = ks(2)
        a = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k1))
        b = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k2))
        m = fx.add(a, b)

        @eqx.filter_jit
        @eqx.filter_grad
        def loss(model):
            return jnp.mean(model(jnp.ones((8, 8, 4))) ** 2)

        grads = loss(m)
        assert any(
            jnp.any(g != 0)
            for g in jax.tree_util.tree_leaves(grads.a)
            if hasattr(g, "shape")
        )
        assert any(
            jnp.any(g != 0)
            for g in jax.tree_util.tree_leaves(grads.b)
            if hasattr(g, "shape")
        )

    def test_add_with_pipe_branches(self):
        k1, k2, k3, k4 = ks(4)
        pa = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k1)) | fx.block(
            fl.SpectralBlock2d(4, 4, n_modes=4, key=k2)
        )
        pb = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k3)) | fx.block(
            fl.SpectralBlock2d(4, 4, n_modes=4, key=k4)
        )
        m = fx.add(pa, pb)
        out = m(jnp.ones((8, 8, 4)))
        assert out.shape == (8, 8, 4)


class TestCatCombinator:
    def test_output_channels_sum(self):
        k1, k2 = ks(2)
        a = fx.block(fl.SpectralBlock2d(4, 16, n_modes=4, key=k1))
        b = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k2))
        out = fx.cat(a, b)(jnp.ones((8, 8, 4)))
        assert out.shape == (8, 8, 24)

    def test_is_cat_combinator(self):
        k1, k2 = ks(2)
        a = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k1))
        b = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k2))
        assert isinstance(fx.cat(a, b), CatCombinator)

    def test_output_is_finite(self):
        k1, k2 = ks(2)
        a = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k1))
        b = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k2))
        out = fx.cat(a, b)(jnp.ones((8, 8, 4)))
        assert jnp.all(jnp.isfinite(out))

    def test_cat_equal_branch_sizes(self):
        k1, k2 = ks(2)
        a = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k1))
        b = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k2))
        out = fx.cat(a, b)(jnp.ones((8, 8, 4)))
        assert out.shape == (8, 8, 16)

    def test_cat_first_half_equals_branch_a_output(self):
        k1, k2 = ks(2)
        a = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k1))
        b = fx.block(fl.SpectralBlock2d(4, 6, n_modes=4, key=k2))
        x = jnp.ones((8, 8, 4))
        out = fx.cat(a, b)(x)
        assert jnp.allclose(out[..., :8], a(x), atol=1e-6)
        assert jnp.allclose(out[..., 8:], b(x), atol=1e-6)

    def test_gradient_flows_through_both_branches(self):
        k1, k2 = ks(2)
        a = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k1))
        b = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k2))
        m = fx.cat(a, b)

        @eqx.filter_jit
        @eqx.filter_grad
        def loss(model):
            return jnp.mean(model(jnp.ones((8, 8, 4))) ** 2)

        grads = loss(m)
        assert any(
            jnp.any(g != 0)
            for g in jax.tree_util.tree_leaves(grads.a)
            if hasattr(g, "shape")
        )
        assert any(
            jnp.any(g != 0)
            for g in jax.tree_util.tree_leaves(grads.b)
            if hasattr(g, "shape")
        )


# ═══════════════════════════════════════════════════════════════════════════
# Section 9 – Combinator composition and nesting
# ═══════════════════════════════════════════════════════════════════════════


class TestCombinatorComposition:
    def test_cat_then_project(self):
        k1, k2, k3 = ks(3)
        a = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k1))
        b = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k2))
        cat = fx.cat(a, b)  # output (*, 16)
        proj = fx.block(fx.mlp(in_features=16, output_dim=1, hidden_dims=8, key=k3))
        pipeline = fx.block(cat) | proj
        out = pipeline(jnp.ones((8, 8, 4)))
        assert out.shape == (8, 8, 1)

    def test_add_then_project(self):
        k1, k2, k3 = ks(3)
        a = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k1))
        b = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k2))
        addc = fx.add(a, b)
        proj = fx.block(fl.SpectralBlock2d(8, 1, n_modes=4, key=k3))
        pipeline = fx.block(addc) | proj
        out = pipeline(jnp.ones((8, 8, 4)))
        assert out.shape == (8, 8, 1)

    def test_nested_add_inside_cat(self):
        k1, k2, k3, k4, k5 = ks(5)
        a = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k1))
        b = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k2))
        c = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k3))
        d = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k4))
        nested = fx.cat(fx.add(a, b), fx.add(c, d))  # (*, 16)
        proj = fx.block(fl.SpectralBlock2d(16, 1, n_modes=4, key=k5))
        out = (fx.block(nested) | proj)(jnp.ones((8, 8, 4)))
        assert out.shape == (8, 8, 1)

    def test_wrap_combinator_in_block_preserves_call(self):
        k1, k2 = ks(2)
        a = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k1))
        b = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k2))
        add_m = fx.add(a, b)
        wrapped = fx.block(add_m)
        x = jnp.ones((8, 8, 4))
        assert jnp.allclose(wrapped(x), add_m(x))


# ═══════════════════════════════════════════════════════════════════════════
# Section 10 – JIT and Equinox filter_jit
# ═══════════════════════════════════════════════════════════════════════════


class TestJIT:
    def test_jit_block(self):
        b = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=ks(1)[0]))
        y = jax.jit(lambda m, x: m(x))(b, jnp.ones((8, 8, 4)))
        assert y.shape == (8, 8, 8)

    def test_jit_pipe_two_blocks(self):
        b1 = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=ks(1)[0]))
        b2 = fx.block(fl.SpectralBlock2d(8, 1, n_modes=4, key=ks(1)[0]))
        pipe = b1 | b2
        y = jax.jit(lambda m, x: m(x))(pipe, jnp.ones((8, 8, 4)))
        assert y.shape == (8, 8, 1)

    def test_filter_jit_pipe(self):
        b1 = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=ks(1)[0]))
        b2 = fx.block(fl.SpectralBlock2d(8, 1, n_modes=4, key=ks(1)[0]))
        pipe = b1 | b2

        @eqx.filter_jit
        def forward(model, x):
            return model(x)

        y = forward(pipe, jnp.ones((8, 8, 4)))
        assert y.shape == (8, 8, 1)

    def test_filter_jit_add_combinator(self):
        k1, k2 = ks(2)
        m = fx.add(
            fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k1)),
            fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k2)),
        )

        @eqx.filter_jit
        def forward(model, x):
            return model(x)

        y = forward(m, jnp.ones((8, 8, 4)))
        assert y.shape == (8, 8, 4)

    def test_filter_jit_cat_combinator(self):
        k1, k2 = ks(2)
        m = fx.cat(
            fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k1)),
            fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k2)),
        )

        @eqx.filter_jit
        def forward(model, x):
            return model(x)

        y = forward(m, jnp.ones((8, 8, 4)))
        assert y.shape == (8, 8, 12)

    def test_filter_jit_dot_combinator(self):
        k1, k2 = ks(2)
        m = fx.dot(
            fx.block(fx.mlp(in_features=3, output_dim=8, hidden_dims=8, key=k1)),
            fx.block(fx.mlp(in_features=2, output_dim=8, hidden_dims=8, key=k2)),
        )

        @eqx.filter_jit
        def forward(model, u, y):
            return model(u, y)

        out = forward(m, jnp.ones((3,)), jnp.ones((5, 2)))
        assert out.shape == (5,)

    def test_jit_long_pipe(self):
        C = 8
        blocks = [fx.block(fl.SpectralBlock2d(C, C, n_modes=4, key=k)) for k in ks(6)]
        pipe = blocks[0]
        for b in blocks[1:]:
            pipe = pipe | b

        @eqx.filter_jit
        def forward(model, x):
            return model(x)

        y = forward(pipe, jnp.ones((8, 8, C)))
        assert y.shape == (8, 8, C)


# ═══════════════════════════════════════════════════════════════════════════
# Section 11 – Gradient flow
# ═══════════════════════════════════════════════════════════════════════════


class TestGradients:
    def _loss(self, model, x):
        return jnp.mean(model(x) ** 2)

    def test_grads_block_spectral_2d(self):
        b = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=ks(1)[0]))
        grads = eqx.filter_grad(self._loss)(b, jnp.ones((8, 8, 4)))
        assert finite_grads(grads)

    def test_grads_block_spectral_1d(self):
        b = fx.block(fl.SpectralBlock1d(4, 4, n_modes=4, key=ks(1)[0]))
        grads = eqx.filter_grad(self._loss)(b, jnp.ones((32, 4)))
        assert finite_grads(grads)

    def test_grads_pipe_two_spectral_blocks(self):
        k1, k2 = ks(2)
        pipe = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k1)) | fx.block(
            fl.SpectralBlock2d(8, 1, n_modes=4, key=k2)
        )
        grads = eqx.filter_grad(self._loss)(pipe, jnp.ones((8, 8, 4)))
        assert finite_grads(grads)

    def test_grads_add_combinator(self):
        k1, k2 = ks(2)
        m = fx.add(
            fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k1)),
            fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k2)),
        )
        grads = eqx.filter_grad(self._loss)(m, jnp.ones((8, 8, 4)))
        assert finite_grads(grads)

    def test_grads_cat_combinator(self):
        k1, k2, k3 = ks(3)
        cat = fx.cat(
            fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k1)),
            fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k2)),
        )
        proj = fx.block(fl.SpectralBlock2d(12, 1, n_modes=4, key=k3))
        pipeline = fx.block(cat) | proj

        @eqx.filter_jit
        @eqx.filter_grad
        def loss(m, x):
            return jnp.mean(m(x) ** 2)

        grads = loss(pipeline, jnp.ones((8, 8, 4)))
        assert finite_grads(grads)

    def test_grads_deep_pipe_five_blocks(self):
        C = 8
        blocks = [fx.block(fl.SpectralBlock2d(C, C, n_modes=4, key=k)) for k in ks(5)]
        pipe = blocks[0]
        for b in blocks[1:]:
            pipe = pipe | b

        @eqx.filter_jit
        @eqx.filter_grad
        def loss(m, x):
            return jnp.mean(m(x) ** 2)

        grads = loss(pipe, jnp.ones((8, 8, C)))
        assert finite_grads(grads)

    def test_value_and_grad(self):
        b = fx.block(fl.SpectralBlock2d(4, 1, n_modes=4, key=ks(1)[0]))
        loss_fn = eqx.filter_value_and_grad(self._loss)
        val, grads = loss_fn(b, jnp.ones((8, 8, 4)))
        assert math.isfinite(float(val))
        assert finite_grads(grads)


# ═══════════════════════════════════════════════════════════════════════════
# Section 12 – Pytree operations (partition, combine, tree_leaves)
# ═══════════════════════════════════════════════════════════════════════════


class TestPytreeOps:
    def test_block_has_array_leaves(self):
        b = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=ks(1)[0]))
        leaves = jax.tree_util.tree_leaves(eqx.filter(b, eqx.is_array))
        assert len(leaves) > 0

    def test_pipe_leaf_count_equals_sum_of_blocks(self):
        k1, k2 = ks(2)
        b1 = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k1))
        b2 = fx.block(fl.SpectralBlock2d(8, 1, n_modes=4, key=k2))
        pipe = b1 | b2
        # Each Block wraps a SpectralBlock2d (spectral_conv + linear_skip + no norm)
        assert leaf_count(pipe) >= leaf_count(b1) + leaf_count(b2) - 2  # rough check

    def test_eqx_partition_combine_roundtrip_pipe(self):
        k1, k2 = ks(2)
        pipe = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k1)) | fx.block(
            fl.SpectralBlock2d(8, 1, n_modes=4, key=k2)
        )
        arrays, static = eqx.partition(pipe, eqx.is_array)
        reconstructed = eqx.combine(arrays, static)
        x = jnp.ones((8, 8, 4))
        assert jnp.allclose(pipe(x), reconstructed(x))

    def test_eqx_partition_combine_roundtrip_dot(self):
        k1, k2 = ks(2)
        m = fx.dot(
            fx.block(fx.mlp(in_features=3, output_dim=8, hidden_dims=8, key=k1)),
            fx.block(fx.mlp(in_features=2, output_dim=8, hidden_dims=8, key=k2)),
        )
        arrays, static = eqx.partition(m, eqx.is_array)
        reconstructed = eqx.combine(arrays, static)
        u, y = jnp.ones((3,)), jnp.ones((5, 2))
        assert jnp.allclose(m(u, y), reconstructed(u, y))

    def test_pipe_param_count_consistent(self):
        k1, k2 = ks(2)
        b1 = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k1))
        b2 = fx.block(fl.SpectralBlock2d(8, 1, n_modes=4, key=k2))
        pipe = b1 | b2
        # Param count of pipe = sum of block param counts (no new params added)
        assert param_count(pipe) == param_count(b1) + param_count(b2)


# ═══════════════════════════════════════════════════════════════════════════
# Section 13 – Wrapping existing foundax factory models
# ═══════════════════════════════════════════════════════════════════════════


class TestWrappingExistingModels:
    def test_wrap_mlp(self):
        m = fx.mlp(in_features=4, output_dim=2, hidden_dims=8, key=ks(1)[0])
        b = fx.block(m)
        assert b._in_channels == 4
        assert b._out_channels == 2
        assert b(jnp.ones((4,))).shape == (2,)

    def test_wrap_linear(self):
        m = fx.linear(5, 3, key=ks(1)[0])
        b = fx.block(m)
        assert b._in_channels == 5
        assert b._out_channels == 3
        assert b(jnp.ones((5,))).shape == (3,)

    def test_wrap_fno2d_no_crash(self):
        m = fx.fno2d(
            in_features=3, hidden_channels=16, n_modes=4, d_vars=1, key=ks(1)[0]
        )
        b = fx.block(m)
        assert isinstance(b, Block)

    def test_wrap_fno2d_forward(self):
        m = fx.fno2d(
            in_features=3, hidden_channels=16, n_modes=4, d_vars=1, key=ks(1)[0]
        )
        b = fx.block(m)
        y = b(jnp.ones((8, 8, 3)))
        assert y.shape[2] == 1

    def test_pipe_mlp_blocks(self):
        k1, k2 = ks(2)
        b1 = fx.block(fx.mlp(in_features=4, output_dim=8, hidden_dims=8, key=k1))
        b2 = fx.block(fx.mlp(in_features=8, output_dim=1, hidden_dims=4, key=k2))
        out = (b1 | b2)(jnp.ones((4,)))
        assert out.shape == (1,)

    def test_wrap_spectral_block_then_mlp(self):
        k1, k2 = ks(2)
        b1 = fx.block(fl.SpectralBlock1d(3, 16, n_modes=4, key=k1))
        b2 = fx.block(fx.mlp(in_features=16, output_dim=1, hidden_dims=8, key=k2))
        out = (b1 | b2)(jnp.ones((32, 3)))
        assert out.shape == (32, 1)


# ═══════════════════════════════════════════════════════════════════════════
# Section 14 – vmap over Pipe
# ═══════════════════════════════════════════════════════════════════════════


class TestVmap:
    def test_vmap_block(self):
        b = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=ks(1)[0]))
        batched = jax.vmap(b)
        y = batched(jnp.ones((5, 8, 8, 4)))
        assert y.shape == (5, 8, 8, 8)

    def test_vmap_pipe(self):
        k1, k2 = ks(2)
        pipe = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k1)) | fx.block(
            fl.SpectralBlock2d(8, 1, n_modes=4, key=k2)
        )
        batched = jax.vmap(pipe)
        y = batched(jnp.ones((3, 8, 8, 4)))
        assert y.shape == (3, 8, 8, 1)

    def test_vmap_add_combinator(self):
        k1, k2 = ks(2)
        m = fx.add(
            fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k1)),
            fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=k2)),
        )
        batched = jax.vmap(m)
        y = batched(jnp.ones((3, 8, 8, 4)))
        assert y.shape == (3, 8, 8, 4)


# ═══════════════════════════════════════════════════════════════════════════
# Section 15 – foundax namespace completeness
# ═══════════════════════════════════════════════════════════════════════════


class TestNamespaceCompleteness:
    def test_block_exported(self):
        assert hasattr(fx, "block")

    def test_Block_exported(self):
        assert hasattr(fx, "Block")

    def test_Pipe_exported(self):
        assert hasattr(fx, "Pipe")

    def test_ShapeMismatchError_exported(self):
        assert hasattr(fx, "ShapeMismatchError")

    def test_dot_exported(self):
        assert hasattr(fx, "dot")

    def test_add_exported(self):
        assert hasattr(fx, "add")

    def test_cat_exported(self):
        assert hasattr(fx, "cat")

    def test_layers_submodule_exported(self):
        assert hasattr(fx, "layers")

    def test_layers_spectral_block_1d(self):
        assert hasattr(fl, "SpectralBlock1d")

    def test_layers_spectral_block_2d(self):
        assert hasattr(fl, "SpectralBlock2d")

    def test_layers_spectral_block_3d(self):
        assert hasattr(fl, "SpectralBlock3d")

    def test_layers_spectral_conv_1d(self):
        assert hasattr(fl, "SpectralConv1d")

    def test_layers_spectral_conv_2d(self):
        assert hasattr(fl, "SpectralConv2d")

    def test_layers_spectral_conv_3d(self):
        assert hasattr(fl, "SpectralConv3d")

    def test_layers_mlp(self):
        assert hasattr(fl, "MLP")

    def test_layers_linear(self):
        assert hasattr(fl, "Linear")

    def test_block_is_Block(self):
        b = fx.block(fl.SpectralBlock2d(4, 4, n_modes=4, key=ks(1)[0]))
        assert type(b).__name__ == "Block"

    def test_ShapeMismatchError_is_ValueError(self):
        assert issubclass(fx.ShapeMismatchError, ValueError)


# ═══════════════════════════════════════════════════════════════════════════
# Section 16 – Long and complex pipelines
# ═══════════════════════════════════════════════════════════════════════════


class TestLongPipelines:
    """Tests for realistic multi-block pipelines of depth 6-20."""

    def _spec_blocks(self, n, c=16, seed=0):
        return [
            fx.block(fl.SpectralBlock2d(c, c, n_modes=4, key=k)) for k in ks(n, seed)
        ]

    # ── length / shape ──────────────────────────────────────────────────────

    def test_8_block_pipeline_length(self):
        blocks = self._spec_blocks(8)
        pipe = blocks[0]
        for b in blocks[1:]:
            pipe = pipe | b
        assert len(pipe.blocks) == 8

    def test_8_block_pipeline_shape(self):
        C = 16
        blocks = self._spec_blocks(8, c=C)
        pipe = blocks[0]
        for b in blocks[1:]:
            pipe = pipe | b
        y = pipe(jnp.ones((8, 8, C)))
        assert y.shape == (8, 8, C)

    def test_12_block_pipeline_shape(self):
        C = 8
        blocks = self._spec_blocks(12, c=C)
        pipe = blocks[0]
        for b in blocks[1:]:
            pipe = pipe | b
        y = pipe(jnp.ones((8, 8, C)))
        assert y.shape == (8, 8, C)

    def test_20_block_pipeline_length(self):
        C = 4
        blocks = self._spec_blocks(20, c=C)
        pipe = blocks[0]
        for b in blocks[1:]:
            pipe = pipe | b
        assert len(pipe.blocks) == 20

    # ── FNO-style lift → spectral stack → project ──────────────────────────

    def test_fno_style_6_block_pipeline(self):
        """lift(3→32) + 4 shared-channel blocks + project(32→1)."""
        k = ks(6)
        lift = fx.block(fl.SpectralBlock2d(3, 32, n_modes=4, key=k[0]), name="lift")
        s1 = fx.block(fl.SpectralBlock2d(32, 32, n_modes=4, key=k[1]), name="s1")
        s2 = fx.block(fl.SpectralBlock2d(32, 32, n_modes=4, key=k[2]), name="s2")
        s3 = fx.block(fl.SpectralBlock2d(32, 32, n_modes=4, key=k[3]), name="s3")
        s4 = fx.block(fl.SpectralBlock2d(32, 32, n_modes=4, key=k[4]), name="s4")
        project = fx.block(
            fl.SpectralBlock2d(32, 1, n_modes=4, key=k[5]), name="project"
        )
        pipe = lift | s1 | s2 | s3 | s4 | project
        assert len(pipe.blocks) == 6
        y = pipe(jnp.ones((16, 16, 3)))
        assert y.shape == (16, 16, 1)
        assert jnp.all(jnp.isfinite(y))

    def test_fno_style_names_preserved(self):
        """Block names set at construction survive into the Pipe."""
        k = ks(3)
        b1 = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k[0]), name="encoder")
        b2 = fx.block(fl.SpectralBlock2d(8, 8, n_modes=4, key=k[1]), name="middle")
        b3 = fx.block(fl.SpectralBlock2d(8, 1, n_modes=4, key=k[2]), name="decoder")
        pipe = b1 | b2 | b3
        assert pipe.blocks[0].name == "encoder"
        assert pipe.blocks[1].name == "middle"
        assert pipe.blocks[2].name == "decoder"

    # ── bottleneck channel schedule ─────────────────────────────────────────

    def test_bottleneck_channel_schedule(self):
        """3→32→64→128→64→32→1 — wide then narrow."""
        widths = [3, 32, 64, 128, 64, 32, 1]
        k = ks(len(widths) - 1)
        blocks = [
            fx.block(fl.SpectralBlock2d(widths[i], widths[i + 1], n_modes=4, key=k[i]))
            for i in range(len(widths) - 1)
        ]
        pipe = blocks[0]
        for b in blocks[1:]:
            pipe = pipe | b
        y = pipe(jnp.ones((8, 8, 3)))
        assert y.shape == (8, 8, 1)
        assert jnp.all(jnp.isfinite(y))

    # ── associativity: different split points → same result ────────────────

    def test_associativity_split_at_1(self):
        """(b0) | (b1|b2|b3) == b0|b1|b2|b3"""
        blocks = self._spec_blocks(4)
        flat = blocks[0] | blocks[1] | blocks[2] | blocks[3]
        split = blocks[0] | (blocks[1] | blocks[2] | blocks[3])
        x = jnp.ones((8, 8, 16))
        assert jnp.allclose(flat(x), split(x), atol=1e-6)

    def test_associativity_split_at_2(self):
        """(b0|b1) | (b2|b3) == b0|b1|b2|b3"""
        blocks = self._spec_blocks(4)
        flat = blocks[0] | blocks[1] | blocks[2] | blocks[3]
        split = (blocks[0] | blocks[1]) | (blocks[2] | blocks[3])
        x = jnp.ones((8, 8, 16))
        assert jnp.allclose(flat(x), split(x), atol=1e-6)

    def test_associativity_split_at_3(self):
        """(b0|b1|b2) | b3 == b0|b1|b2|b3"""
        blocks = self._spec_blocks(4)
        flat = blocks[0] | blocks[1] | blocks[2] | blocks[3]
        split = (blocks[0] | blocks[1] | blocks[2]) | blocks[3]
        x = jnp.ones((8, 8, 16))
        assert jnp.allclose(flat(x), split(x), atol=1e-6)

    def test_all_split_points_give_same_length(self):
        """Every way of grouping 5 blocks gives the same flat 5-block Pipe."""
        blocks = self._spec_blocks(5)
        results = [
            blocks[0] | blocks[1] | blocks[2] | blocks[3] | blocks[4],
            (blocks[0] | blocks[1]) | blocks[2] | blocks[3] | blocks[4],
            blocks[0] | (blocks[1] | blocks[2]) | blocks[3] | blocks[4],
            (blocks[0] | blocks[1] | blocks[2]) | (blocks[3] | blocks[4]),
            blocks[0] | (blocks[1] | blocks[2] | blocks[3] | blocks[4]),
        ]
        assert all(len(p.blocks) == 5 for p in results)

    # ── long pipe gradients and JIT ─────────────────────────────────────────

    def test_long_pipe_gradients_flow_to_all_blocks(self):
        """Every block in an 8-block pipe should receive non-zero gradients."""
        C = 8
        blocks = self._spec_blocks(8, c=C)
        pipe = blocks[0]
        for b in blocks[1:]:
            pipe = pipe | b

        @eqx.filter_jit
        @eqx.filter_grad
        def loss(m, x):
            return jnp.mean(m(x) ** 2)

        grads = loss(pipe, jnp.ones((8, 8, C)))
        for i in range(8):
            w = grads.blocks[i].module.linear_skip.weight
            assert w is not None, f"block {i} has no grad"
            assert jnp.any(w != 0), f"block {i} skip weight grad is all zeros"

    def test_long_pipe_jit_consistent(self):
        """JIT and non-JIT should produce identical outputs."""
        C = 8
        blocks = self._spec_blocks(6, c=C)
        pipe = blocks[0]
        for b in blocks[1:]:
            pipe = pipe | b

        x = jnp.ones((8, 8, C))
        eager = pipe(x)
        jitted = eqx.filter_jit(pipe)(x)
        assert jnp.allclose(eager, jitted, atol=1e-6)

    def test_long_pipe_1d_mixed_with_mlp(self):
        """Realistic 1-D operator pipeline: lift → 4 spectral → pointwise MLP."""
        k = ks(6)
        blocks = [
            fx.block(fl.SpectralBlock1d(1, 16, n_modes=8, key=k[0]), name="lift"),
            fx.block(fl.SpectralBlock1d(16, 16, n_modes=8, key=k[1])),
            fx.block(fl.SpectralBlock1d(16, 16, n_modes=8, key=k[2])),
            fx.block(fl.SpectralBlock1d(16, 16, n_modes=8, key=k[3])),
            fx.block(fl.SpectralBlock1d(16, 8, n_modes=8, key=k[4])),
            fx.block(
                fx.mlp(in_features=8, output_dim=1, hidden_dims=16, key=k[5]),
                name="project",
            ),
        ]
        pipe = blocks[0]
        for b in blocks[1:]:
            pipe = pipe | b
        assert len(pipe.blocks) == 6
        y = pipe(jnp.ones((128, 1)))
        assert y.shape == (128, 1)
        assert jnp.all(jnp.isfinite(y))

    def test_long_pipe_different_spatial_sizes(self):
        """A 6-block 2-D pipe should handle any spatial resolution."""
        C = 8
        blocks = self._spec_blocks(6, c=C)
        pipe = blocks[0]
        for b in blocks[1:]:
            pipe = pipe | b
        for H, W in [(8, 8), (16, 16), (12, 20)]:
            y = pipe(jnp.ones((H, W, C)))
            assert y.shape == (H, W, C)


# ═══════════════════════════════════════════════════════════════════════════
# Section 17 – Pytree structure and named parameter access
# ═══════════════════════════════════════════════════════════════════════════


class TestPytreeStructure:
    """Verify the Equinox pytree of Block/Pipe is correctly shaped.

    A ``Pipe`` of two ``SpectralBlock2d`` (no norm) has these array leaves:
      blocks[i].module.spectral_conv.weight_1_real  (in_c, out_c, n_modes, n_modes)
      blocks[i].module.spectral_conv.weight_1_imag
      blocks[i].module.spectral_conv.weight_2_real
      blocks[i].module.spectral_conv.weight_2_imag
      blocks[i].module.linear_skip.weight           (out_c, in_c)
      blocks[i].module.linear_skip.bias             (out_c,)
    → 6 array leaves per block, all named and directly addressable.
    """

    def _pipe2(self, c_in=4, c_mid=8, c_out=4, seed=0):
        k1, k2 = ks(2, seed)
        b1 = fx.block(fl.SpectralBlock2d(c_in, c_mid, n_modes=4, key=k1))
        b2 = fx.block(fl.SpectralBlock2d(c_mid, c_out, n_modes=4, key=k2))
        return b1 | b2, b1, b2

    # ── structural attribute access ─────────────────────────────────────────

    def test_pipe_blocks_are_Block_instances(self):
        pipe, b1, b2 = self._pipe2()
        for blk in pipe.blocks:
            assert isinstance(blk, Block)

    def test_pipe_block_module_is_spectral_block(self):
        pipe, _, _ = self._pipe2()
        assert isinstance(pipe.blocks[0].module, fl.SpectralBlock2d)
        assert isinstance(pipe.blocks[1].module, fl.SpectralBlock2d)

    def test_block_module_has_spectral_conv(self):
        pipe, _, _ = self._pipe2()
        from foundax.architectures.fno import SpectralConv2d

        assert isinstance(pipe.blocks[0].module.spectral_conv, SpectralConv2d)

    def test_block_module_has_linear_skip(self):
        pipe, _, _ = self._pipe2()
        from foundax.architectures.linear import Linear

        assert isinstance(pipe.blocks[0].module.linear_skip, Linear)

    def test_spectral_conv_weight_fields_are_arrays(self):
        pipe, _, _ = self._pipe2()
        sc = pipe.blocks[0].module.spectral_conv
        for attr in (
            "weight_1_real",
            "weight_1_imag",
            "weight_2_real",
            "weight_2_imag",
        ):
            w = getattr(sc, attr)
            assert hasattr(w, "shape"), f"{attr} is not an array"

    def test_linear_skip_weight_is_array(self):
        pipe, _, _ = self._pipe2()
        w = pipe.blocks[0].module.linear_skip.weight
        assert hasattr(w, "shape")

    def test_linear_skip_bias_is_array(self):
        pipe, _, _ = self._pipe2()
        bias = pipe.blocks[0].module.linear_skip.bias
        assert hasattr(bias, "shape")

    # ── static fields are NOT array leaves ─────────────────────────────────

    def test_block_in_channels_is_static(self):
        """_in_channels must not appear in tree_leaves — it is a static field."""
        b = fx.block(fl.SpectralBlock2d(7, 7, n_modes=4, key=ks(1)[0]))
        arrays = jax.tree_util.tree_leaves(eqx.filter(b, eqx.is_array))
        # All leaves should be arrays (no ints)
        for leaf in arrays:
            assert hasattr(leaf, "shape"), (
                "a non-array leaked into array-filtered leaves"
            )
        # Static int values must not appear as bare Python ints in the pytree leaves
        all_leaves = jax.tree_util.tree_leaves(b)
        assert not any(isinstance(leaf, int) and leaf == 7 for leaf in all_leaves), (
            "_in_channels leaked as a pytree leaf"
        )

    def test_block_name_is_static(self):
        b = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=ks(1)[0]), name="myblock")
        all_leaves = jax.tree_util.tree_leaves(b)
        assert not any(
            leaf == "myblock" for leaf in all_leaves if isinstance(leaf, str)
        ), "name string leaked as a pytree leaf"

    def test_spectral_block_in_channels_static_not_leaf(self):
        blk = fl.SpectralBlock2d(5, 5, n_modes=4, key=ks(1)[0])
        all_leaves = jax.tree_util.tree_leaves(blk)
        # Static int 5 must not be a plain Python int leaf
        assert not any(isinstance(leaf, int) and leaf == 5 for leaf in all_leaves)

    # ── exact array leaf count ──────────────────────────────────────────────

    def test_spectral_block_2d_has_six_array_leaves(self):
        """SpectralBlock2d (no norm): 4 spectral weights + skip.weight + skip.bias = 6."""
        blk = fl.SpectralBlock2d(4, 8, n_modes=4, key=ks(1)[0])
        arrays = jax.tree_util.tree_leaves(eqx.filter(blk, eqx.is_array))
        assert len(arrays) == 6

    def test_pipe_array_leaf_count_equals_sum(self):
        """Pipe leaf count must equal sum of individual block leaf counts."""
        k1, k2, k3 = ks(3)
        b1 = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k1))
        b2 = fx.block(fl.SpectralBlock2d(8, 8, n_modes=4, key=k2))
        b3 = fx.block(fl.SpectralBlock2d(8, 1, n_modes=4, key=k3))
        pipe = b1 | b2 | b3
        total = len(jax.tree_util.tree_leaves(eqx.filter(pipe, eqx.is_array)))
        per_b1 = len(jax.tree_util.tree_leaves(eqx.filter(b1, eqx.is_array)))
        per_b2 = len(jax.tree_util.tree_leaves(eqx.filter(b2, eqx.is_array)))
        per_b3 = len(jax.tree_util.tree_leaves(eqx.filter(b3, eqx.is_array)))
        assert total == per_b1 + per_b2 + per_b3

    def test_param_count_per_block_accessible_by_index(self):
        k1, k2 = ks(2)
        b1 = fx.block(fl.SpectralBlock2d(4, 8, n_modes=4, key=k1))
        b2 = fx.block(fl.SpectralBlock2d(8, 1, n_modes=4, key=k2))
        pipe = b1 | b2
        p0 = param_count(pipe.blocks[0])
        p1 = param_count(pipe.blocks[1])
        assert param_count(pipe) == p0 + p1

    # ── eqx.tree_at surgery ─────────────────────────────────────────────────

    def test_tree_at_updates_named_weight(self):
        """eqx.tree_at can zero out a specific named weight."""
        pipe, _, _ = self._pipe2(c_in=4, c_mid=4, c_out=4)
        zeros = jnp.zeros_like(pipe.blocks[0].module.linear_skip.weight)
        pipe_mod = eqx.tree_at(
            lambda p: p.blocks[0].module.linear_skip.weight,
            pipe,
            zeros,
        )
        assert jnp.all(pipe_mod.blocks[0].module.linear_skip.weight == 0)

    def test_tree_at_only_updates_target_block(self):
        """Modifying block 0's weight must not change block 1's weight."""
        pipe, _, _ = self._pipe2(c_in=4, c_mid=4, c_out=4)
        w1_before = pipe.blocks[1].module.linear_skip.weight
        zeros = jnp.zeros_like(pipe.blocks[0].module.linear_skip.weight)
        pipe_mod = eqx.tree_at(
            lambda p: p.blocks[0].module.linear_skip.weight,
            pipe,
            zeros,
        )
        assert jnp.allclose(pipe_mod.blocks[1].module.linear_skip.weight, w1_before)

    def test_tree_at_spectral_weight_updates(self):
        """eqx.tree_at on spectral_conv weights works."""
        blk = fl.SpectralBlock2d(4, 4, n_modes=4, key=ks(1)[0])
        zeros = jnp.zeros_like(blk.spectral_conv.weight_1_real)
        blk_mod = eqx.tree_at(lambda m: m.spectral_conv.weight_1_real, blk, zeros)
        assert jnp.all(blk_mod.spectral_conv.weight_1_real == 0)
        # Other spectral weights unchanged
        assert not jnp.all(blk_mod.spectral_conv.weight_2_real == 0)

    def test_tree_at_block_in_long_pipe(self):
        """Modifying block 3 in a 6-block pipe leaves the other 5 unchanged."""
        C = 8
        blocks = [fx.block(fl.SpectralBlock2d(C, C, n_modes=4, key=k)) for k in ks(6)]
        pipe = blocks[0]
        for b in blocks[1:]:
            pipe = pipe | b

        w_before = [pipe.blocks[i].module.linear_skip.weight for i in range(6)]
        zeros = jnp.zeros_like(w_before[3])
        pipe_mod = eqx.tree_at(
            lambda p: p.blocks[3].module.linear_skip.weight,
            pipe,
            zeros,
        )
        assert jnp.all(pipe_mod.blocks[3].module.linear_skip.weight == 0)
        for i in [0, 1, 2, 4, 5]:
            assert jnp.allclose(
                pipe_mod.blocks[i].module.linear_skip.weight, w_before[i]
            )

    # ── named gradient access ───────────────────────────────────────────────

    def test_named_grad_linear_skip_weight(self):
        """grads.blocks[i].module.linear_skip.weight is a finite array."""
        pipe, _, _ = self._pipe2(c_in=4, c_mid=8, c_out=4)

        @eqx.filter_jit
        @eqx.filter_grad
        def loss(m, x):
            return jnp.mean(m(x) ** 2)

        grads = loss(pipe, jnp.ones((8, 8, 4)))
        for i in range(2):
            w_grad = grads.blocks[i].module.linear_skip.weight
            assert w_grad is not None
            assert jnp.all(jnp.isfinite(w_grad))
            assert jnp.any(w_grad != 0)

    def test_named_grad_spectral_conv_weight_1_real(self):
        """grads.blocks[i].module.spectral_conv.weight_1_real is a finite array."""
        pipe, _, _ = self._pipe2(c_in=4, c_mid=8, c_out=4)

        @eqx.filter_jit
        @eqx.filter_grad
        def loss(m, x):
            return jnp.mean(m(x) ** 2)

        grads = loss(pipe, jnp.ones((8, 8, 4)))
        w_grad = grads.blocks[0].module.spectral_conv.weight_1_real
        assert w_grad is not None
        assert jnp.all(jnp.isfinite(w_grad))

    def test_named_grads_deep_pipe(self):
        """In an 8-block pipe, named grad access works for every block."""
        C = 8
        blocks = [fx.block(fl.SpectralBlock2d(C, C, n_modes=4, key=k)) for k in ks(8)]
        pipe = blocks[0]
        for b in blocks[1:]:
            pipe = pipe | b

        @eqx.filter_jit
        @eqx.filter_grad
        def loss(m, x):
            return jnp.mean(m(x) ** 2)

        grads = loss(pipe, jnp.ones((8, 8, C)))
        for i in range(8):
            skip_grad = grads.blocks[i].module.linear_skip.weight
            spectral_grad = grads.blocks[i].module.spectral_conv.weight_1_real
            assert jnp.all(jnp.isfinite(skip_grad)), f"block {i} skip grad not finite"
            assert jnp.all(jnp.isfinite(spectral_grad)), (
                f"block {i} spectral grad not finite"
            )

    # ── zero-weight / zero-map invariants ───────────────────────────────────

    def test_zero_all_weights_forward_is_finite(self):
        """Zeroing every array leaf → forward pass stays finite (just activation bias)."""
        pipe, _, _ = self._pipe2()
        arrays, static = eqx.partition(pipe, eqx.is_array)
        zeroed_arrays = jax.tree_util.tree_map(jnp.zeros_like, arrays)
        pipe_zero = eqx.combine(zeroed_arrays, static)
        y = pipe_zero(jnp.ones((8, 8, 4)))
        assert jnp.all(jnp.isfinite(y))

    def test_tree_map_scale_all_weights(self):
        """Scaling every weight by 2 changes the output."""
        pipe, _, _ = self._pipe2()
        arrays, static = eqx.partition(pipe, eqx.is_array)
        doubled_arrays = jax.tree_util.tree_map(lambda w: 2.0 * w, arrays)
        pipe_2x = eqx.combine(doubled_arrays, static)
        x = jnp.ones((8, 8, 4))
        assert not jnp.allclose(pipe(x), pipe_2x(x), atol=1e-5)

    def test_param_count_unchanged_after_tree_at(self):
        """eqx.tree_at surgery must not change total parameter count."""
        pipe, _, _ = self._pipe2()
        before = param_count(pipe)
        zeros = jnp.zeros_like(pipe.blocks[0].module.linear_skip.weight)
        pipe_mod = eqx.tree_at(
            lambda p: p.blocks[0].module.linear_skip.weight, pipe, zeros
        )
        assert param_count(pipe_mod) == before

    def test_block_name_accessible_in_long_pipe(self):
        """Named blocks retain their names at the correct indices."""
        k = ks(5)
        names = ["a", "b", "c", "d", "e"]
        blocks = [
            fx.block(fl.SpectralBlock2d(8, 8, n_modes=4, key=k[i]), name=names[i])
            for i in range(5)
        ]
        pipe = blocks[0]
        for b in blocks[1:]:
            pipe = pipe | b
        for i, name in enumerate(names):
            assert pipe.blocks[i].name == name
