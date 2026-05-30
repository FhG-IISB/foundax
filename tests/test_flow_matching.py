"""Tests for flow-matching / diffusion backbone architectures.

All tests use small configs that run on CPU in the default pixi environment.
No optax required.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

import foundax as fx
from foundax.architectures.time_embed import (
    SinusoidalTimeEmbedding,
    FiLMLayer,
    AdaLayerNorm,
    AdaLayerNormZero,
)
from foundax.architectures.ffno import (
    FactorizedSpectralBlock2d,
    FactorizedSpectralBlock3d,
)
from foundax.architectures.wno import (
    WaveletBlock1d,
    WaveletBlock2d,
    WaveletBlock3d,
    _DB8_LO,
    _DB8_HI,
    _dwt_axis,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

KEY = jax.random.PRNGKey(42)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Time-conditioning primitives
# ═══════════════════════════════════════════════════════════════════════════════


class TestTimeEmbedding:
    def test_sinusoidal_shape(self):
        emb_fn = SinusoidalTimeEmbedding(dim=32, key=KEY)
        out = emb_fn(0.5)
        assert out.shape == (32,)

    def test_sinusoidal_different_t(self):
        emb_fn = SinusoidalTimeEmbedding(dim=32, key=KEY)
        e1 = emb_fn(0.1)
        e2 = emb_fn(0.9)
        assert not jnp.allclose(e1, e2)

    def test_sinusoidal_jit(self):
        emb_fn = SinusoidalTimeEmbedding(dim=32, key=KEY)
        out = eqx.filter_jit(emb_fn)(0.5)
        assert out.shape == (32,)

    def test_film_shape(self):
        film = FiLMLayer(emb_dim=32, feature_dim=16, key=KEY)
        x = jnp.ones((8, 8, 16))
        emb = jnp.ones(32)
        out = film(x, emb)
        assert out.shape == (8, 8, 16)

    def test_film_modulates(self):
        film = FiLMLayer(emb_dim=32, feature_dim=16, key=KEY)
        x = jnp.ones((4, 16))
        e1 = jax.random.normal(KEY, (32,))
        e2 = jax.random.normal(jax.random.PRNGKey(1), (32,))
        assert not jnp.allclose(film(x, e1), film(x, e2))

    def test_adaLN_shape(self):
        norm = AdaLayerNorm(dim=16, emb_dim=32, key=KEY)
        x = jnp.ones(16)
        emb = jnp.ones(32)
        out = norm(x, emb)
        assert out.shape == (16,)

    def test_adaLN_zero_identity_at_init(self):
        # With fully zero-init projection, all conditioning outputs are zero
        ada = AdaLayerNormZero(dim=8, emb_dim=16, key=KEY)
        x = jax.random.normal(KEY, (4, 8))
        emb = jax.random.normal(KEY, (16,))
        x_normed, gate_attn, shift_mlp, scale_mlp, gate_mlp = ada(x, emb)
        # Both weight and bias are zero-init → all outputs are 0
        assert jnp.allclose(gate_attn, jnp.zeros(8))
        assert jnp.allclose(gate_mlp, jnp.zeros(8))
        assert jnp.allclose(shift_mlp, jnp.zeros(8))
        assert jnp.allclose(scale_mlp, jnp.zeros(8))

    def test_adaLN_zero_shape(self):
        ada = AdaLayerNormZero(dim=8, emb_dim=16, key=KEY)
        x = jax.random.normal(KEY, (6, 8))
        emb = jnp.ones(16)
        x_n, g_a, s_m, sc_m, g_m = ada(x, emb)
        assert x_n.shape == (6, 8)
        assert g_a.shape == (8,)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DiT
# ═══════════════════════════════════════════════════════════════════════════════


class TestDiT:
    # Use hidden_size divisible by 4 (2D) and 6 (3D)
    def test_dit2d_shape(self):
        model = fx.dit2d(in_channels=2, patch_size=4, hidden_size=48,
                         depth=2, num_heads=4, key=KEY)
        x = jax.random.normal(KEY, (16, 16, 2))
        out = model(x, t=0.5)
        assert out.shape == (16, 16, 2)

    def test_dit2d_different_t(self):
        model = fx.dit2d(in_channels=1, patch_size=4, hidden_size=48,
                         depth=2, num_heads=4, key=KEY)
        x = jax.random.normal(KEY, (16, 16, 1))
        o1 = model(x, t=0.1)
        o2 = model(x, t=0.9)
        assert not jnp.allclose(o1, o2)

    def test_dit2d_jit(self):
        model = fx.dit2d(in_channels=1, patch_size=4, hidden_size=48,
                         depth=2, num_heads=4, key=KEY)
        x = jax.random.normal(KEY, (16, 16, 1))
        out = eqx.filter_jit(model)(x, t=0.5)
        assert out.shape == (16, 16, 1)

    def test_dit2d_gradients_finite(self):
        model = fx.dit2d(in_channels=1, patch_size=4, hidden_size=48,
                         depth=2, num_heads=4, key=KEY)
        x = jax.random.normal(KEY, (16, 16, 1))

        def loss(m):
            return jnp.mean(m(x, t=0.5) ** 2)

        grads = eqx.filter_grad(loss)(model)
        leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves)

    def test_dit2d_pipe_sniffable(self):
        model = fx.dit2d(in_channels=2, patch_size=4, hidden_size=48,
                         depth=2, num_heads=4, key=KEY)
        blk = fx.block(model)
        assert blk._in_channels == 2
        assert blk._out_channels == 2

    def test_dit3d_shape(self):
        # hidden_size must be divisible by 6
        model = fx.dit3d(in_channels=1, patch_size=4, hidden_size=48,
                         depth=2, num_heads=4, key=KEY)
        x = jax.random.normal(KEY, (16, 16, 16, 1))
        out = model(x, t=0.5)
        assert out.shape == (16, 16, 16, 1)

    def test_dit3d_different_t(self):
        model = fx.dit3d(in_channels=1, patch_size=4, hidden_size=48,
                         depth=2, num_heads=4, key=KEY)
        x = jax.random.normal(KEY, (16, 16, 16, 1))
        o1 = model(x, t=0.1)
        o2 = model(x, t=0.9)
        assert not jnp.allclose(o1, o2)

    def test_dit3d_jit(self):
        model = fx.dit3d(in_channels=1, patch_size=4, hidden_size=48,
                         depth=2, num_heads=4, key=KEY)
        x = jax.random.normal(KEY, (16, 16, 16, 1))
        out = eqx.filter_jit(model)(x, t=0.5)
        assert out.shape == (16, 16, 16, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. F-FNO
# ═══════════════════════════════════════════════════════════════════════════════


class TestFFNO:
    def test_block2d_shape(self):
        blk = FactorizedSpectralBlock2d(4, 8, n_modes=4, key=KEY)
        x = jax.random.normal(KEY, (16, 16, 4))
        out = blk(x)
        assert out.shape == (16, 16, 8)

    def test_block2d_pipe_sniffable(self):
        blk = FactorizedSpectralBlock2d(4, 8, n_modes=4, key=KEY)
        wrapped = fx.block(blk)
        assert wrapped._in_channels == 4
        assert wrapped._out_channels == 8

    def test_block2d_film_conditioning(self):
        blk = FactorizedSpectralBlock2d(4, 4, n_modes=4, use_film=True, emb_dim=16, key=KEY)
        x = jax.random.normal(KEY, (16, 16, 4))
        emb = jax.random.normal(KEY, (16,))
        out_cond = blk(x, t_emb=emb)
        out_uncond = blk(x)
        assert out_cond.shape == (16, 16, 4)
        assert not jnp.allclose(out_cond, out_uncond)

    def test_block3d_shape(self):
        blk = FactorizedSpectralBlock3d(2, 4, n_modes=4, key=KEY)
        x = jax.random.normal(KEY, (8, 8, 8, 2))
        out = blk(x)
        assert out.shape == (8, 8, 8, 4)

    def test_block3d_pipe_sniffable(self):
        blk = FactorizedSpectralBlock3d(2, 4, n_modes=4, key=KEY)
        wrapped = fx.block(blk)
        assert wrapped._in_channels == 2
        assert wrapped._out_channels == 4

    def test_ffno2d_shape(self):
        model = fx.ffno2d(in_channels=2, hidden_channels=8, n_modes=4, n_layers=2, key=KEY)
        x = jax.random.normal(KEY, (16, 16, 2))
        out = model(x)
        assert out.shape == (16, 16, 2)

    def test_ffno2d_with_film(self):
        model = fx.ffno2d(in_channels=2, hidden_channels=8, n_modes=4, n_layers=2,
                          use_film=True, emb_dim=16, key=KEY)
        x = jax.random.normal(KEY, (16, 16, 2))
        emb = jax.random.normal(KEY, (16,))
        out = model(x, t_emb=emb)
        assert out.shape == (16, 16, 2)

    def test_ffno2d_jit(self):
        model = fx.ffno2d(in_channels=2, hidden_channels=8, n_modes=4, n_layers=2, key=KEY)
        x = jax.random.normal(KEY, (16, 16, 2))
        out = eqx.filter_jit(model)(x)
        assert out.shape == (16, 16, 2)

    def test_ffno2d_gradients_finite(self):
        model = fx.ffno2d(in_channels=2, hidden_channels=8, n_modes=4, n_layers=2, key=KEY)
        x = jax.random.normal(KEY, (16, 16, 2))

        def loss(m):
            return jnp.mean(m(x) ** 2)

        grads = eqx.filter_grad(loss)(model)
        leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves)

    def test_ffno3d_shape(self):
        model = fx.ffno3d(in_channels=1, hidden_channels=4, n_modes=4, n_layers=2, key=KEY)
        x = jax.random.normal(KEY, (8, 8, 8, 1))
        out = model(x)
        assert out.shape == (8, 8, 8, 1)

    def test_ffno2d_fewer_params_than_fno2d(self):
        # F-FNO should have fewer parameters than standard FNO2d with same hidden_channels
        ffno = fx.ffno2d(in_channels=1, hidden_channels=16, n_modes=8, n_layers=4, key=KEY)
        fno = fx.fno2d(in_features=1, hidden_channels=16, n_modes=8, n_layers=4, key=KEY)
        def count_params(m):
            return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array)))
        assert count_params(ffno) < count_params(fno)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. WNO
# ═══════════════════════════════════════════════════════════════════════════════


class TestWNO:
    def test_filter_coefficients(self):
        # Filters should be near unit energy
        assert abs(float(jnp.sum(_DB8_LO ** 2)) - 1.0) < 0.1
        assert abs(float(jnp.sum(_DB8_HI ** 2)) - 1.0) < 0.1

    def test_dwt_axis_shape_1d(self):
        x = jax.random.normal(KEY, (32, 4))  # (W, C)
        approx, detail = _dwt_axis(x, _DB8_LO, _DB8_HI, axis=0)
        assert approx.shape == (16, 4)
        assert detail.shape == (16, 4)

    def test_dwt_axis_shape_2d_along_w(self):
        x = jax.random.normal(KEY, (16, 32, 4))  # (H, W, C)
        approx, detail = _dwt_axis(x, _DB8_LO, _DB8_HI, axis=1)
        assert approx.shape == (16, 16, 4)

    def test_block1d_shape(self):
        blk = WaveletBlock1d(4, 8, n_scales=2, key=KEY)
        x = jax.random.normal(KEY, (32, 4))
        out = blk(x)
        assert out.shape == (32, 8)

    def test_block1d_pipe_sniffable(self):
        blk = WaveletBlock1d(4, 8, n_scales=2, key=KEY)
        wrapped = fx.block(blk)
        assert wrapped._in_channels == 4
        assert wrapped._out_channels == 8

    def test_block2d_shape(self):
        blk = WaveletBlock2d(4, 8, n_scales=2, key=KEY)
        x = jax.random.normal(KEY, (32, 32, 4))
        out = blk(x)
        assert out.shape == (32, 32, 8)

    def test_block2d_pipe_sniffable(self):
        blk = WaveletBlock2d(4, 8, n_scales=2, key=KEY)
        wrapped = fx.block(blk)
        assert wrapped._in_channels == 4
        assert wrapped._out_channels == 8

    def test_block3d_shape(self):
        blk = WaveletBlock3d(2, 4, n_scales=2, key=KEY)
        x = jax.random.normal(KEY, (16, 16, 16, 2))
        out = blk(x)
        assert out.shape == (16, 16, 16, 4)

    def test_wno1d_shape(self):
        model = fx.wno1d(in_channels=2, hidden_channels=8, depth=2, key=KEY)
        x = jax.random.normal(KEY, (32, 2))
        out = model(x)
        assert out.shape == (32, 2)

    def test_wno1d_jit(self):
        model = fx.wno1d(in_channels=2, hidden_channels=8, depth=2, key=KEY)
        x = jax.random.normal(KEY, (32, 2))
        out = eqx.filter_jit(model)(x)
        assert out.shape == (32, 2)

    def test_wno1d_gradients_finite(self):
        model = fx.wno1d(in_channels=1, hidden_channels=8, depth=2, key=KEY)
        x = jax.random.normal(KEY, (32, 1))

        def loss(m):
            return jnp.mean(m(x) ** 2)

        grads = eqx.filter_grad(loss)(model)
        leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves)

    def test_wno2d_shape(self):
        model = fx.wno2d(in_channels=1, hidden_channels=8, depth=2, key=KEY)
        x = jax.random.normal(KEY, (32, 32, 1))
        out = model(x)
        assert out.shape == (32, 32, 1)

    def test_wno3d_shape(self):
        model = fx.wno3d(in_channels=1, hidden_channels=4, depth=2, key=KEY)
        x = jax.random.normal(KEY, (16, 16, 16, 1))
        out = model(x)
        assert out.shape == (16, 16, 16, 1)

    def test_wno2d_jit(self):
        model = fx.wno2d(in_channels=1, hidden_channels=8, depth=2, key=KEY)
        x = jax.random.normal(KEY, (32, 32, 1))
        out = eqx.filter_jit(model)(x)
        assert out.shape == (32, 32, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Pipe integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestPipeIntegration:
    def test_ffno_block_pipe_chain(self):
        ks = jax.random.split(KEY, 3)
        b1 = fx.block(FactorizedSpectralBlock2d(4, 8, n_modes=4, key=ks[0]))
        b2 = fx.block(FactorizedSpectralBlock2d(8, 8, n_modes=4, key=ks[1]))
        b3 = fx.block(FactorizedSpectralBlock2d(8, 2, n_modes=4, key=ks[2]))
        pipe = b1 | b2 | b3
        x = jax.random.normal(KEY, (16, 16, 4))
        out = pipe(x)
        assert out.shape == (16, 16, 2)

    def test_ffno_pipe_film_forwarding(self):
        # t_emb kwarg forwarded to FiLM-enabled blocks via pipe
        ks = jax.random.split(KEY, 2)
        b1 = fx.block(FactorizedSpectralBlock2d(4, 4, n_modes=4, use_film=True, emb_dim=16, key=ks[0]))
        b2 = fx.block(FactorizedSpectralBlock2d(4, 4, n_modes=4, use_film=True, emb_dim=16, key=ks[1]))
        pipe = b1 | b2
        x = jax.random.normal(KEY, (16, 16, 4))
        emb = jax.random.normal(KEY, (16,))
        out_cond = pipe(x, t_emb=emb)
        out_uncond = pipe(x)
        assert out_cond.shape == (16, 16, 4)
        assert not jnp.allclose(out_cond, out_uncond)

    def test_heterogeneous_pipe_ffno_wno(self):
        # Mixed FFNO + WNO pipe: t_emb forwarded to FFNO, ignored by WNO
        ks = jax.random.split(KEY, 3)
        b1 = fx.block(FactorizedSpectralBlock2d(4, 8, n_modes=4, use_film=True, emb_dim=16, key=ks[0]))
        b2 = fx.block(WaveletBlock2d(8, 8, n_scales=2, key=ks[1]))
        b3 = fx.block(FactorizedSpectralBlock2d(8, 4, n_modes=4, use_film=True, emb_dim=16, key=ks[2]))
        pipe = b1 | b2 | b3
        x = jax.random.normal(KEY, (32, 32, 4))
        emb = jax.random.normal(KEY, (16,))
        out = pipe(x, t_emb=emb)
        assert out.shape == (32, 32, 4)

    def test_wno_block_pipe_chain(self):
        ks = jax.random.split(KEY, 3)
        b1 = fx.block(WaveletBlock1d(2, 8, n_scales=2, key=ks[0]))
        b2 = fx.block(WaveletBlock1d(8, 8, n_scales=2, key=ks[1]))
        b3 = fx.block(WaveletBlock1d(8, 1, n_scales=2, key=ks[2]))
        pipe = b1 | b2 | b3
        x = jax.random.normal(KEY, (32, 2))
        out = pipe(x)
        assert out.shape == (32, 1)

    def test_shape_mismatch_still_raised(self):
        ks = jax.random.split(KEY, 2)
        b1 = fx.block(FactorizedSpectralBlock2d(4, 8, n_modes=4, key=ks[0]))
        b2 = fx.block(FactorizedSpectralBlock2d(16, 4, n_modes=4, key=ks[1]))  # 16 ≠ 8
        with pytest.raises(fx.ShapeMismatchError):
            _ = b1 | b2

    def test_ffno2d_full_model_pipe_sniffable(self):
        model = fx.ffno2d(in_channels=3, hidden_channels=8, n_modes=4, n_layers=2, key=KEY)
        blk = fx.block(model)
        assert blk._in_channels == 3
        assert blk._out_channels == 3

    def test_wno2d_full_model_pipe_sniffable(self):
        model = fx.wno2d(in_channels=2, hidden_channels=8, depth=2, key=KEY)
        blk = fx.block(model)
        assert blk._in_channels == 2
        assert blk._out_channels == 2

    def test_pipe_jit(self):
        ks = jax.random.split(KEY, 2)
        b1 = fx.block(FactorizedSpectralBlock2d(4, 8, n_modes=4, key=ks[0]))
        b2 = fx.block(FactorizedSpectralBlock2d(8, 4, n_modes=4, key=ks[1]))
        pipe = b1 | b2
        x = jax.random.normal(KEY, (16, 16, 4))
        out = eqx.filter_jit(pipe)(x)
        assert out.shape == (16, 16, 4)
