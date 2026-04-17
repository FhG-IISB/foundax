"""Tests for Equinox PROSE model variants."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "repos", "jax_prose"))

from jax_prose.model_eqx import (
    PROSE1to1 as EqxPROSE1to1,
    PROSE2to1 as EqxPROSE2to1,
    PROSEODE2to1 as EqxPROSEODE2to1,
    PROSEPDE2to1 as EqxPROSEPDE2to1,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Small test configs
# ═══════════════════════════════════════════════════════════════════════════════

_X_NUM = 16
_DIM = 64
_FFN = 128
_NHEAD = 4
_DATA_DIM = 2
_PATCH = 4
_PATCH_OUT = 4
_N_WORDS = 32

# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPROSE1to1:
    @pytest.fixture
    def model(self):
        return EqxPROSE1to1(
            dim_emb=_DIM,
            dim_ffn=_FFN,
            n_head=_NHEAD,
            n_enc_layers=2,
            n_dec_layers=2,
            patch_num=_PATCH,
            patch_num_output=_PATCH_OUT,
            x_num=_X_NUM,
            max_output_dim=_DATA_DIM,
            norm_type="rms",
            norm_first=True,
            final_ln=True,
            time_embed_type="learnable",
            max_time_len=10,
            conv_dim=8,
            key=jax.random.PRNGKey(0),
        )

    def test_eqx_init(self, model):
        assert isinstance(model, EqxPROSE1to1)

    def test_forward_runs(self, model):
        data = jax.random.normal(
            jax.random.PRNGKey(1), (1, 2, _X_NUM, _X_NUM, _DATA_DIM)
        )
        tin = jax.random.uniform(jax.random.PRNGKey(2), (1, 2, 1))
        tout = jax.random.uniform(jax.random.PRNGKey(3), (1, 2, 1))
        out = model(data, tin, tout, deterministic=True)
        assert jnp.all(jnp.isfinite(out))

    def test_output_shape(self, model):
        data = jax.random.normal(
            jax.random.PRNGKey(4), (1, 2, _X_NUM, _X_NUM, _DATA_DIM)
        )
        tin = jax.random.uniform(jax.random.PRNGKey(5), (1, 2, 1))
        tout = jax.random.uniform(jax.random.PRNGKey(6), (1, 2, 1))
        out = model(data, tin, tout, deterministic=True)
        assert out.shape[0] == 1

    def test_gradients(self, model):
        data = jax.random.normal(
            jax.random.PRNGKey(7), (1, 2, _X_NUM, _X_NUM, _DATA_DIM)
        )
        tin = jax.random.uniform(jax.random.PRNGKey(8), (1, 2, 1))
        tout = jax.random.uniform(jax.random.PRNGKey(9), (1, 2, 1))

        @eqx.filter_grad
        def loss_fn(m):
            return jnp.mean(m(data, tin, tout, deterministic=True) ** 2)

        grads = loss_fn(model)
        leaves = [l for l in jax.tree_util.tree_leaves(grads) if eqx.is_array(l)]
        assert any(jnp.any(g != 0) for g in leaves)


class TestPROSE2to1:
    @pytest.fixture
    def model(self):
        return EqxPROSE2to1(
            n_words=_N_WORDS,
            x_num=_X_NUM,
            max_output_dim=_DATA_DIM,
            dim_emb=_DIM,
            dim_ffn=_FFN,
            n_head=_NHEAD,
            patch_num=_PATCH,
            patch_num_output=_PATCH_OUT,
            data_encoder_layers=1,
            symbol_encoder_layers=1,
            fusion_layers=1,
            data_decoder_layers=1,
            key=jax.random.PRNGKey(0),
        )

    def test_eqx_init(self, model):
        assert isinstance(model, EqxPROSE2to1)

    def test_forward_runs(self, model):
        data = jax.random.normal(
            jax.random.PRNGKey(1), (1, 2, _X_NUM, _X_NUM, _DATA_DIM)
        )
        tin = jax.random.uniform(jax.random.PRNGKey(2), (1, 2, 1))
        tout = jax.random.uniform(jax.random.PRNGKey(3), (1, 2, 1))
        sym = jax.random.randint(jax.random.PRNGKey(4), (1, 8), 0, _N_WORDS)
        sym_mask = jnp.zeros((1, 8), dtype=bool)
        out = model(data, tin, tout, sym, sym_mask)
        assert jnp.all(jnp.isfinite(out))

    def test_output_shape(self, model):
        data = jax.random.normal(
            jax.random.PRNGKey(5), (1, 2, _X_NUM, _X_NUM, _DATA_DIM)
        )
        tin = jax.random.uniform(jax.random.PRNGKey(6), (1, 2, 1))
        tout = jax.random.uniform(jax.random.PRNGKey(7), (1, 2, 1))
        sym = jax.random.randint(jax.random.PRNGKey(8), (1, 8), 0, _N_WORDS)
        sym_mask = jnp.zeros((1, 8), dtype=bool)
        out = model(data, tin, tout, sym, sym_mask)
        assert out.shape[0] == 1

    def test_gradients(self, model):
        data = jax.random.normal(
            jax.random.PRNGKey(9), (1, 2, _X_NUM, _X_NUM, _DATA_DIM)
        )
        tin = jax.random.uniform(jax.random.PRNGKey(10), (1, 2, 1))
        tout = jax.random.uniform(jax.random.PRNGKey(11), (1, 2, 1))
        sym = jax.random.randint(jax.random.PRNGKey(12), (1, 8), 0, _N_WORDS)
        sym_mask = jnp.zeros((1, 8), dtype=bool)

        @eqx.filter_grad
        def loss_fn(m):
            return jnp.mean(m(data, tin, tout, sym, sym_mask) ** 2)

        grads = loss_fn(model)
        leaves = [l for l in jax.tree_util.tree_leaves(grads) if eqx.is_array(l)]
        assert any(jnp.any(g != 0) for g in leaves)


class TestPROSEODE2to1:
    @pytest.fixture
    def model(self):
        return EqxPROSEODE2to1(
            n_words=_N_WORDS,
            pad_index=0,
            max_output_dimension=3,
            emb_dim=_DIM,
            n_text_enc_layers=1,
            n_data_enc_layers=1,
            n_data_dec_layers=1,
            n_fusion_layers=1,
            n_text_heads=_NHEAD,
            n_data_heads=_NHEAD,
            n_fusion_heads=_NHEAD,
            n_text_hidden_layers=1,
            n_data_hidden_layers=1,
            n_fusion_hidden_layers=1,
            key=jax.random.PRNGKey(0),
        )

    def test_eqx_init(self, model):
        assert isinstance(model, EqxPROSEODE2to1)

    def test_forward_runs(self, model):
        data = jax.random.normal(jax.random.PRNGKey(1), (10, 1, 4))
        data_len = jnp.array([10], dtype=jnp.int32)
        query = jnp.linspace(0, 1, 5)
        text = jax.random.randint(jax.random.PRNGKey(2), (8, 1), 0, _N_WORDS)
        text_len = jnp.array([8], dtype=jnp.int32)
        out = model(data, data_len, query, text, text_len)
        assert jnp.all(jnp.isfinite(out))

    def test_output_shape(self, model):
        data = jax.random.normal(jax.random.PRNGKey(3), (10, 1, 4))
        data_len = jnp.array([10], dtype=jnp.int32)
        query = jnp.linspace(0, 1, 5)
        text = jax.random.randint(jax.random.PRNGKey(4), (8, 1), 0, _N_WORDS)
        text_len = jnp.array([8], dtype=jnp.int32)
        out = model(data, data_len, query, text, text_len)
        assert out.ndim >= 1

    def test_gradients(self, model):
        data = jax.random.normal(jax.random.PRNGKey(5), (10, 1, 4))
        data_len = jnp.array([10], dtype=jnp.int32)
        query = jnp.linspace(0, 1, 5)
        text = jax.random.randint(jax.random.PRNGKey(6), (8, 1), 0, _N_WORDS)
        text_len = jnp.array([8], dtype=jnp.int32)

        @eqx.filter_grad
        def loss_fn(m):
            return jnp.mean(m(data, data_len, query, text, text_len) ** 2)

        grads = loss_fn(model)
        leaves = [l for l in jax.tree_util.tree_leaves(grads) if eqx.is_array(l)]
        assert any(jnp.any(g != 0) for g in leaves)


class TestPROSEPDE2to1:
    @pytest.fixture
    def model(self):
        return EqxPROSEPDE2to1(
            n_words=_N_WORDS,
            pad_index=0,
            max_output_dimension=1,
            emb_dim=_DIM,
            n_text_enc_layers=1,
            n_data_enc_layers=1,
            n_data_dec_layers=1,
            n_fusion_layers=1,
            n_text_heads=_NHEAD,
            n_data_heads=_NHEAD,
            n_fusion_heads=_NHEAD,
            n_text_hidden_layers=1,
            n_data_hidden_layers=1,
            n_fusion_hidden_layers=1,
            x_grid_size=4,
            normalization=True,
            key=jax.random.PRNGKey(0),
        )

    def test_eqx_init(self, model):
        assert isinstance(model, EqxPROSEPDE2to1)

    def test_forward_runs(self, model):
        # in_dim = 1 + max_output_dimension * x_patch_size = 1 + 1*1 = 2
        data = jax.random.normal(jax.random.PRNGKey(1), (10, 1, 2))
        data_len = jnp.array([10], dtype=jnp.int32)
        query = jnp.linspace(0, 1, 5)
        text = jax.random.randint(jax.random.PRNGKey(2), (8, 1), 0, _N_WORDS)
        text_len = jnp.array([8], dtype=jnp.int32)
        out = model(data, data_len, query, text, text_len)
        assert jnp.all(jnp.isfinite(out))

    def test_output_shape(self, model):
        data = jax.random.normal(jax.random.PRNGKey(3), (10, 1, 2))
        data_len = jnp.array([10], dtype=jnp.int32)
        query = jnp.linspace(0, 1, 5)
        text = jax.random.randint(jax.random.PRNGKey(4), (8, 1), 0, _N_WORDS)
        text_len = jnp.array([8], dtype=jnp.int32)
        out = model(data, data_len, query, text, text_len)
        assert out.ndim >= 1

    def test_gradients(self, model):
        data = jax.random.normal(jax.random.PRNGKey(5), (10, 1, 2))
        data_len = jnp.array([10], dtype=jnp.int32)
        query = jnp.linspace(0, 1, 5)
        text = jax.random.randint(jax.random.PRNGKey(6), (8, 1), 0, _N_WORDS)
        text_len = jnp.array([8], dtype=jnp.int32)

        @eqx.filter_grad
        def loss_fn(m):
            return jnp.mean(m(data, data_len, query, text, text_len) ** 2)

        grads = loss_fn(model)
        leaves = [l for l in jax.tree_util.tree_leaves(grads) if eqx.is_array(l)]
        assert any(jnp.any(g != 0) for g in leaves)
