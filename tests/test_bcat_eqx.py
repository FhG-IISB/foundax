"""Tests verifying Equinox BCAT matches Flax BCAT forward pass."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import pytest

from jax_bcat.model import BCAT as FlaxBCAT
from jax_bcat.model_eqx import BCAT as EqxBCAT, RMSNorm as EqxRMSNorm


# ---------------------------------------------------------------------------
# Small test config
# ---------------------------------------------------------------------------
_CFG = dict(
    n_layer=2,
    dim_emb=64,
    dim_ffn=128,
    n_head=4,
    norm_first=True,
    norm_type="rms",
    activation="swiglu",
    qk_norm=True,
    x_num=16,
    max_output_dim=2,
    patch_num=4,
    patch_num_output=4,
    conv_dim=8,
    time_embed="learnable",
    max_time_len=10,
    max_data_len=10,
    deep=False,
)


# ---------------------------------------------------------------------------
# Weight transfer helpers
# ---------------------------------------------------------------------------


def _transfer_dense_to_linear(flax_params, eqx_linear):
    new = eqx.tree_at(lambda m: m.weight, eqx_linear, flax_params["kernel"].T)
    if "bias" in flax_params and eqx_linear.bias is not None:
        new = eqx.tree_at(lambda m: m.bias, new, flax_params["bias"])
    return new


def _transfer_conv_to_conv2d(flax_params, eqx_conv):
    """Flax Conv kernel (kH,kW,Cin,Cout) → eqx Conv2d weight (Cout,Cin,kH,kW)."""
    new = eqx.tree_at(
        lambda m: m.weight, eqx_conv, flax_params["kernel"].transpose(3, 2, 0, 1)
    )
    if "bias" in flax_params and eqx_conv.bias is not None:
        new = eqx.tree_at(lambda m: m.bias, new, flax_params["bias"].reshape(-1, 1, 1))
    return new


def _transfer_conv_transpose(flax_params, eqx_conv, transpose_kernel=False):
    """Flax ConvTranspose kernel → eqx ConvTranspose2d weight.

    With transpose_kernel=True, Flax stores a forward kernel that it internally
    flips; eqx does *not* flip, so we must flip spatial dims ourselves.
    """
    kernel = flax_params["kernel"]  # (kH, kW, Cout, Cin)
    if transpose_kernel:
        kernel = kernel[::-1, ::-1, :, :]
    new_weight = kernel.transpose(2, 3, 0, 1)  # (Cout, Cin, kH, kW)
    new = eqx.tree_at(lambda m: m.weight, eqx_conv, new_weight)
    if "bias" in flax_params and eqx_conv.bias is not None:
        new = eqx.tree_at(lambda m: m.bias, new, flax_params["bias"].reshape(-1, 1, 1))
    return new


def _transfer_rmsnorm(flax_params, eqx_norm):
    return eqx.tree_at(lambda m: m.weight, eqx_norm, flax_params["scale"])


def _transfer_layernorm(flax_params, eqx_norm):
    new = eqx.tree_at(lambda m: m.weight, eqx_norm, flax_params["scale"])
    if "bias" in flax_params:
        new = eqx.tree_at(lambda m: m.bias, new, flax_params["bias"])
    return new


def _transfer_mha(flax_params, eqx_mha):
    """Transfer MultiheadAttention weights."""
    new = eqx_mha
    for proj in ("linear_q", "linear_k", "linear_v", "out_proj"):
        new = eqx.tree_at(
            lambda m, p=proj: getattr(m, p),
            new,
            _transfer_dense_to_linear(flax_params[proj], getattr(eqx_mha, proj)),
        )
    if eqx_mha.qk_norm and "q_norm" in flax_params:
        new = eqx.tree_at(
            lambda m: m.q_norm,
            new,
            _transfer_layernorm(flax_params["q_norm"], eqx_mha.q_norm),
        )
        new = eqx.tree_at(
            lambda m: m.k_norm,
            new,
            _transfer_layernorm(flax_params["k_norm"], eqx_mha.k_norm),
        )
    return new


def _transfer_ffn(flax_params, eqx_ffn):
    new = eqx_ffn
    new = eqx.tree_at(
        lambda m: m.fc1,
        new,
        _transfer_dense_to_linear(flax_params["fc1"], eqx_ffn.fc1),
    )
    new = eqx.tree_at(
        lambda m: m.fc2,
        new,
        _transfer_dense_to_linear(flax_params["fc2"], eqx_ffn.fc2),
    )
    if eqx_ffn.fc_gate is not None and "fc_gate" in flax_params:
        new = eqx.tree_at(
            lambda m: m.fc_gate,
            new,
            _transfer_dense_to_linear(flax_params["fc_gate"], eqx_ffn.fc_gate),
        )
    return new


def _transfer_norm(flax_params, eqx_norm):
    if isinstance(eqx_norm, EqxRMSNorm):
        return _transfer_rmsnorm(flax_params, eqx_norm)
    return _transfer_layernorm(flax_params, eqx_norm)


def _transfer_encoder_layer(flax_params, eqx_layer):
    new = eqx_layer
    new = eqx.tree_at(
        lambda m: m.self_attn,
        new,
        _transfer_mha(flax_params["self_attn"], eqx_layer.self_attn),
    )
    new = eqx.tree_at(
        lambda m: m.ffn,
        new,
        _transfer_ffn(flax_params["ffn"], eqx_layer.ffn),
    )
    new = eqx.tree_at(
        lambda m: m.norm1,
        new,
        _transfer_norm(flax_params["norm1"], eqx_layer.norm1),
    )
    new = eqx.tree_at(
        lambda m: m.norm2,
        new,
        _transfer_norm(flax_params["norm2"], eqx_layer.norm2),
    )
    return new


def transfer_weights(flax_params, eqx_model):
    """Transfer all weights from Flax BCAT params to Equinox BCAT model."""
    p = flax_params["params"]
    new = eqx_model

    # --- Embedder ---
    ep = p["embedder"]

    # in_proj
    new = eqx.tree_at(
        lambda m: m.embedder.in_proj,
        new,
        _transfer_conv_to_conv2d(ep["in_proj"], eqx_model.embedder.in_proj),
    )
    # conv_proj
    new = eqx.tree_at(
        lambda m: m.embedder.conv_proj,
        new,
        _transfer_conv_to_conv2d(ep["conv_proj"], eqx_model.embedder.conv_proj),
    )

    # Time embeddings
    if "time_embeddings" in ep:
        new = eqx.tree_at(
            lambda m: m.embedder.time_embeddings,
            new,
            ep["time_embeddings"],
        )
    else:
        new = eqx.tree_at(
            lambda m: m.embedder.time_proj_0,
            new,
            _transfer_dense_to_linear(
                ep["time_proj_0"], eqx_model.embedder.time_proj_0
            ),
        )
        new = eqx.tree_at(
            lambda m: m.embedder.time_proj_1,
            new,
            _transfer_dense_to_linear(
                ep["time_proj_1"], eqx_model.embedder.time_proj_1
            ),
        )

    # Patch position embeddings
    new = eqx.tree_at(
        lambda m: m.embedder.patch_position_embeddings,
        new,
        ep["patch_position_embeddings"],
    )

    # post_deconv (ConvTranspose with transpose_kernel=True)
    new = eqx.tree_at(
        lambda m: m.embedder.post_deconv,
        new,
        _transfer_conv_transpose(
            ep["post_deconv"], eqx_model.embedder.post_deconv, transpose_kernel=True
        ),
    )
    # post_conv
    new = eqx.tree_at(
        lambda m: m.embedder.post_conv,
        new,
        _transfer_conv_to_conv2d(ep["post_conv"], eqx_model.embedder.post_conv),
    )
    # head
    new = eqx.tree_at(
        lambda m: m.embedder.head,
        new,
        _transfer_conv_to_conv2d(ep["head"], eqx_model.embedder.head),
    )

    # --- Transformer ---
    tp = p["transformer"]
    n_layer = len(eqx_model.transformer.layers)
    for i in range(n_layer):
        lp = tp[f"layers_{i}"]
        new = eqx.tree_at(
            lambda m, idx=i: m.transformer.layers[idx],
            new,
            _transfer_encoder_layer(lp, eqx_model.transformer.layers[i]),
        )

    # Final norm
    if eqx_model.transformer.norm is not None and "norm" in tp:
        new = eqx.tree_at(
            lambda m: m.transformer.norm,
            new,
            _transfer_norm(tp["norm"], eqx_model.transformer.norm),
        )

    return new


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture
def small_flax_model():
    return FlaxBCAT(**_CFG)


@pytest.fixture
def small_eqx_model():
    return EqxBCAT(**_CFG, data_dim=2, key=jax.random.PRNGKey(0))


@pytest.fixture
def flax_params(small_flax_model, rng):
    bs, t_total, x_num, data_dim = 1, 5, _CFG["x_num"], 2
    data = jnp.ones((bs, t_total, x_num, x_num, data_dim))
    times = jnp.ones((bs, t_total, 1))
    variables = small_flax_model.init(rng, data, times, input_len=3)
    return variables


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBCATEqx:
    """Verify Equinox BCAT matches Flax BCAT output."""

    def test_flax_init(self, small_flax_model, flax_params):
        """Flax model initialises without error."""
        assert "params" in flax_params

    def test_eqx_init(self, small_eqx_model):
        """Equinox model initialises without error."""
        assert isinstance(small_eqx_model, EqxBCAT)

    def test_weight_transfer(self, small_eqx_model, flax_params):
        """Weights transfer from Flax → Equinox without error."""
        eqx_model = transfer_weights(flax_params, small_eqx_model)
        assert isinstance(eqx_model, EqxBCAT)

    def test_forward_equivalence(
        self, small_flax_model, small_eqx_model, flax_params, rng
    ):
        """Forward pass outputs match between Flax and Equinox."""
        bs, t_total, x_num, data_dim = 1, 5, _CFG["x_num"], 2
        input_len = 3

        data = jax.random.normal(rng, (bs, t_total, x_num, x_num, data_dim))
        times = jax.random.uniform(jax.random.PRNGKey(1), (bs, t_total, 1))

        flax_out = small_flax_model.apply(flax_params, data, times, input_len=input_len)

        eqx_model = transfer_weights(flax_params, small_eqx_model)
        eqx_out = eqx_model(data, times, input_len=input_len)

        diff = jnp.max(jnp.abs(flax_out - eqx_out))
        print(f"\nMax absolute difference: {diff:.2e}")
        assert diff < 1e-4, f"Outputs differ by {diff:.2e}"

    def test_forward_equivalence_continuous_time(self, rng):
        """Forward pass with continuous time embeddings."""
        cfg = {**_CFG, "time_embed": "continuous"}
        flax_model = FlaxBCAT(**cfg)
        eqx_model = EqxBCAT(**cfg, data_dim=2, key=jax.random.PRNGKey(0))

        bs, t_total, x_num, data_dim = 1, 5, cfg["x_num"], 2
        input_len = 3

        data = jnp.ones((bs, t_total, x_num, x_num, data_dim))
        times = jnp.ones((bs, t_total, 1))
        flax_params = flax_model.init(rng, data, times, input_len=input_len)

        data = jax.random.normal(rng, (bs, t_total, x_num, x_num, data_dim))
        times = jax.random.uniform(jax.random.PRNGKey(1), (bs, t_total, 1))

        flax_out = flax_model.apply(flax_params, data, times, input_len=input_len)

        eqx_model = transfer_weights(flax_params, eqx_model)
        eqx_out = eqx_model(data, times, input_len=input_len)

        diff = jnp.max(jnp.abs(flax_out - eqx_out))
        print(f"\nMax absolute difference (continuous time): {diff:.2e}")
        assert diff < 1e-4, f"Outputs differ by {diff:.2e}"
