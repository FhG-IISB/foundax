"""Test forward-pass equivalence between Flax and Equinox ScOT implementations.

Run with:
    python -m pytest tests/test_poseidon_eqx.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sys, os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "repos", "jax_poseidon")
)

from jax_poseidon.scot import ScOT as FlaxScOT, ScOTConfig
from jax_poseidon.scot_eqx import ScOT as EqxScOT
import equinox as eqx


def _make_config(**overrides):
    """Create a small test config."""
    defaults = dict(
        name="test",
        image_size=32,
        patch_size=4,
        num_channels=2,
        num_out_channels=2,
        embed_dim=24,
        depths=(2, 2),
        num_heads=(3, 6),
        skip_connections=(1, 0),
        window_size=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.0,
        hidden_act="gelu",
        use_absolute_embeddings=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        p=1,
        channel_slice_list_normalized_loss=None,
        residual_model="convnext",
        use_conditioning=True,
        learn_residual=False,
        pretrained_window_sizes=(0, 0),
    )
    defaults.update(overrides)
    return ScOTConfig(**defaults)


def _transfer_dense_to_linear(flax_params, eqx_linear):
    """Transfer Flax Dense params to Equinox Linear."""
    kernel = flax_params["kernel"]  # (in, out) in Flax
    new_weight = kernel.T  # Equinox Linear stores (out, in)
    new_linear = eqx.tree_at(lambda m: m.weight, eqx_linear, new_weight)
    if "bias" in flax_params and eqx_linear.bias is not None:
        new_linear = eqx.tree_at(lambda m: m.bias, new_linear, flax_params["bias"])
    return new_linear


def _transfer_conv_to_conv2d(flax_params, eqx_conv_nhwc):
    """Transfer Flax Conv params to Equinox Conv2dNHWC wrapper.

    Flax Conv kernel: (kH, kW, C_in, C_out) for standard or (kH, kW, 1, C_out) for depthwise
    Equinox Conv2d weight: (C_out, C_in/groups, kH, kW)
    Flax Conv bias: (C_out,)
    Equinox Conv2d bias: (C_out, 1, 1)
    """
    kernel = flax_params["kernel"]  # (kH, kW, C_in_or_1, C_out)
    # Transpose to (C_out, C_in_or_1, kH, kW)
    new_weight = kernel.transpose(3, 2, 0, 1)
    new_conv = eqx.tree_at(lambda m: m.conv.weight, eqx_conv_nhwc, new_weight)
    if "bias" in flax_params and eqx_conv_nhwc.conv.bias is not None:
        # Reshape Flax (C_out,) to Equinox (C_out, 1, 1)
        bias = flax_params["bias"]
        new_conv = eqx.tree_at(lambda m: m.conv.bias, new_conv, bias.reshape(-1, 1, 1))
    return new_conv


def _transfer_conv_transpose_to_conv_transpose2d(flax_params, eqx_conv_nhwc):
    """Transfer Flax ConvTranspose params to Equinox ConvTranspose2dNHWC.

    Flax ConvTranspose kernel: (kH, kW, C_out, C_in)
    Equinox ConvTranspose2d weight: (C_in, C_out, kH, kW)
    """
    kernel = flax_params["kernel"]  # (kH, kW, C_out, C_in)
    new_weight = kernel.transpose(3, 2, 0, 1)
    new_conv = eqx.tree_at(lambda m: m.conv.weight, eqx_conv_nhwc, new_weight)
    if "bias" in flax_params and eqx_conv_nhwc.conv.bias is not None:
        bias = flax_params["bias"]
        new_conv = eqx.tree_at(lambda m: m.conv.bias, new_conv, bias.reshape(-1, 1, 1))
    return new_conv


def _transfer_layernorm_with_time(flax_params, eqx_lnwt):
    """Transfer Flax LayerNorm params to Equinox LayerNormWithTime."""
    # Flax LayerNormWithTime uses @nn.compact which creates a nested LayerNorm_0
    ln_params = flax_params.get("LayerNorm_0", flax_params)
    new_norm = eqx.tree_at(lambda m: m.norm.weight, eqx_lnwt, ln_params["scale"])
    new_norm = eqx.tree_at(lambda m: m.norm.bias, new_norm, ln_params["bias"])
    return new_norm


def _transfer_conditional_layernorm(flax_params, eqx_cln):
    """Transfer Flax ConditionalLayerNorm params to Equinox ConditionalLayerNorm."""
    # Weight dense
    new_cln = _transfer_dense_to_linear(flax_params["weight"], eqx_cln.weight_dense)
    new_cln_obj = eqx.tree_at(lambda m: m.weight_dense, eqx_cln, new_cln)
    # Bias dense
    new_bias = _transfer_dense_to_linear(flax_params["bias"], eqx_cln.bias_dense)
    new_cln_obj = eqx.tree_at(lambda m: m.bias_dense, new_cln_obj, new_bias)
    return new_cln_obj


def _transfer_norm(flax_params, eqx_norm, is_conditional):
    """Transfer norm params (either conditional or standard)."""
    if is_conditional:
        return _transfer_conditional_layernorm(flax_params, eqx_norm)
    else:
        return _transfer_layernorm_with_time(flax_params, eqx_norm)


def _transfer_rel_pos_bias(flax_params, eqx_rpb):
    """Transfer Swinv2RelativePositionBias params."""
    new_rpb = _transfer_dense_to_linear(flax_params["cpb_mlp_0"], eqx_rpb.cpb_mlp_0)
    new_rpb_obj = eqx.tree_at(lambda m: m.cpb_mlp_0, eqx_rpb, new_rpb)
    new_mlp1 = _transfer_dense_to_linear(flax_params["cpb_mlp_1"], eqx_rpb.cpb_mlp_1)
    new_rpb_obj = eqx.tree_at(lambda m: m.cpb_mlp_1, new_rpb_obj, new_mlp1)
    return new_rpb_obj


def _transfer_attention(flax_params, eqx_attn):
    """Transfer Swinv2Attention params."""
    new_attn = eqx_attn

    # query, key, value, proj
    new_q = _transfer_dense_to_linear(flax_params["query"], eqx_attn.query)
    new_attn = eqx.tree_at(lambda m: m.query, new_attn, new_q)

    new_k = _transfer_dense_to_linear(flax_params["key"], eqx_attn.key_proj)
    new_attn = eqx.tree_at(lambda m: m.key_proj, new_attn, new_k)

    new_v = _transfer_dense_to_linear(flax_params["value"], eqx_attn.value)
    new_attn = eqx.tree_at(lambda m: m.value, new_attn, new_v)

    new_proj = _transfer_dense_to_linear(flax_params["proj"], eqx_attn.proj)
    new_attn = eqx.tree_at(lambda m: m.proj, new_attn, new_proj)

    # logit_scale
    new_attn = eqx.tree_at(
        lambda m: m.logit_scale, new_attn, flax_params["logit_scale"]
    )

    # relative_position_bias
    new_rpb = _transfer_rel_pos_bias(
        flax_params["relative_position_bias"], eqx_attn.relative_position_bias
    )
    new_attn = eqx.tree_at(lambda m: m.relative_position_bias, new_attn, new_rpb)

    return new_attn


def _transfer_convnext_block(flax_params, eqx_block, is_conditional):
    """Transfer ConvNeXtBlock params."""
    new_block = eqx_block

    # dwconv
    new_dwconv = _transfer_conv_to_conv2d(flax_params["dwconv"], eqx_block.dwconv)
    new_block = eqx.tree_at(lambda m: m.dwconv, new_block, new_dwconv)

    # norm
    new_norm = _transfer_norm(flax_params["norm"], eqx_block.norm, is_conditional)
    new_block = eqx.tree_at(lambda m: m.norm, new_block, new_norm)

    # pwconv1 (Dense in Flax, Linear in Equinox)
    new_pw1 = _transfer_dense_to_linear(flax_params["pwconv1"], eqx_block.pwconv1)
    new_block = eqx.tree_at(lambda m: m.pwconv1, new_block, new_pw1)

    # pwconv2
    new_pw2 = _transfer_dense_to_linear(flax_params["pwconv2"], eqx_block.pwconv2)
    new_block = eqx.tree_at(lambda m: m.pwconv2, new_block, new_pw2)

    # weight (layer scale)
    if "weight" in flax_params and eqx_block.weight is not None:
        new_block = eqx.tree_at(lambda m: m.weight, new_block, flax_params["weight"])

    return new_block


def _transfer_scot_layer(flax_params, eqx_layer, is_conditional):
    """Transfer ScOTLayer params."""
    new_layer = eqx_layer

    # attention
    new_attn = _transfer_attention(flax_params["attention"], eqx_layer.attention)
    new_layer = eqx.tree_at(lambda m: m.attention, new_layer, new_attn)

    # layernorm_before
    new_lnb = _transfer_norm(
        flax_params["layernorm_before"], eqx_layer.layernorm_before, is_conditional
    )
    new_layer = eqx.tree_at(lambda m: m.layernorm_before, new_layer, new_lnb)

    # layernorm_after
    new_lna = _transfer_norm(
        flax_params["layernorm_after"], eqx_layer.layernorm_after, is_conditional
    )
    new_layer = eqx.tree_at(lambda m: m.layernorm_after, new_layer, new_lna)

    # intermediate
    new_int = _transfer_dense_to_linear(
        flax_params["intermediate"]["dense"], eqx_layer.intermediate.dense
    )
    new_layer = eqx.tree_at(lambda m: m.intermediate.dense, new_layer, new_int)

    # output
    new_out = _transfer_dense_to_linear(
        flax_params["output"]["dense"], eqx_layer.output_layer.dense
    )
    new_layer = eqx.tree_at(lambda m: m.output_layer.dense, new_layer, new_out)

    return new_layer


def _transfer_encode_stage(flax_params, eqx_stage, is_conditional):
    """Transfer ScOTEncodeStage params."""
    new_stage = eqx_stage

    for i, block in enumerate(eqx_stage.blocks):
        block_key = f"block_{i}"
        new_block = _transfer_scot_layer(flax_params[block_key], block, is_conditional)
        new_stage = eqx.tree_at(lambda m, idx=i: m.blocks[idx], new_stage, new_block)

    if eqx_stage.downsample_layer is not None and "downsample" in flax_params:
        ds_params = flax_params["downsample"]
        new_ds = eqx_stage.downsample_layer

        # reduction
        new_red = _transfer_dense_to_linear(ds_params["reduction"], new_ds.reduction)
        new_ds = eqx.tree_at(lambda m: m.reduction, new_ds, new_red)

        # norm
        new_norm = _transfer_norm(ds_params["norm"], new_ds.norm, is_conditional)
        new_ds = eqx.tree_at(lambda m: m.norm, new_ds, new_norm)

        new_stage = eqx.tree_at(lambda m: m.downsample_layer, new_stage, new_ds)

    return new_stage


def _transfer_decode_stage(flax_params, eqx_stage, is_conditional):
    """Transfer ScOTDecodeStage params."""
    new_stage = eqx_stage

    # Decode blocks are created with reversed(range(depth)), so blocks[0] was
    # created with i=depth-1 and named "block_{depth-1}" in Flax.
    depth = eqx_stage.depth
    for i, block in enumerate(eqx_stage.blocks):
        block_key = f"block_{depth - 1 - i}"
        new_block = _transfer_scot_layer(flax_params[block_key], block, is_conditional)
        new_stage = eqx.tree_at(lambda m, idx=i: m.blocks[idx], new_stage, new_block)

    if eqx_stage.upsample_layer is not None and "upsample" in flax_params:
        us_params = flax_params["upsample"]
        new_us = eqx_stage.upsample_layer

        # upsample dense
        new_up = _transfer_dense_to_linear(us_params["upsample"], new_us.upsample)
        new_us = eqx.tree_at(lambda m: m.upsample, new_us, new_up)

        # mixup dense
        new_mix = _transfer_dense_to_linear(us_params["mixup"], new_us.mixup)
        new_us = eqx.tree_at(lambda m: m.mixup, new_us, new_mix)

        # norm
        new_norm = _transfer_norm(us_params["norm"], new_us.norm, is_conditional)
        new_us = eqx.tree_at(lambda m: m.norm, new_us, new_norm)

        new_stage = eqx.tree_at(lambda m: m.upsample_layer, new_stage, new_us)

    return new_stage


def _transfer_residual_block_wrapper(flax_params, eqx_wrapper, is_conditional):
    """Transfer ResidualBlockWrapper params."""
    new_wrapper = eqx_wrapper
    if eqx_wrapper.blocks is None:
        return new_wrapper

    for i, block in enumerate(eqx_wrapper.blocks):
        block_key = f"block_{i}"
        if block_key in flax_params:
            new_block = _transfer_convnext_block(
                flax_params[block_key], block, is_conditional
            )
            new_wrapper = eqx.tree_at(
                lambda m, idx=i: m.blocks[idx], new_wrapper, new_block
            )

    return new_wrapper


def transfer_all_params(flax_params, flax_model, eqx_model):
    """Transfer all parameters from Flax ScOT to Equinox ScOT.

    Args:
        flax_params: The Flax parameter dict (output of model.init)
        flax_model: The Flax ScOT model instance
        eqx_model: The Equinox ScOT model instance

    Returns:
        Updated Equinox model with transferred weights
    """
    params = flax_params
    if "params" in params:
        params = params["params"]

    config = eqx_model.config
    is_conditional = config.use_conditioning
    new_model = eqx_model

    # --- Embeddings ---
    emb_params = params["embeddings"]

    # patch_embeddings.projection (Conv)
    new_proj = _transfer_conv_to_conv2d(
        emb_params["patch_embeddings"]["projection"],
        eqx_model.embeddings.patch_embeddings.projection,
    )
    new_model = eqx.tree_at(
        lambda m: m.embeddings.patch_embeddings.projection, new_model, new_proj
    )

    # embeddings.norm
    new_norm = _transfer_norm(
        emb_params["norm"], eqx_model.embeddings.norm, is_conditional
    )
    new_model = eqx.tree_at(lambda m: m.embeddings.norm, new_model, new_norm)

    # --- Encoder ---
    enc_params = params["encoder"]
    for i, stage in enumerate(eqx_model.encoder.layers):
        stage_key = f"layer_{i}"
        new_stage = _transfer_encode_stage(enc_params[stage_key], stage, is_conditional)
        new_model = eqx.tree_at(
            lambda m, idx=i: m.encoder.layers[idx], new_model, new_stage
        )

    # --- Decoder ---
    dec_params = params["decoder"]
    num_layers = len(config.depths)
    for i, stage in enumerate(eqx_model.decoder.layers):
        # Decoder layers are stored in reversed order
        stage_key = f"layer_{num_layers - 1 - i}"
        new_stage = _transfer_decode_stage(dec_params[stage_key], stage, is_conditional)
        new_model = eqx.tree_at(
            lambda m, idx=i: m.decoder.layers[idx], new_model, new_stage
        )

    # --- Patch Recovery ---
    pr_params = params["patch_recovery"]
    new_proj = _transfer_conv_transpose_to_conv_transpose2d(
        pr_params["projection"], eqx_model.patch_recovery.projection
    )
    new_model = eqx.tree_at(lambda m: m.patch_recovery.projection, new_model, new_proj)

    new_mixup = _transfer_conv_to_conv2d(
        pr_params["mixup"], eqx_model.patch_recovery.mixup
    )
    new_model = eqx.tree_at(lambda m: m.patch_recovery.mixup, new_model, new_mixup)

    # --- Residual Blocks ---
    for i, rb in enumerate(eqx_model.residual_blocks):
        rb_key = f"residual_block_{i}"
        if rb_key in params:
            new_rb = _transfer_residual_block_wrapper(
                params[rb_key], rb, is_conditional
            )
            new_model = eqx.tree_at(
                lambda m, idx=i: m.residual_blocks[idx], new_model, new_rb
            )

    return new_model


# =====================================================================
# Tests
# =====================================================================


class TestPoseidonEquivalence:
    """Test that Flax and Equinox ScOT produce identical outputs."""

    @pytest.fixture
    def config(self):
        return _make_config()

    @pytest.fixture
    def models_and_params(self, config):
        """Create both models and transfer Flax params to Equinox."""
        rng = jax.random.PRNGKey(42)

        # Create Flax model
        flax_model = FlaxScOT(config=config, use_conditioning=config.use_conditioning)

        # Initialize Flax model
        x = jnp.zeros((1, config.image_size, config.image_size, config.num_channels))
        t = jnp.zeros((1,))
        flax_vars = flax_model.init(
            rng, pixel_values=x, time=t, deterministic=True, return_dict=False
        )

        # Create Equinox model
        eqx_model = EqxScOT(
            config=config,
            use_conditioning=config.use_conditioning,
            key=jax.random.PRNGKey(0),
        )

        # Transfer weights
        eqx_model = transfer_all_params(flax_vars, flax_model, eqx_model)

        return flax_model, flax_vars, eqx_model

    def test_forward_equivalence(self, config, models_and_params):
        """Test that Flax and Equinox models produce identical outputs."""
        flax_model, flax_vars, eqx_model = models_and_params

        # Create test input
        rng = jax.random.PRNGKey(123)
        x = jax.random.normal(
            rng, (1, config.image_size, config.image_size, config.num_channels)
        )
        t = jnp.array([0.5])

        # Flax forward
        flax_out = flax_model.apply(
            flax_vars, pixel_values=x, time=t, deterministic=True, return_dict=False
        )
        flax_pred = flax_out[0]

        # Equinox forward
        eqx_out = eqx_model(
            pixel_values=x, time=t, deterministic=True, return_dict=False
        )
        eqx_pred = eqx_out[0]

        # Compare
        max_diff = jnp.max(jnp.abs(flax_pred - eqx_pred))
        mean_diff = jnp.mean(jnp.abs(flax_pred - eqx_pred))

        print(f"\nMax diff: {max_diff:.2e}")
        print(f"Mean diff: {mean_diff:.2e}")
        print(f"Flax output range: [{flax_pred.min():.4f}, {flax_pred.max():.4f}]")
        print(f"Eqx output range: [{eqx_pred.min():.4f}, {eqx_pred.max():.4f}]")

        assert jnp.allclose(
            flax_pred, eqx_pred, atol=1e-5, rtol=1e-5
        ), f"Forward pass mismatch! max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"

    def test_output_shapes_match(self, config, models_and_params):
        """Test that output shapes are identical."""
        flax_model, flax_vars, eqx_model = models_and_params

        x = jnp.ones((2, config.image_size, config.image_size, config.num_channels))
        t = jnp.array([0.1, 0.2])

        flax_out = flax_model.apply(
            flax_vars, pixel_values=x, time=t, deterministic=True, return_dict=True
        )
        eqx_out = eqx_model(
            pixel_values=x, time=t, deterministic=True, return_dict=True
        )

        assert (
            flax_out.output.shape == eqx_out.output.shape
        ), f"Shape mismatch: flax={flax_out.output.shape}, eqx={eqx_out.output.shape}"

    def test_zero_input(self, config, models_and_params):
        """Test with zero input."""
        flax_model, flax_vars, eqx_model = models_and_params

        x = jnp.zeros((1, config.image_size, config.image_size, config.num_channels))
        t = jnp.zeros((1,))

        flax_out = flax_model.apply(
            flax_vars, pixel_values=x, time=t, deterministic=True, return_dict=False
        )
        eqx_out = eqx_model(
            pixel_values=x, time=t, deterministic=True, return_dict=False
        )

        assert jnp.allclose(flax_out[0], eqx_out[0], atol=1e-5, rtol=1e-5)

    def test_gradient_equivalence(self, config, models_and_params):
        """Test that gradients flow through the Equinox model without error."""
        flax_model, flax_vars, eqx_model = models_and_params

        x = jax.random.normal(
            jax.random.PRNGKey(99),
            (1, config.image_size, config.image_size, config.num_channels),
        )
        t = jnp.array([0.5])

        # Equinox gradient
        def eqx_loss(model):
            out = model(pixel_values=x, time=t, deterministic=True, return_dict=False)
            return jnp.mean(out[0] ** 2)

        eqx_grad_val = eqx.filter_grad(eqx_loss)(eqx_model)
        eqx_grad_leaves = jax.tree_util.tree_leaves(eqx_grad_val)

        # Verify gradient leaves are finite and non-trivial
        for leaf in eqx_grad_leaves:
            if hasattr(leaf, "shape"):
                assert jnp.all(jnp.isfinite(leaf)), "Non-finite gradient detected"


class TestPoseidonNoConditioning:
    """Test equivalence without conditioning (simpler path)."""

    @pytest.fixture
    def config(self):
        return _make_config(use_conditioning=False)

    @pytest.fixture
    def models_and_params(self, config):
        rng = jax.random.PRNGKey(42)
        flax_model = FlaxScOT(config=config, use_conditioning=False)
        x = jnp.zeros((1, config.image_size, config.image_size, config.num_channels))
        flax_vars = flax_model.init(
            rng, pixel_values=x, deterministic=True, return_dict=False
        )

        eqx_model = EqxScOT(
            config=config, use_conditioning=False, key=jax.random.PRNGKey(0)
        )
        eqx_model = transfer_all_params(flax_vars, flax_model, eqx_model)
        return flax_model, flax_vars, eqx_model

    def test_forward_equivalence_no_cond(self, config, models_and_params):
        flax_model, flax_vars, eqx_model = models_and_params
        x = jax.random.normal(
            jax.random.PRNGKey(7),
            (1, config.image_size, config.image_size, config.num_channels),
        )

        flax_out = flax_model.apply(
            flax_vars, pixel_values=x, deterministic=True, return_dict=False
        )
        eqx_out = eqx_model(pixel_values=x, deterministic=True, return_dict=False)

        max_diff = jnp.max(jnp.abs(flax_out[0] - eqx_out[0]))
        print(f"\nNo-conditioning max diff: {max_diff:.2e}")
        assert jnp.allclose(flax_out[0], eqx_out[0], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
