"""Test forward-pass equivalence between Flax and Equinox DPOTNet (2-D).

Run with:
    python -m pytest tests/test_dpot_eqx.py -v -s
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "repos", "jax_dpot"))

import jax
import jax.numpy as jnp
import pytest
import equinox as eqx

from jax_dpot.model import DPOTNet as FlaxDPOTNet
from jax_dpot.model_eqx import DPOTNet as EqxDPOTNet


# ---------------------------------------------------------------------------
# Weight transfer utilities
# ---------------------------------------------------------------------------


def _transfer_dense_to_linear(flax_params, eqx_linear):
    """Transfer Flax Dense (kernel, bias) → Equinox Linear (weight, bias)."""
    new = eqx.tree_at(lambda m: m.weight, eqx_linear, flax_params["kernel"].T)
    if "bias" in flax_params and eqx_linear.bias is not None:
        new = eqx.tree_at(lambda m: m.bias, new, flax_params["bias"])
    return new


def _transfer_conv_to_conv2d(flax_params, eqx_conv):
    """Flax Conv kernel (kH,kW,Cin,Cout) → Equinox Conv2d weight (Cout,Cin,kH,kW)."""
    new = eqx.tree_at(
        lambda m: m.weight, eqx_conv, flax_params["kernel"].transpose(3, 2, 0, 1)
    )
    if "bias" in flax_params and eqx_conv.bias is not None:
        new = eqx.tree_at(lambda m: m.bias, new, flax_params["bias"].reshape(-1, 1, 1))
    return new


def _transfer_conv_transpose_to_conv_transpose2d(
    flax_params, eqx_conv, transpose_kernel=False
):
    """Flax ConvTranspose kernel → Equinox ConvTranspose2d weight.

    Flax ConvTranspose kernel layout: (kH, kW, C_out, C_in).
    When ``transpose_kernel=True`` Flax stores a *forward* conv kernel and
    internally flips it.  Equinox ConvTranspose2d uses the kernel directly
    (no internal flip), so we must flip spatial dims ourselves.
    """
    kernel = flax_params["kernel"]  # (kH, kW, C_out, C_in)
    if transpose_kernel:
        kernel = kernel[::-1, ::-1, :, :]  # flip spatial dims
    new_weight = kernel.transpose(2, 3, 0, 1)  # (C_out, C_in, kH, kW)
    new = eqx.tree_at(lambda m: m.weight, eqx_conv, new_weight)
    if "bias" in flax_params and eqx_conv.bias is not None:
        new = eqx.tree_at(lambda m: m.bias, new, flax_params["bias"].reshape(-1, 1, 1))
    return new


def _transfer_groupnorm(flax_params, eqx_gn):
    """Transfer Flax GroupNorm scale/bias → Equinox GroupNorm weight/bias."""
    new = eqx.tree_at(lambda m: m.weight, eqx_gn, flax_params["scale"])
    new = eqx.tree_at(lambda m: m.bias, new, flax_params["bias"])
    return new


def _transfer_afno2d(flax_params, eqx_afno):
    """Transfer AFNO2D raw parameters."""
    new = eqx_afno
    for name in ("w1", "b1", "w2", "b2"):
        new = eqx.tree_at(lambda m, n=name: getattr(m, n), new, flax_params[name])
    return new


def _transfer_block(flax_params, eqx_block):
    """Transfer a single Block."""
    new = eqx_block

    # GroupNorm_0 → norm1
    new = eqx.tree_at(
        lambda m: m.norm1,
        new,
        _transfer_groupnorm(flax_params["GroupNorm_0"], new.norm1),
    )
    # AFNO2D_0 → afno
    new = eqx.tree_at(
        lambda m: m.afno, new, _transfer_afno2d(flax_params["AFNO2D_0"], new.afno)
    )
    # GroupNorm_1 → norm2
    new = eqx.tree_at(
        lambda m: m.norm2,
        new,
        _transfer_groupnorm(flax_params["GroupNorm_1"], new.norm2),
    )
    # mlp_dense_1
    new = eqx.tree_at(
        lambda m: m.mlp_dense_1,
        new,
        _transfer_dense_to_linear(flax_params["mlp_dense_1"], new.mlp_dense_1),
    )
    # mlp_dense_2
    new = eqx.tree_at(
        lambda m: m.mlp_dense_2,
        new,
        _transfer_dense_to_linear(flax_params["mlp_dense_2"], new.mlp_dense_2),
    )
    return new


def transfer_all_params(flax_params, eqx_model):
    """Transfer all Flax DPOTNet params → Equinox DPOTNet."""
    p = flax_params
    if "params" in p:
        p = p["params"]

    new = eqx_model

    # --- patch_embed ---
    pe = p["patch_embed"]
    new_conv_patch = _transfer_conv_to_conv2d(
        pe["conv_patch"], new.patch_embed.conv_patch
    )
    new = eqx.tree_at(lambda m: m.patch_embed.conv_patch, new, new_conv_patch)
    new_conv_1x1 = _transfer_conv_to_conv2d(pe["conv_1x1"], new.patch_embed.conv_1x1)
    new = eqx.tree_at(lambda m: m.patch_embed.conv_1x1, new, new_conv_1x1)

    # --- pos_embed ---
    new = eqx.tree_at(lambda m: m.pos_embed, new, p["pos_embed"])

    # --- time_agg_layer ---
    ta = p["time_agg_layer"]
    new = eqx.tree_at(lambda m: m.time_agg_layer.w, new, ta["w"])
    if "gamma" in ta and new.time_agg_layer.gamma is not None:
        new = eqx.tree_at(lambda m: m.time_agg_layer.gamma, new, ta["gamma"])

    # --- blocks ---
    for i in range(eqx_model.depth):
        bk = p[f"blocks_{i}"]
        new_block = _transfer_block(bk, new.blocks[i])
        new = eqx.tree_at(lambda m, idx=i: m.blocks[idx], new, new_block)

    # --- cls head ---
    new = eqx.tree_at(
        lambda m: m.cls_dense_1,
        new,
        _transfer_dense_to_linear(p["cls_dense_1"], new.cls_dense_1),
    )
    new = eqx.tree_at(
        lambda m: m.cls_dense_2,
        new,
        _transfer_dense_to_linear(p["cls_dense_2"], new.cls_dense_2),
    )
    new = eqx.tree_at(
        lambda m: m.cls_dense_3,
        new,
        _transfer_dense_to_linear(p["cls_dense_3"], new.cls_dense_3),
    )

    # --- output head ---
    new = eqx.tree_at(
        lambda m: m.out_deconv,
        new,
        _transfer_conv_transpose_to_conv_transpose2d(
            p["out_deconv"], new.out_deconv, transpose_kernel=True
        ),
    )
    new = eqx.tree_at(
        lambda m: m.out_conv_1,
        new,
        _transfer_conv_to_conv2d(p["out_conv_1"], new.out_conv_1),
    )
    new = eqx.tree_at(
        lambda m: m.out_conv_2,
        new,
        _transfer_conv_to_conv2d(p["out_conv_2"], new.out_conv_2),
    )

    # --- optional normalisation layers ---
    if "scale_feats_mu" in p:
        new = eqx.tree_at(
            lambda m: m.scale_feats_mu,
            new,
            _transfer_dense_to_linear(p["scale_feats_mu"], new.scale_feats_mu),
        )
        new = eqx.tree_at(
            lambda m: m.scale_feats_sigma,
            new,
            _transfer_dense_to_linear(p["scale_feats_sigma"], new.scale_feats_sigma),
        )

    return new


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDPOTEquivalence:

    @pytest.fixture
    def models_and_params(self):
        """Create small Flax + Equinox DPOTNet, transfer weights."""
        rng = jax.random.PRNGKey(42)

        kwargs = dict(
            img_size=32,
            patch_size=4,
            mixing_type="afno",
            in_channels=2,
            out_channels=2,
            in_timesteps=3,
            out_timesteps=1,
            n_blocks=4,
            embed_dim=32,
            out_layer_dim=16,
            depth=2,
            modes=8,
            mlp_ratio=1.0,
            n_cls=4,
            normalize=False,
            act="gelu",
            time_agg="exp_mlp",
        )

        flax_model = FlaxDPOTNet(**kwargs)
        x = jnp.zeros((1, 32, 32, 3, 2))
        flax_vars = flax_model.init(rng, x)

        eqx_model = EqxDPOTNet(**kwargs, key=jax.random.PRNGKey(0))
        eqx_model = transfer_all_params(flax_vars, eqx_model)

        return flax_model, flax_vars, eqx_model, kwargs

    def test_forward_equivalence(self, models_and_params):
        flax_model, flax_vars, eqx_model, kwargs = models_and_params
        x = jax.random.normal(jax.random.PRNGKey(7), (1, 32, 32, 3, 2))

        f_pred, f_cls = flax_model.apply(flax_vars, x)
        e_pred, e_cls = eqx_model(x)

        pred_diff = jnp.max(jnp.abs(f_pred - e_pred))
        cls_diff = jnp.max(jnp.abs(f_cls - e_cls))
        print(f"\nPred max diff: {pred_diff:.2e}")
        print(f"Cls  max diff: {cls_diff:.2e}")
        print(f"Flax pred range: [{f_pred.min():.4f}, {f_pred.max():.4f}]")
        print(f"Eqx  pred range: [{e_pred.min():.4f}, {e_pred.max():.4f}]")

        assert jnp.allclose(
            f_pred, e_pred, atol=1e-5, rtol=1e-5
        ), f"pred diff={pred_diff:.2e}"
        assert jnp.allclose(
            f_cls, e_cls, atol=1e-5, rtol=1e-5
        ), f"cls diff={cls_diff:.2e}"

    def test_output_shapes(self, models_and_params):
        flax_model, flax_vars, eqx_model, kwargs = models_and_params
        x = jnp.ones((2, 32, 32, 3, 2))

        f_pred, f_cls = flax_model.apply(flax_vars, x)
        e_pred, e_cls = eqx_model(x)

        assert f_pred.shape == e_pred.shape, f"{f_pred.shape} vs {e_pred.shape}"
        assert f_cls.shape == e_cls.shape, f"{f_cls.shape} vs {e_cls.shape}"

    def test_gradient_flows(self, models_and_params):
        _, _, eqx_model, _ = models_and_params
        x = jax.random.normal(jax.random.PRNGKey(99), (1, 32, 32, 3, 2))

        def loss(model):
            pred, cls_pred = model(x)
            return jnp.mean(pred**2) + jnp.mean(cls_pred**2)

        grads = eqx.filter_grad(loss)(eqx_model)
        for leaf in jax.tree_util.tree_leaves(grads):
            if hasattr(leaf, "shape"):
                assert jnp.all(jnp.isfinite(leaf)), "Non-finite gradient"


class TestDPOTWithNormalize:

    @pytest.fixture
    def models_and_params(self):
        rng = jax.random.PRNGKey(42)

        kwargs = dict(
            img_size=32,
            patch_size=4,
            mixing_type="afno",
            in_channels=2,
            out_channels=2,
            in_timesteps=3,
            out_timesteps=1,
            n_blocks=4,
            embed_dim=32,
            out_layer_dim=16,
            depth=2,
            modes=8,
            mlp_ratio=1.0,
            n_cls=4,
            normalize=True,
            act="gelu",
            time_agg="exp_mlp",
        )

        flax_model = FlaxDPOTNet(**kwargs)
        x = jnp.ones((1, 32, 32, 3, 2))
        flax_vars = flax_model.init(rng, x)

        eqx_model = EqxDPOTNet(**kwargs, key=jax.random.PRNGKey(0))
        eqx_model = transfer_all_params(flax_vars, eqx_model)

        return flax_model, flax_vars, eqx_model

    def test_forward_with_normalize(self, models_and_params):
        flax_model, flax_vars, eqx_model = models_and_params
        x = jax.random.normal(jax.random.PRNGKey(7), (1, 32, 32, 3, 2)) + 5.0

        f_pred, f_cls = flax_model.apply(flax_vars, x)
        e_pred, e_cls = eqx_model(x)

        pred_diff = jnp.max(jnp.abs(f_pred - e_pred))
        cls_diff = jnp.max(jnp.abs(f_cls - e_cls))
        print(f"\nNormalize pred diff: {pred_diff:.2e}")
        print(f"Normalize cls  diff: {cls_diff:.2e}")

        assert jnp.allclose(
            f_pred, e_pred, atol=1e-5, rtol=1e-5
        ), f"pred diff={pred_diff:.2e}"
        assert jnp.allclose(
            f_cls, e_cls, atol=1e-5, rtol=1e-5
        ), f"cls diff={cls_diff:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
