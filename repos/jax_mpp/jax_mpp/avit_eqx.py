"""Equinox reimplementation of AViT (MPP).

Mirrors the Flax Linen modules in this package but uses ``equinox.Module``
instead of ``flax.linen``.  All shapes and forward-pass logic are identical
so that weights can be transferred 1-to-1.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange


# ---------------------------------------------------------------------------
# Position bias modules
# ---------------------------------------------------------------------------


class RelativePositionBias(eqx.Module):
    bidirectional: bool = eqx.field(static=True)
    num_buckets: int = eqx.field(static=True)
    max_distance: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)

    embedding: jnp.ndarray  # (num_buckets, n_heads)

    def __init__(
        self,
        n_heads: int,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
        *,
        key: jax.Array,
    ):
        self.n_heads = n_heads
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.embedding = jax.random.normal(key, (num_buckets, n_heads)) * 0.02

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=32
    ):
        ret = jnp.zeros_like(relative_position, dtype=jnp.int32)
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret = ret + (n < 0).astype(jnp.int32) * num_buckets
            n = jnp.abs(n)
        else:
            n = jnp.maximum(n, 0)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            jnp.log(n.astype(jnp.float32) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(jnp.int32)
        val_if_large = jnp.minimum(val_if_large, num_buckets - 1)
        ret = ret + jnp.where(is_small, n, val_if_large)
        return ret

    def __call__(self, qlen: int, klen: int, bc: int = 0):
        context_position = jnp.arange(qlen)[:, None]
        memory_position = jnp.arange(klen)[None, :]
        relative_position = memory_position - context_position

        is_periodic = bc == 1
        thresh = klen // 2
        rp_wrapped = jnp.where(
            relative_position < -thresh, relative_position % thresh, relative_position
        )
        rp_wrapped = jnp.where(rp_wrapped > thresh, rp_wrapped % (-thresh), rp_wrapped)
        relative_position = jnp.where(is_periodic, rp_wrapped, relative_position)

        rp_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
        )
        values = self.embedding[rp_bucket]  # (qlen, klen, n_heads)
        return values.transpose(2, 0, 1)[None, :, :, :]


class ContinuousPositionBias1D(eqx.Module):
    n_heads: int = eqx.field(static=True)
    cpb_mlp_0: eqx.nn.Linear  # 1 -> 512
    cpb_mlp_2: eqx.nn.Linear  # 512 -> n_heads (no bias)

    def __init__(self, n_heads: int, *, key: jax.Array):
        self.n_heads = n_heads
        k1, k2 = jax.random.split(key)
        self.cpb_mlp_0 = eqx.nn.Linear(1, 512, key=k1)
        self.cpb_mlp_2 = eqx.nn.Linear(512, n_heads, use_bias=False, key=k2)

    def __call__(self, h: int, h2: int, bc: int = 0):
        coords_open = jnp.arange(-(h - 1), h, dtype=jnp.float32) / (h - 1)
        periodic_parts = jnp.concatenate(
            [
                jnp.arange(1, h // 2 + 1, dtype=jnp.float32),
                jnp.arange(-(h // 2 - 1), h // 2 + 1, dtype=jnp.float32),
                jnp.arange(-(h // 2 - 1), 0, dtype=jnp.float32),
            ]
        ) / (h - 1)
        pad_len = (2 * h - 1) - periodic_parts.shape[0]
        coords_periodic = jnp.concatenate(
            [periodic_parts, jnp.zeros(pad_len, dtype=jnp.float32)]
        )

        is_periodic = bc == 1
        relative_coords = jnp.where(is_periodic, coords_periodic, coords_open)

        coords = jnp.arange(h, dtype=jnp.float32)
        coords = coords[None, :] - coords[:, None]
        coords = coords + (h - 1)

        x = relative_coords[:, None]  # (2h-1, 1)
        x = jax.vmap(self.cpb_mlp_0)(x)
        x = jax.nn.relu(x)
        x = jax.vmap(self.cpb_mlp_2)(x)

        rel_pos_model = 16.0 * jax.nn.sigmoid(x.squeeze())  # (2h-1, n_heads)
        biases = rel_pos_model[coords.astype(jnp.int32)]  # (h, h, n_heads)
        return biases.transpose(2, 0, 1)[None, :, :, :]


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class MLP(eqx.Module):
    hidden_dim: int = eqx.field(static=True)
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, hidden_dim: int, exp_factor: float = 4.0, *, key: jax.Array):
        self.hidden_dim = hidden_dim
        inner_dim = int(hidden_dim * exp_factor)
        k1, k2 = jax.random.split(key)
        self.fc1 = eqx.nn.Linear(hidden_dim, inner_dim, key=k1)
        self.fc2 = eqx.nn.Linear(inner_dim, hidden_dim, key=k2)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.fc1(x)
        x = jax.nn.gelu(x, approximate=False)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------


class RMSInstanceNorm2d(eqx.Module):
    dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    weight: jnp.ndarray  # (dim,)
    bias: jnp.ndarray  # (dim,) unused in forward but kept for weight compat

    def __init__(self, dim: int, eps: float = 1e-8):
        self.dim = dim
        self.eps = eps
        self.weight = jnp.ones(dim)
        self.bias = jnp.zeros(dim)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (B, H, W, C)"""
        std = jnp.std(x, axis=(1, 2), keepdims=True, ddof=1)
        x = x / (std + self.eps)
        return x * self.weight[None, None, None, :]


class InstanceNorm2d(eqx.Module):
    dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    weight: jnp.ndarray  # (dim,)
    bias: jnp.ndarray  # (dim,)

    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.weight = jnp.ones(dim)
        self.bias = jnp.zeros(dim)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (B, H, W, C)"""
        mean = jnp.mean(x, axis=(1, 2), keepdims=True)
        var = jnp.var(x, axis=(1, 2), keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        return x * self.weight[None, None, None, :] + self.bias[None, None, None, :]


# ---------------------------------------------------------------------------
# SubsampledLinear
# ---------------------------------------------------------------------------


class SubsampledLinear(eqx.Module):
    dim_in: int = eqx.field(static=True)
    dim_out: int = eqx.field(static=True)
    subsample_in: bool = eqx.field(static=True)

    weight: jnp.ndarray  # (dim_out, dim_in) — same layout as PyTorch
    bias: jnp.ndarray  # (dim_out,)

    def __init__(
        self, dim_in: int, dim_out: int, subsample_in: bool = True, *, key: jax.Array
    ):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.subsample_in = subsample_in
        k1, k2 = jax.random.split(key)
        # lecun_normal init matching Flax
        stddev = 1.0 / math.sqrt(dim_in)
        self.weight = jax.random.truncated_normal(k1, -2, 2, (dim_out, dim_in)) * stddev
        self.bias = jnp.zeros(dim_out)

    def __call__(self, x: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        label_size = labels.shape[0]
        if self.subsample_in:
            scale = (self.dim_in / label_size) ** 0.5
            w_sub = self.weight[:, labels]
            return scale * (x @ w_sub.T + self.bias)
        else:
            w_sub = self.weight[labels]
            b_sub = self.bias[labels]
            return x @ w_sub.T + b_sub


# ---------------------------------------------------------------------------
# hMLP_stem (hierarchical patch embedding)
# ---------------------------------------------------------------------------


class hMLP_stem(eqx.Module):
    embed_dim: int = eqx.field(static=True)

    in_proj_0: eqx.nn.Conv2d  # 4x4 stride 4
    in_proj_1: RMSInstanceNorm2d
    in_proj_3: eqx.nn.Conv2d  # 2x2 stride 2
    in_proj_4: RMSInstanceNorm2d
    in_proj_6: eqx.nn.Conv2d  # 2x2 stride 2
    in_proj_7: RMSInstanceNorm2d

    def __init__(
        self,
        patch_size=(16, 16),
        in_chans: int = 3,
        embed_dim: int = 768,
        *,
        key: jax.Array,
    ):
        self.embed_dim = embed_dim
        q = embed_dim // 4
        k1, k2, k3 = jax.random.split(key, 3)
        self.in_proj_0 = eqx.nn.Conv2d(
            in_chans, q, kernel_size=4, stride=4, use_bias=False, key=k1
        )
        self.in_proj_1 = RMSInstanceNorm2d(q)
        self.in_proj_3 = eqx.nn.Conv2d(
            q, q, kernel_size=2, stride=2, use_bias=False, key=k2
        )
        self.in_proj_4 = RMSInstanceNorm2d(q)
        self.in_proj_6 = eqx.nn.Conv2d(
            q, embed_dim, kernel_size=2, stride=2, use_bias=False, key=k3
        )
        self.in_proj_7 = RMSInstanceNorm2d(embed_dim)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (B, H, W, C) channels-last -> (B, H', W', embed_dim)."""

        # Conv2d expects (C, H, W), so convert
        def _single(xi):
            # xi: (H, W, C)
            xi = xi.transpose(2, 0, 1)  # (C, H, W)
            xi = self.in_proj_0(xi)
            return xi.transpose(1, 2, 0)  # (H', W', q)

        x = jax.vmap(_single)(x)  # (B, H', W', q)
        x = self.in_proj_1(x)
        x = jax.nn.gelu(x, approximate=False)

        def _single2(xi):
            xi = xi.transpose(2, 0, 1)
            xi = self.in_proj_3(xi)
            return xi.transpose(1, 2, 0)

        x = jax.vmap(_single2)(x)
        x = self.in_proj_4(x)
        x = jax.nn.gelu(x, approximate=False)

        def _single3(xi):
            xi = xi.transpose(2, 0, 1)
            xi = self.in_proj_6(xi)
            return xi.transpose(1, 2, 0)

        x = jax.vmap(_single3)(x)
        x = self.in_proj_7(x)
        return x


# ---------------------------------------------------------------------------
# hMLP_output (hierarchical patch de-embedding)
# ---------------------------------------------------------------------------


class hMLP_output(eqx.Module):
    embed_dim: int = eqx.field(static=True)
    out_chans: int = eqx.field(static=True)

    out_proj_0: eqx.nn.ConvTranspose2d  # 2x2 stride 2
    out_proj_1: RMSInstanceNorm2d
    out_proj_3: eqx.nn.ConvTranspose2d  # 2x2 stride 2
    out_proj_4: RMSInstanceNorm2d
    out_kernel: jnp.ndarray  # (out_chans, q, 4, 4)
    out_bias: jnp.ndarray  # (out_chans,)

    def __init__(
        self,
        patch_size=(16, 16),
        out_chans: int = 3,
        embed_dim: int = 768,
        *,
        key: jax.Array,
    ):
        self.embed_dim = embed_dim
        self.out_chans = out_chans
        q = embed_dim // 4
        k1, k2, k3 = jax.random.split(key, 3)
        self.out_proj_0 = eqx.nn.ConvTranspose2d(
            embed_dim, q, kernel_size=2, stride=2, use_bias=False, key=k1
        )
        self.out_proj_1 = RMSInstanceNorm2d(q)
        self.out_proj_3 = eqx.nn.ConvTranspose2d(
            q, q, kernel_size=2, stride=2, use_bias=False, key=k2
        )
        self.out_proj_4 = RMSInstanceNorm2d(q)
        stddev = 1.0 / math.sqrt(q)
        self.out_kernel = (
            jax.random.truncated_normal(k3, -2, 2, (out_chans, q, 4, 4)) * stddev
        )
        self.out_bias = jnp.zeros(out_chans)

    def __call__(self, x: jnp.ndarray, state_labels: jnp.ndarray) -> jnp.ndarray:
        """x: (B, H', W', embed_dim) -> (B, H, W, n_out)."""
        q = self.embed_dim // 4

        def _ct_single(xi):
            xi = xi.transpose(2, 0, 1)  # (C, H, W)
            xi = self.out_proj_0(xi)
            return xi.transpose(1, 2, 0)

        x = jax.vmap(_ct_single)(x)
        x = self.out_proj_1(x)
        x = jax.nn.gelu(x, approximate=False)

        def _ct_single2(xi):
            xi = xi.transpose(2, 0, 1)
            xi = self.out_proj_3(xi)
            return xi.transpose(1, 2, 0)

        x = jax.vmap(_ct_single2)(x)
        x = self.out_proj_4(x)
        x = jax.nn.gelu(x, approximate=False)

        # Final transposed conv with subsampled kernel
        kernel_sub = self.out_kernel[state_labels]  # (n_out, q, 4, 4)
        kernel_flax = kernel_sub.transpose(2, 3, 0, 1)  # (4, 4, n_out, q)
        bias_sub = self.out_bias[state_labels]

        x = jax.lax.conv_transpose(
            x,
            kernel_flax,
            strides=(4, 4),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            transpose_kernel=True,
        )
        x = x + bias_sub[None, None, None, :]
        return x


# ---------------------------------------------------------------------------
# Attention helper
# ---------------------------------------------------------------------------


def _scaled_dot_product_attention(q, k, v, attn_mask=None):
    d = q.shape[-1]
    scale = d**-0.5
    attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = jax.nn.softmax(attn, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", attn, v)


# ---------------------------------------------------------------------------
# AttentionBlock (temporal)
# ---------------------------------------------------------------------------


class AttentionBlock(eqx.Module):
    hidden_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    layer_scale_init_value: float = eqx.field(static=True)
    bias_type: str = eqx.field(static=True)

    norm1: InstanceNorm2d
    input_head: eqx.nn.Conv2d  # 1x1, hidden_dim -> 3*hidden_dim
    qnorm: eqx.nn.LayerNorm
    knorm: eqx.nn.LayerNorm
    rel_pos_bias: Optional[
        object
    ]  # RelativePositionBias or ContinuousPositionBias1D or None
    norm2: InstanceNorm2d
    output_head: eqx.nn.Conv2d  # 1x1, hidden_dim -> hidden_dim
    gamma: Optional[jnp.ndarray]

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        bias_type: str = "rel",
        layer_scale_init_value: float = 1e-6,
        *,
        key: jax.Array,
    ):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.bias_type = bias_type
        self.layer_scale_init_value = layer_scale_init_value

        head_dim = hidden_dim // num_heads
        k1, k2, k3 = jax.random.split(key, 3)

        self.norm1 = InstanceNorm2d(hidden_dim)
        self.input_head = eqx.nn.Conv2d(
            hidden_dim, 3 * hidden_dim, kernel_size=1, key=k1
        )
        self.qnorm = eqx.nn.LayerNorm(head_dim)
        self.knorm = eqx.nn.LayerNorm(head_dim)

        if bias_type == "rel":
            self.rel_pos_bias = RelativePositionBias(n_heads=num_heads, key=k2)
        elif bias_type == "continuous":
            self.rel_pos_bias = ContinuousPositionBias1D(n_heads=num_heads, key=k2)
        else:
            self.rel_pos_bias = None

        self.norm2 = InstanceNorm2d(hidden_dim)
        self.output_head = eqx.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, key=k3)

        if layer_scale_init_value > 0:
            self.gamma = jnp.full((hidden_dim,), layer_scale_init_value)
        else:
            self.gamma = None

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """x: (T, B, H, W, C) -> (T, B, H, W, C)."""
        T, B, H, W, C = x.shape
        head_dim = C // self.num_heads
        residual = x

        # Pre-norm
        x_flat = rearrange(x, "t b h w c -> (t b) h w c")
        x_flat = self.norm1(x_flat)

        # QKV via 1x1 conv
        def _conv_single(xi):
            xi = xi.transpose(2, 0, 1)  # (C, H, W)
            xi = self.input_head(xi)
            return xi.transpose(1, 2, 0)  # (H, W, 3C)

        x_flat = jax.vmap(_conv_single)(x_flat)

        # Rearrange for temporal attention
        x_tmp = rearrange(
            x_flat, "(t b) h w (he c) -> (b h w) he t c", t=T, he=self.num_heads
        )
        q, k, v = jnp.split(x_tmp, 3, axis=-1)

        # QK norm — vmap over (BHW, heads, T)
        q = jax.vmap(jax.vmap(jax.vmap(self.qnorm)))(q)
        k = jax.vmap(jax.vmap(jax.vmap(self.knorm)))(k)

        # Position bias
        if self.rel_pos_bias is not None:
            rel_pos = self.rel_pos_bias(T, T)
        else:
            rel_pos = None

        out = _scaled_dot_product_attention(q, k, v, rel_pos)

        out = rearrange(
            out, "(b h w) he t c -> (t b) h w (he c)", h=H, w=W, he=self.num_heads
        )

        # Post-norm + output projection
        out = self.norm2(out)

        def _conv_out(xi):
            xi = xi.transpose(2, 0, 1)
            xi = self.output_head(xi)
            return xi.transpose(1, 2, 0)

        out = jax.vmap(_conv_out)(out)
        out = rearrange(out, "(t b) h w c -> t b h w c", t=T)

        # Layer scale
        if self.gamma is not None:
            out = out * self.gamma[None, None, None, None, :]

        return out + residual


# ---------------------------------------------------------------------------
# AxialAttentionBlock (spatial)
# ---------------------------------------------------------------------------


class AxialAttentionBlock(eqx.Module):
    hidden_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    layer_scale_init_value: float = eqx.field(static=True)
    bias_type: str = eqx.field(static=True)

    norm1: RMSInstanceNorm2d
    input_head: eqx.nn.Conv2d  # 1x1, hidden_dim -> 3*hidden_dim
    qnorm: eqx.nn.LayerNorm
    knorm: eqx.nn.LayerNorm
    rel_pos_bias: Optional[object]
    norm2: RMSInstanceNorm2d
    output_head: eqx.nn.Conv2d  # 1x1, hidden_dim -> hidden_dim
    gamma_att: Optional[jnp.ndarray]

    mlp: MLP
    mlp_norm: RMSInstanceNorm2d
    gamma_mlp: Optional[jnp.ndarray]

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        bias_type: str = "rel",
        layer_scale_init_value: float = 1e-6,
        *,
        key: jax.Array,
    ):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.bias_type = bias_type
        self.layer_scale_init_value = layer_scale_init_value

        head_dim = hidden_dim // num_heads
        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.norm1 = RMSInstanceNorm2d(hidden_dim)
        self.input_head = eqx.nn.Conv2d(
            hidden_dim, 3 * hidden_dim, kernel_size=1, key=k1
        )
        self.qnorm = eqx.nn.LayerNorm(head_dim)
        self.knorm = eqx.nn.LayerNorm(head_dim)

        if bias_type == "rel":
            self.rel_pos_bias = RelativePositionBias(n_heads=num_heads, key=k2)
        elif bias_type == "continuous":
            self.rel_pos_bias = ContinuousPositionBias1D(n_heads=num_heads, key=k2)
        else:
            self.rel_pos_bias = None

        self.norm2 = RMSInstanceNorm2d(hidden_dim)
        self.output_head = eqx.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, key=k3)

        if layer_scale_init_value > 0:
            self.gamma_att = jnp.full((hidden_dim,), layer_scale_init_value)
            self.gamma_mlp = jnp.full((hidden_dim,), layer_scale_init_value)
        else:
            self.gamma_att = None
            self.gamma_mlp = None

        self.mlp = MLP(hidden_dim=hidden_dim, key=k4)
        self.mlp_norm = RMSInstanceNorm2d(hidden_dim)

    def __call__(
        self, x: jnp.ndarray, bcs: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        """x: (B, H, W, C) -> (B, H, W, C)."""
        B, H, W, C = x.shape
        head_dim = C // self.num_heads

        # --- Attention branch ---
        residual = x
        x = self.norm1(x)

        def _conv_qkv(xi):
            xi = xi.transpose(2, 0, 1)
            xi = self.input_head(xi)
            return xi.transpose(1, 2, 0)

        x = jax.vmap(_conv_qkv)(x)  # (B, H, W, 3C)
        x = rearrange(x, "b h w (he c) -> b he h w c", he=self.num_heads)
        q, k, v = jnp.split(x, 3, axis=-1)

        # QK norm — vmap over (B, heads, H, W)
        q = jax.vmap(jax.vmap(jax.vmap(jax.vmap(self.qnorm))))(q)
        k = jax.vmap(jax.vmap(jax.vmap(jax.vmap(self.knorm))))(k)

        # X-axis attention
        qx = rearrange(q, "b he h w c -> (b h) he w c")
        kx = rearrange(k, "b he h w c -> (b h) he w c")
        vx = rearrange(v, "b he h w c -> (b h) he w c")

        if self.rel_pos_bias is not None:
            rel_x = self.rel_pos_bias(W, W, bcs[0, 0])
        else:
            rel_x = None
        xx = _scaled_dot_product_attention(qx, kx, vx, rel_x)
        xx = rearrange(xx, "(b h) he w c -> b (he c) h w", h=H)

        # Y-axis attention
        qy = rearrange(q, "b he h w c -> (b w) he h c")
        ky = rearrange(k, "b he h w c -> (b w) he h c")
        vy = rearrange(v, "b he h w c -> (b w) he h c")

        if self.rel_pos_bias is not None:
            rel_y = self.rel_pos_bias(H, H, bcs[0, 1])
        else:
            rel_y = None
        xy = _scaled_dot_product_attention(qy, ky, vy, rel_y)
        xy = rearrange(xy, "(b w) he h c -> b (he c) h w", w=W)

        x = (xx + xy) / 2.0  # (B, C, H, W)

        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm2(x)
        x = rearrange(x, "b h w c -> b c h w")
        x = rearrange(x, "b (he c) h w -> b h w (he c)", he=self.num_heads)

        def _conv_out(xi):
            xi = xi.transpose(2, 0, 1)
            xi = self.output_head(xi)
            return xi.transpose(1, 2, 0)

        x = jax.vmap(_conv_out)(x)  # (B, H, W, C)
        x = rearrange(x, "b h w c -> b c h w")

        if self.gamma_att is not None:
            x = x * self.gamma_att[None, :, None, None]

        x = rearrange(x, "b c h w -> b h w c")
        x = x + residual

        # --- MLP branch ---
        residual = x
        x = jax.vmap(jax.vmap(jax.vmap(self.mlp)))(x)  # (B, H, W, C)
        x = rearrange(x, "b h w c -> b c h w")
        x = self.mlp_norm(rearrange(x, "b c h w -> b h w c"))
        x = rearrange(x, "b h w c -> b c h w")

        if self.gamma_mlp is not None:
            x = x * self.gamma_mlp[None, :, None, None]

        x = rearrange(x, "b c h w -> b h w c")
        x = x + residual
        return x


# ---------------------------------------------------------------------------
# SpaceTimeBlock
# ---------------------------------------------------------------------------


class SpaceTimeBlock(eqx.Module):
    hidden_dim: int = eqx.field(static=True)

    temporal: AttentionBlock
    spatial: AxialAttentionBlock

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        bias_type: str = "rel",
        *,
        key: jax.Array,
    ):
        self.hidden_dim = hidden_dim
        k1, k2 = jax.random.split(key)
        self.temporal = AttentionBlock(
            hidden_dim, num_heads, bias_type=bias_type, key=k1
        )
        self.spatial = AxialAttentionBlock(
            hidden_dim, num_heads, bias_type=bias_type, key=k2
        )

    def __call__(
        self, x: jnp.ndarray, bcs: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        """x: (T, B, H, W, C) -> (T, B, H, W, C)."""
        T = x.shape[0]
        x = self.temporal(x, deterministic=deterministic)
        x = rearrange(x, "t b h w c -> (t b) h w c")
        x = self.spatial(x, bcs, deterministic=deterministic)
        x = rearrange(x, "(t b) h w c -> t b h w c", t=T)
        return x


# ---------------------------------------------------------------------------
# AViT (top-level model)
# ---------------------------------------------------------------------------


class AViT(eqx.Module):
    patch_size: tuple = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    processor_blocks: int = eqx.field(static=True)
    n_states: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    bias_type: str = eqx.field(static=True)

    space_bag: SubsampledLinear
    embed: hMLP_stem
    blocks: list  # list of SpaceTimeBlock
    debed: hMLP_output

    def __init__(
        self,
        patch_size: tuple = (16, 16),
        embed_dim: int = 768,
        processor_blocks: int = 8,
        n_states: int = 6,
        drop_path: float = 0.2,
        bias_type: str = "rel",
        num_heads: int = 12,
        *,
        key: jax.Array,
    ):
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.processor_blocks = processor_blocks
        self.n_states = n_states
        self.num_heads = num_heads
        self.bias_type = bias_type

        keys = jax.random.split(key, processor_blocks + 3)

        self.space_bag = SubsampledLinear(
            dim_in=n_states,
            dim_out=embed_dim // 4,
            key=keys[0],
        )
        self.embed = hMLP_stem(
            patch_size=patch_size,
            in_chans=embed_dim // 4,
            embed_dim=embed_dim,
            key=keys[1],
        )

        self.blocks = [
            SpaceTimeBlock(
                hidden_dim=embed_dim,
                num_heads=num_heads,
                bias_type=bias_type,
                key=keys[2 + i],
            )
            for i in range(processor_blocks)
        ]

        self.debed = hMLP_output(
            patch_size=patch_size,
            out_chans=n_states,
            embed_dim=embed_dim,
            key=keys[2 + processor_blocks],
        )

    def __call__(
        self,
        x: jnp.ndarray,
        state_labels: jnp.ndarray,
        bcs: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Args:
            x: (T, B, C, H, W)
            state_labels: 1-D array of active state indices
            bcs: (B, 2) boundary condition flags
            deterministic: disable stochastic depth

        Returns:
            (B, C, H, W) prediction for last timestep
        """
        T, B, C, H, W = x.shape
        state_labels = jnp.asarray(state_labels)

        # 1. Normalise (stop_gradient matches PyTorch's torch.no_grad())
        data_mean = jax.lax.stop_gradient(
            jnp.mean(x, axis=(0, 3, 4), keepdims=True)
        )
        data_std = jax.lax.stop_gradient(
            jnp.std(x, axis=(0, 3, 4), keepdims=True, ddof=1) + 1e-7
        )
        x = (x - data_mean) / data_std

        # 2. Sparse channel projection
        x = rearrange(x, "t b c h w -> t b h w c")
        x = self.space_bag(x, state_labels)

        # 3. Patch embedding
        x = rearrange(x, "t b h w c -> (t b) h w c")
        x = self.embed(x)
        x = rearrange(x, "(t b) h w c -> t b h w c", t=T)

        # 4. Processor blocks
        for block in self.blocks:
            x = block(x, bcs, deterministic=deterministic)

        # 5. Output projection
        x = rearrange(x, "t b h w c -> (t b) h w c")
        x = self.debed(x, state_labels)
        x = rearrange(x, "(t b) h w c -> t b c h w", t=T)

        # 6. De-normalise
        x = x * data_std + data_mean

        return x[-1]
