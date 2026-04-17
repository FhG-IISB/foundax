"""Equinox reimplementation of MORPH (ViT3DRegression).

Mirrors the Flax Linen modules in this package but uses ``equinox.Module``
instead of ``flax.linen``.  All shapes and forward-pass logic are identical
so that weights can be transferred 1-to-1.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp

from jax_morph.patchify import custom_patchify_3d
from jax_morph.positional_encoding import (
    _interpolate_linear_1d,
    _interpolate_bilinear_2d,
)


# ---------------------------------------------------------------------------
# ConvOperator  (channels-first wrapper around eqx.nn.Conv3d)
# ---------------------------------------------------------------------------


class ConvOperator(eqx.Module):
    max_in_ch: int = eqx.field(static=True)
    conv_filter: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)

    input_proj: eqx.nn.Conv3d
    conv_stack: list  # list of eqx.nn.Conv3d

    def __init__(
        self,
        max_in_ch: int = 3,
        conv_filter: int = 8,
        hidden_dim: int = 8,
        *,
        key: jax.Array,
    ):
        self.max_in_ch = max_in_ch
        self.conv_filter = conv_filter
        self.hidden_dim = hidden_dim

        keys = jax.random.split(key, 20)
        ki = 0

        self.input_proj = eqx.nn.Conv3d(
            max_in_ch,
            hidden_dim,
            kernel_size=1,
            use_bias=False,
            key=keys[ki],
        )
        ki += 1

        stack = []
        prev = hidden_dim
        while prev < conv_filter:
            nxt = min(prev * 2, conv_filter)
            stack.append(
                eqx.nn.Conv3d(
                    prev, nxt, kernel_size=3, padding=1, use_bias=False, key=keys[ki]
                )
            )
            prev = nxt
            ki += 1
        # Final conv
        stack.append(
            eqx.nn.Conv3d(
                prev,
                conv_filter,
                kernel_size=3,
                padding=1,
                use_bias=False,
                key=keys[ki],
            )
        )
        self.conv_stack = stack

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (B, D, H, W, C) channels-last -> (B, D, H, W, conv_filter)."""
        in_ch = x.shape[-1]
        if in_ch < self.max_in_ch:
            pad_sz = self.max_in_ch - in_ch
            pad = jnp.zeros((*x.shape[:-1], pad_sz), dtype=x.dtype)
            x = jnp.concatenate([x, pad], axis=-1)

        # channels-last -> channels-first for Conv3d
        x = x.transpose(0, 4, 1, 2, 3)  # (B, C, D, H, W)

        # vmap over batch
        def _single(xi):
            xi = self.input_proj(xi)
            for i, conv in enumerate(self.conv_stack[:-1]):
                xi = conv(xi)
                xi = jax.nn.leaky_relu(xi, negative_slope=0.2)
            xi = self.conv_stack[-1](xi)
            xi = jax.nn.leaky_relu(xi, negative_slope=0.2)
            return xi

        x = jax.vmap(_single)(x)  # (B, conv_filter, D, H, W)
        x = x.transpose(0, 2, 3, 4, 1)  # (B, D, H, W, conv_filter)
        return x


# ---------------------------------------------------------------------------
# FieldCrossAttention
# ---------------------------------------------------------------------------


class FieldCrossAttention(eqx.Module):
    embed_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)

    q: jnp.ndarray  # (1, 1, E) learned query
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear

    def __init__(self, embed_dim: int, num_heads: int = 4, *, key: jax.Array):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads

        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.q = jax.random.normal(k1, (1, 1, embed_dim))

        # DenseGeneral(features=(num_heads, head_dim)) is equivalent to
        # Linear(embed_dim, num_heads*head_dim) with reshape
        self.q_proj = eqx.nn.Linear(embed_dim, num_heads * head_dim, key=k2)
        self.k_proj = eqx.nn.Linear(embed_dim, num_heads * head_dim, key=k3)
        self.v_proj = eqx.nn.Linear(embed_dim, num_heads * head_dim, key=k4)
        self.out_proj = eqx.nn.Linear(embed_dim, embed_dim, key=k5)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (Bn, F, E) -> (Bn, E)."""
        Bn, F, E = x.shape
        head_dim = E // self.num_heads

        q = jnp.broadcast_to(self.q, (Bn, 1, E))

        # Project — vmap over Bn and sequence dims
        def proj(linear, inp):
            # inp: (Bn, seq, E) -> (Bn, seq, out)
            return jax.vmap(jax.vmap(linear))(inp)

        q_p = proj(self.q_proj, q).reshape(Bn, 1, self.num_heads, head_dim)
        k_p = proj(self.k_proj, x).reshape(Bn, F, self.num_heads, head_dim)
        v_p = proj(self.v_proj, x).reshape(Bn, F, self.num_heads, head_dim)

        # (Bn, heads, seq, head_dim)
        q_p = q_p.transpose(0, 2, 1, 3)
        k_p = k_p.transpose(0, 2, 1, 3)
        v_p = v_p.transpose(0, 2, 1, 3)

        scale = head_dim**-0.5
        attn = jnp.matmul(q_p, k_p.transpose(0, 1, 3, 2)) * scale
        attn = jax.nn.softmax(attn, axis=-1)

        out = jnp.matmul(attn, v_p)  # (Bn, heads, 1, head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(Bn, 1, E)

        out = jax.vmap(jax.vmap(self.out_proj))(out)
        return out.squeeze(1)


# ---------------------------------------------------------------------------
# HybridPatchEmbedding3D
# ---------------------------------------------------------------------------


class HybridPatchEmbedding3D(eqx.Module):
    patch_size: Tuple[int, int, int] = eqx.field(static=True)
    max_components: int = eqx.field(static=True)
    conv_filter: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    heads_xa: int = eqx.field(static=True)

    conv_features: ConvOperator
    projection: eqx.nn.Linear
    field_attn: FieldCrossAttention

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (8, 8, 8),
        max_components: int = 3,
        conv_filter: int = 8,
        embed_dim: int = 256,
        heads_xa: int = 32,
        *,
        key: jax.Array,
    ):
        self.patch_size = patch_size
        self.max_components = max_components
        self.conv_filter = conv_filter
        self.embed_dim = embed_dim
        self.heads_xa = heads_xa

        k1, k2, k3 = jax.random.split(key, 3)
        self.conv_features = ConvOperator(
            max_in_ch=max_components,
            conv_filter=conv_filter,
            hidden_dim=8,
            key=k1,
        )

        pW = patch_size[2]
        max_patch_vol = pW**3
        max_features = max_patch_vol * conv_filter
        self.projection = eqx.nn.Linear(max_features, embed_dim, key=k2)
        self.field_attn = FieldCrossAttention(
            embed_dim=embed_dim,
            num_heads=heads_xa,
            key=k3,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (B, t, F, C, D, H, W) -> (B, t, n_patches, embed_dim)."""
        B, t, F, C, D, H, W = x.shape
        pW = self.patch_size[2]
        max_patch_vol = pW**3
        max_features = max_patch_vol * self.conv_filter

        x = x.reshape(B * t * F, C, D, H, W)
        x = x.transpose(0, 2, 3, 4, 1)  # (BtF, D, H, W, C)
        x = self.conv_features(x)  # (BtF, D, H, W, conv_filter)
        x = custom_patchify_3d(x, self.patch_size)  # (BtF, n, features)
        n_patches = x.shape[1]
        features = x.shape[2]

        x = x.reshape(B * t, F, n_patches, features)
        x = x.transpose(0, 2, 1, 3)  # (Bt, n, F, features)
        x = x.reshape(-1, F, features)  # (Bt*n, F, features)

        if features < max_features:
            pad_amt = max_features - features
            pad = jnp.zeros((x.shape[0], x.shape[1], pad_amt), dtype=x.dtype)
            x = jnp.concatenate([x, pad], axis=-1)

        # Project: (Bt*n, F, max_features) -> (Bt*n, F, embed_dim)
        x = jax.vmap(jax.vmap(self.projection))(x)

        # Field cross-attention: (Bt*n, F, embed_dim) -> (Bt*n, embed_dim)
        x = self.field_attn(x)

        x = x.reshape(B, t, n_patches, self.embed_dim)
        return x


# ---------------------------------------------------------------------------
# Positional Encodings (pure param + interpolation, no Flax)
# ---------------------------------------------------------------------------


class PositionalEncodingSLinTSlice(eqx.Module):
    max_ar: int = eqx.field(static=True)
    max_patches: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    pos_embedding: jnp.ndarray  # (1, max_ar, max_patches, dim)

    def __init__(
        self,
        max_ar: int = 5,
        max_patches: int = 512,
        dim: int = 256,
        *,
        key: jax.Array,
    ):
        self.max_ar = max_ar
        self.max_patches = max_patches
        self.dim = dim
        self.pos_embedding = jax.random.normal(key, (1, max_ar, max_patches, dim))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (B, t, n_patches, dim) -> (B, t, n_patches, dim)."""
        B, t, n, D = x.shape
        pe = self.pos_embedding[:, :t, :, :]
        pe = pe.transpose(0, 3, 1, 2).reshape(1, D * t, -1)
        pe = _interpolate_linear_1d(pe, n)
        pe = pe.reshape(1, D, t, n).transpose(0, 2, 3, 1)
        pe = jnp.broadcast_to(pe, (B, t, n, D))
        return pe


class PositionalEncodingSTBilinear(eqx.Module):
    max_ar: int = eqx.field(static=True)
    max_patches: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    pos_embedding: jnp.ndarray  # (1, max_ar, max_patches, dim)

    def __init__(
        self,
        max_ar: int = 16,
        max_patches: int = 4096,
        dim: int = 1024,
        *,
        key: jax.Array,
    ):
        self.max_ar = max_ar
        self.max_patches = max_patches
        self.dim = dim
        self.pos_embedding = jax.random.normal(key, (1, max_ar, max_patches, dim))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (B, t, n_patches, dim) -> (B, t, n_patches, dim)."""
        B, t, n, D = x.shape
        pe = self.pos_embedding.transpose(0, 3, 1, 2)
        pe = _interpolate_bilinear_2d(pe, t, n, antialias=True)
        pe = pe.transpose(0, 2, 3, 1)
        pe = jnp.broadcast_to(pe, (B, t, n, D))
        return pe


# ---------------------------------------------------------------------------
# LoRALinear
# ---------------------------------------------------------------------------


class LoRALinear(eqx.Module):
    features: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    alpha: Optional[int] = eqx.field(static=True)

    base: eqx.nn.Linear
    A: Optional[jnp.ndarray]  # (in_features, rank) or None
    B: Optional[jnp.ndarray]  # (rank, features) or None

    def __init__(
        self,
        in_features: int,
        features: int,
        use_bias: bool = True,
        rank: int = 0,
        alpha: Optional[int] = None,
        *,
        key: jax.Array,
    ):
        self.features = features
        self.rank = rank
        self.alpha = alpha

        k1, k2, k3 = jax.random.split(key, 3)
        self.base = eqx.nn.Linear(in_features, features, use_bias=use_bias, key=k1)

        if rank > 0:
            self.A = jax.random.uniform(k2, (in_features, rank)) * math.sqrt(
                1.0 / in_features
            )
            self.B = jnp.zeros((rank, features))
        else:
            self.A = None
            self.B = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = self.base(x)
        if self.rank > 0 and self.A is not None:
            _alpha = self.alpha if self.alpha is not None else 2 * self.rank
            scaling = _alpha / self.rank
            upd = jnp.dot(jnp.dot(x, self.A), self.B)
            y = y + scaling * upd
        return y


# ---------------------------------------------------------------------------
# LoRAMHA
# ---------------------------------------------------------------------------


class LoRAMHA(eqx.Module):
    embed_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    alpha: Optional[int] = eqx.field(static=True)

    q: LoRALinear
    k: LoRALinear
    v: LoRALinear
    o: LoRALinear

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rank: int = 0,
        alpha: Optional[int] = None,
        *,
        key: jax.Array,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rank = rank
        self.alpha = alpha

        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.q = LoRALinear(
            embed_dim, embed_dim, use_bias=True, rank=rank, alpha=alpha, key=k1
        )
        self.k = LoRALinear(
            embed_dim, embed_dim, use_bias=True, rank=rank, alpha=alpha, key=k2
        )
        self.v = LoRALinear(
            embed_dim, embed_dim, use_bias=True, rank=rank, alpha=alpha, key=k3
        )
        self.o = LoRALinear(
            embed_dim, embed_dim, use_bias=True, rank=rank, alpha=alpha, key=k4
        )

    def __call__(
        self,
        q_in: jnp.ndarray,
        k_in: jnp.ndarray,
        v_in: jnp.ndarray,
    ) -> jnp.ndarray:
        """q_in, k_in, v_in: (B, L, C) -> (B, L, C)."""
        B, L, C = q_in.shape
        head_dim = self.embed_dim // self.num_heads

        # Project via vmap over (B, L)
        proj = lambda linear, inp: jax.vmap(jax.vmap(linear))(inp)
        q = (
            proj(self.q, q_in)
            .reshape(B, L, self.num_heads, head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            proj(self.k, k_in)
            .reshape(B, L, self.num_heads, head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            proj(self.v, v_in)
            .reshape(B, L, self.num_heads, head_dim)
            .transpose(0, 2, 1, 3)
        )

        # SDPA
        if L == 1:
            y = v
        else:
            scale = 1.0 / math.sqrt(head_dim)
            scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
            scores = scores - scores.max(axis=-1, keepdims=True)
            attn = jax.nn.softmax(scores, axis=-1)
            y = jnp.matmul(attn, v)

        y = y.transpose(0, 2, 1, 3).reshape(B, L, C)
        y = proj(self.o, y)
        return y


# ---------------------------------------------------------------------------
# AxialAttention3DSpaceTime
# ---------------------------------------------------------------------------


class AxialAttention3DSpaceTime(eqx.Module):
    dim: int = eqx.field(static=True)
    heads: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    alpha: Optional[int] = eqx.field(static=True)

    attn_t: LoRAMHA
    attn_d: LoRAMHA
    attn_h: LoRAMHA
    attn_w: LoRAMHA

    def __init__(
        self,
        dim: int,
        heads: int,
        rank: int = 0,
        alpha: Optional[int] = None,
        *,
        key: jax.Array,
    ):
        self.dim = dim
        self.heads = heads
        self.rank = rank
        self.alpha = alpha

        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.attn_t = LoRAMHA(dim, heads, rank=rank, alpha=alpha, key=k1)
        self.attn_d = LoRAMHA(dim, heads, rank=rank, alpha=alpha, key=k2)
        self.attn_h = LoRAMHA(dim, heads, rank=rank, alpha=alpha, key=k3)
        self.attn_w = LoRAMHA(dim, heads, rank=rank, alpha=alpha, key=k4)

    def __call__(
        self,
        x: jnp.ndarray,
        grid_size: Tuple[int, int, int],
    ) -> jnp.ndarray:
        """x: (B, t, N, features) -> (B, t, N, features). N = D*H*W."""
        B, t, N, features = x.shape
        D, H, W = grid_size

        x = x.reshape(B, t, D, H, W, features)

        # Time-axis
        xt = x.transpose(0, 2, 3, 4, 1, 5).reshape(B * D * H * W, t, features)
        xt = self.attn_t(xt, xt, xt)
        xt = xt.reshape(B, D, H, W, t, features).transpose(0, 4, 1, 2, 3, 5)
        if t > 1:
            x = x + xt

        # Depth-axis
        xd = x.transpose(0, 1, 3, 4, 2, 5).reshape(B * t * H * W, D, features)
        xd = self.attn_d(xd, xd, xd)
        xd = xd.reshape(B, t, H, W, D, features).transpose(0, 1, 4, 2, 3, 5)

        # Height-axis
        xh = x.transpose(0, 1, 2, 4, 3, 5).reshape(B * t * D * W, H, features)
        xh = self.attn_h(xh, xh, xh)
        xh = xh.reshape(B, t, D, W, H, features).transpose(0, 1, 2, 4, 3, 5)

        # Width-axis
        xw = x.reshape(B * t * D * H, W, features)
        xw = self.attn_w(xw, xw, xw)
        xw = xw.reshape(B, t, D, H, W, features)

        x_comb = x + xd + xh + xw
        return x_comb.reshape(B, t, D * H * W, features)


# ---------------------------------------------------------------------------
# EncoderBlock
# ---------------------------------------------------------------------------


class EncoderBlock(eqx.Module):
    dim: int = eqx.field(static=True)
    mlp_dim: int = eqx.field(static=True)

    norm1: eqx.nn.LayerNorm
    axial_attn: AxialAttention3DSpaceTime
    norm2: eqx.nn.LayerNorm
    mlp_0: LoRALinear
    mlp_1: LoRALinear

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_dim: int,
        lora_r_attn: int = 0,
        lora_r_mlp: int = 0,
        lora_alpha: Optional[int] = None,
        *,
        key: jax.Array,
    ):
        self.dim = dim
        self.mlp_dim = mlp_dim

        k1, k2, k3 = jax.random.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(dim, eps=1e-5)
        self.axial_attn = AxialAttention3DSpaceTime(
            dim=dim,
            heads=heads,
            rank=lora_r_attn,
            alpha=lora_alpha,
            key=k1,
        )
        self.norm2 = eqx.nn.LayerNorm(dim, eps=1e-5)
        self.mlp_0 = LoRALinear(
            dim, mlp_dim, use_bias=True, rank=lora_r_mlp, alpha=lora_alpha, key=k2
        )
        self.mlp_1 = LoRALinear(
            mlp_dim, dim, use_bias=True, rank=lora_r_mlp, alpha=lora_alpha, key=k3
        )

    def __call__(
        self,
        x: jnp.ndarray,
        grid_size: Tuple[int, int, int],
    ) -> jnp.ndarray:
        """x: (B, t, N, dim) -> (B, t, N, dim)."""
        # Attention block
        residual = x
        x = jax.vmap(jax.vmap(jax.vmap(self.norm1)))(x)
        x = self.axial_attn(x, grid_size)
        x = residual + x

        # MLP block
        residual = x
        x = jax.vmap(jax.vmap(jax.vmap(self.norm2)))(x)
        x = jax.vmap(jax.vmap(jax.vmap(self.mlp_0)))(x)
        x = jax.nn.gelu(x, approximate=False)
        x = jax.vmap(jax.vmap(jax.vmap(self.mlp_1)))(x)
        x = residual + x
        return x


# ---------------------------------------------------------------------------
# SimpleDecoder
# ---------------------------------------------------------------------------


class SimpleDecoder(eqx.Module):
    dim: int = eqx.field(static=True)
    max_out_ch: int = eqx.field(static=True)

    norm: eqx.nn.LayerNorm
    linear: eqx.nn.Linear

    def __init__(self, dim: int, max_out_ch: int, *, key: jax.Array):
        self.dim = dim
        self.max_out_ch = max_out_ch
        self.norm = eqx.nn.LayerNorm(dim, eps=1e-5)
        self.linear = eqx.nn.Linear(dim, max_out_ch, key=key)

    def __call__(
        self,
        x: jnp.ndarray,
        fields: int,
        components: int,
        patch_vol: int,
    ) -> jnp.ndarray:
        """x: (B, t, n, dim) -> (B, t, n, out_ch)."""
        out_ch = fields * patch_vol * components
        x = jax.vmap(jax.vmap(jax.vmap(self.norm)))(x)
        x = jax.vmap(jax.vmap(jax.vmap(self.linear)))(x)
        if out_ch < self.max_out_ch:
            x = x[..., :out_ch]
        return x


# ---------------------------------------------------------------------------
# ViT3DRegression (top-level model)
# ---------------------------------------------------------------------------


class ViT3DRegression(eqx.Module):
    patch_size: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    heads: int = eqx.field(static=True)
    heads_xa: int = eqx.field(static=True)
    mlp_dim: int = eqx.field(static=True)
    max_components: int = eqx.field(static=True)
    conv_filter: int = eqx.field(static=True)
    max_ar: int = eqx.field(static=True)
    max_patches: int = eqx.field(static=True)
    max_fields: int = eqx.field(static=True)
    model_size: str = eqx.field(static=True)

    patch_embedding: HybridPatchEmbedding3D
    pos_encoding: Union[PositionalEncodingSLinTSlice, PositionalEncodingSTBilinear]
    transformer_blocks: list  # list of EncoderBlock
    decoder: SimpleDecoder

    def __init__(
        self,
        patch_size: int = 8,
        dim: int = 256,
        depth: int = 4,
        heads: int = 4,
        heads_xa: int = 32,
        mlp_dim: int = 1024,
        max_components: int = 3,
        conv_filter: int = 8,
        max_ar: int = 1,
        max_patches: int = 4096,
        max_fields: int = 3,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        lora_r_attn: int = 0,
        lora_r_mlp: int = 0,
        lora_alpha: Optional[int] = None,
        lora_p: float = 0.0,
        model_size: str = "Ti",
        *,
        key: jax.Array,
    ):
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.heads_xa = heads_xa
        self.mlp_dim = mlp_dim
        self.max_components = max_components
        self.conv_filter = conv_filter
        self.max_ar = max_ar
        self.max_patches = max_patches
        self.max_fields = max_fields
        self.model_size = model_size

        keys = jax.random.split(key, depth + 4)

        pD = pH = pW = patch_size
        self.patch_embedding = HybridPatchEmbedding3D(
            patch_size=(pD, pH, pW),
            max_components=max_components,
            conv_filter=conv_filter,
            embed_dim=dim,
            heads_xa=heads_xa,
            key=keys[0],
        )

        if model_size == "L" and max_ar > 1:
            self.pos_encoding = PositionalEncodingSTBilinear(
                max_ar=max_ar,
                max_patches=max_patches,
                dim=dim,
                key=keys[1],
            )
        else:
            self.pos_encoding = PositionalEncodingSLinTSlice(
                max_ar=max_ar,
                max_patches=max_patches,
                dim=dim,
                key=keys[1],
            )

        self.transformer_blocks = [
            EncoderBlock(
                dim=dim,
                heads=heads,
                mlp_dim=mlp_dim,
                lora_r_attn=lora_r_attn,
                lora_r_mlp=lora_r_mlp,
                lora_alpha=lora_alpha,
                key=keys[2 + i],
            )
            for i in range(depth)
        ]

        max_patch_vol = pW**3
        max_decoder_out_ch = max_fields * max_components * max_patch_vol
        self.decoder = SimpleDecoder(
            dim=dim, max_out_ch=max_decoder_out_ch, key=keys[2 + depth]
        )

    def _patch_tuple(self):
        if isinstance(self.patch_size, (tuple, list)):
            return tuple(self.patch_size)
        return (self.patch_size, self.patch_size, self.patch_size)

    def _get_patch_info(self, volume: Tuple[int, int, int]):
        D, H, W = volume
        pD, pH, pW = self._patch_tuple()
        pD = 1 if D == 1 else pD
        pH = 1 if H == 1 else pH
        pW = 1 if W == 1 else pW
        patch_sizes = (pD, pH, pW)
        n_patches = (D // pD, H // pH, W // pW)
        return patch_sizes, n_patches

    def __call__(
        self, vol: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Forward pass.

        Args:
            vol: (B, t, F, C, D, H, W)

        Returns:
            (enc, z, x_last) where x_last is (B, F, C, D, H, W).
        """
        B, t, F, C, D, H, W = vol.shape
        pD, pH, pW = self._patch_tuple()

        x = self.patch_embedding(vol)
        enc = x

        pe = self.pos_encoding(x)
        x = x + pe

        (pD_actual, pH_actual, pW_actual), (D_p, H_p, W_p) = self._get_patch_info(
            (D, H, W)
        )
        grid_size = (D_p, H_p, W_p)
        patch_vol = pD_actual * pH_actual * pW_actual

        for block in self.transformer_blocks:
            x = block(x, grid_size)
        z = x

        x = self.decoder(x, F, C, patch_vol)

        b, t_out, n, cpd = x.shape
        x_last = x[:, -1, :, :]
        x_last = x_last.reshape(b, n, F, C, pD_actual, pH_actual, pW_actual)
        x_last = x_last.reshape(b, D_p, H_p, W_p, F, C, pD_actual, pH_actual, pW_actual)
        x_last = x_last.transpose(0, 4, 5, 1, 6, 2, 7, 3, 8)
        x_last = x_last.reshape(b, F, C, D, H, W)

        return enc, z, x_last
