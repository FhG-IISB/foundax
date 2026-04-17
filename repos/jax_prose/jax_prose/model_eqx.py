"""
PROSE model family — Equinox implementation.

Covers all four PROSE variants: PROSE1to1, PROSE2to1, PROSEODE2to1, PROSEPDE2to1.
"""

from __future__ import annotations

import math
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp


# ═══════════════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════════════


def patchify(data, patch_num):
    bs, nt, px, py, d = data.shape
    p = patch_num
    x = px // p
    y = py // p
    data = data.reshape(bs, nt, p, x, p, y, d)
    data = jnp.transpose(data, (0, 1, 2, 4, 3, 5, 6))
    return data.reshape(bs, nt, p * p, x * y * d)


def depatchify(data, patch_num, x, y, d):
    bs, nt, _, _ = data.shape
    p = patch_num
    data = data.reshape(bs, nt, p, p, x, y, d)
    data = jnp.transpose(data, (0, 1, 2, 4, 3, 5, 6))
    return data.reshape(bs, nt, p * x, p * y, d)


def _sinusoidal_pe(max_len, dim):
    position = jnp.arange(max_len, dtype=jnp.float32)[:, None]
    div_term = jnp.exp(
        jnp.arange(0, dim, 2, dtype=jnp.float32) * (-jnp.log(10000.0) / dim)
    )
    pe = jnp.zeros((max_len, 1, dim), dtype=jnp.float32)
    pe = pe.at[:, 0, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 0, 1::2].set(jnp.cos(position * div_term))
    return pe


def _sinusoidal_embedding(n_pos, dim):
    pos = jnp.arange(n_pos, dtype=jnp.float32)[:, None]
    i = jnp.arange(dim, dtype=jnp.float32)[None, :]
    angle = pos / jnp.power(10000.0, 2.0 * jnp.floor(i / 2.0) / float(dim))
    out = jnp.zeros((n_pos, dim), dtype=jnp.float32)
    out = out.at[:, 0::2].set(jnp.sin(angle[:, 0::2]))
    out = out.at[:, 1::2].set(jnp.cos(angle[:, 1::2]))
    return out


def _lengths_to_mask(lengths, max_len):
    ar = jnp.arange(max_len, dtype=jnp.int32)[None, :]
    return ar < lengths[:, None]


def _apply_linear(linear, x):
    """Apply eqx.nn.Linear to batched input via matmul."""
    out = x @ linear.weight.T
    if linear.bias is not None:
        out = out + linear.bias
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# PROSE1to1 components (transformer.py + embedder.py + prose_fd.py)
# ═══════════════════════════════════════════════════════════════════════════════


class RMSNorm1to1(eqx.Module):
    """nn.RMSNorm equivalent for PROSE1to1 transformer."""

    weight: jnp.ndarray
    eps: float

    def __init__(self, dim, eps=1e-6, *, key=None):
        self.weight = jnp.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        var = jnp.mean(x * x, axis=-1, keepdims=True)
        return x * jnp.reciprocal(jnp.sqrt(var + self.eps)) * self.weight


class LayerNorm1to1(eqx.Module):
    """eqx.nn.LayerNorm wrapper matching Flax nn.LayerNorm."""

    ln: eqx.nn.LayerNorm

    def __init__(self, dim, eps=1e-6, *, key=None):
        self.ln = eqx.nn.LayerNorm(dim, eps=eps)

    def __call__(self, x):
        return jax.vmap(jax.vmap(self.ln))(x)


class MultiHeadAttention1to1(eqx.Module):
    """Reimplements Flax nn.MultiHeadDotProductAttention for PROSE1to1."""

    num_heads: int
    head_dim: int
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear

    def __init__(self, dim, num_heads, *, key):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.q_proj = eqx.nn.Linear(dim, dim, use_bias=True, key=k1)
        self.k_proj = eqx.nn.Linear(dim, dim, use_bias=True, key=k2)
        self.v_proj = eqx.nn.Linear(dim, dim, use_bias=True, key=k3)
        self.out_proj = eqx.nn.Linear(dim, dim, use_bias=True, key=k4)

    def __call__(self, query, kv=None, mask=None):
        if kv is None:
            kv = query
        bs, q_len, dim = query.shape
        k_len = kv.shape[1]
        h, d = self.num_heads, self.head_dim

        q = (
            _apply_linear(self.q_proj, query)
            .reshape(bs, q_len, h, d)
            .transpose(0, 2, 1, 3)
        )
        k = (
            _apply_linear(self.k_proj, kv)
            .reshape(bs, k_len, h, d)
            .transpose(0, 2, 1, 3)
        )
        v = (
            _apply_linear(self.v_proj, kv)
            .reshape(bs, k_len, h, d)
            .transpose(0, 2, 1, 3)
        )

        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(float(d))
        if mask is not None:
            scores = jnp.where(mask, scores, -1e30)
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(bs, q_len, dim)
        return _apply_linear(self.out_proj, out)


class EncoderBlock(eqx.Module):
    """Transformer encoder block for PROSE1to1."""

    norm_first: bool
    norm_type: str
    norm1: eqx.Module
    norm2: eqx.Module
    attn: MultiHeadAttention1to1
    dense1: eqx.nn.Linear
    dense2: eqx.nn.Linear

    def __init__(
        self, dim_emb, dim_ffn, n_head, norm_type="rms", norm_first=True, *, key
    ):
        self.norm_first = norm_first
        self.norm_type = norm_type
        k1, k2, k3 = jax.random.split(key, 3)
        NormCls = (
            RMSNorm1to1 if norm_type == "rms" else lambda d, **kw: eqx.nn.LayerNorm(d)
        )
        self.norm1 = NormCls(dim_emb)
        self.norm2 = NormCls(dim_emb)
        self.attn = MultiHeadAttention1to1(dim_emb, n_head, key=k1)
        self.dense1 = eqx.nn.Linear(dim_emb, dim_ffn, key=k2)
        self.dense2 = eqx.nn.Linear(dim_ffn, dim_emb, key=k3)

    def _apply_norm(self, norm, x):
        if isinstance(norm, RMSNorm1to1):
            return norm(x)
        # eqx.nn.LayerNorm needs vmap
        return jax.vmap(jax.vmap(norm))(x)

    def __call__(self, x):
        if self.norm_first:
            y = self._apply_norm(self.norm1, x)
            x = x + self.attn(y)
            y = self._apply_norm(self.norm2, x)
        else:
            y = x
            x = self._apply_norm(self.norm1, x + self.attn(y))
            y = x

        y = _apply_linear(self.dense1, y)
        y = jax.nn.gelu(y)
        y = _apply_linear(self.dense2, y)

        if self.norm_first:
            return x + y
        return self._apply_norm(self.norm2, x + y)


class TransformerDataEncoder(eqx.Module):
    layers: list

    def __init__(
        self,
        n_layer,
        dim_emb,
        dim_ffn,
        n_head,
        norm_type="rms",
        norm_first=True,
        *,
        key,
    ):
        keys = jax.random.split(key, n_layer)
        self.layers = [
            EncoderBlock(dim_emb, dim_ffn, n_head, norm_type, norm_first, key=keys[i])
            for i in range(n_layer)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class OperatorDecoderBlock(eqx.Module):
    """Cross-attention decoder block for PROSE1to1."""

    norm_first: bool
    norm_type: str
    norm1: eqx.Module
    norm2: eqx.Module
    attn: MultiHeadAttention1to1
    dense1: eqx.nn.Linear
    dense2: eqx.nn.Linear

    def __init__(
        self, dim_emb, dim_ffn, n_head, norm_type="rms", norm_first=True, *, key
    ):
        self.norm_first = norm_first
        self.norm_type = norm_type
        k1, k2, k3 = jax.random.split(key, 3)
        NormCls = (
            RMSNorm1to1 if norm_type == "rms" else lambda d, **kw: eqx.nn.LayerNorm(d)
        )
        self.norm1 = NormCls(dim_emb)
        self.norm2 = NormCls(dim_emb)
        self.attn = MultiHeadAttention1to1(dim_emb, n_head, key=k1)
        self.dense1 = eqx.nn.Linear(dim_emb, dim_ffn, key=k2)
        self.dense2 = eqx.nn.Linear(dim_ffn, dim_emb, key=k3)

    def _apply_norm(self, norm, x):
        if isinstance(norm, RMSNorm1to1):
            return norm(x)
        return jax.vmap(jax.vmap(norm))(x)

    def __call__(self, query, memory, mask=None):
        if self.norm_first:
            y = self._apply_norm(self.norm1, query)
            query = query + self.attn(y, kv=memory, mask=mask)
            y = self._apply_norm(self.norm2, query)
        else:
            y = query
            query = self._apply_norm(
                self.norm1, query + self.attn(y, kv=memory, mask=mask)
            )
            y = query

        y = _apply_linear(self.dense1, y)
        y = jax.nn.gelu(y)
        y = _apply_linear(self.dense2, y)

        if self.norm_first:
            return query + y
        return self._apply_norm(self.norm2, query + y)


class DataOperatorDecoder(eqx.Module):
    """Decoder with query embeddings for PROSE1to1."""

    dim: int
    n_layer: int
    norm_first: bool
    norm_type: str
    final_ln: bool
    time_embed_type: str
    time_embed: Optional[jnp.ndarray]  # learnable
    time_proj_dense1: Optional[eqx.nn.Linear]  # continuous
    time_proj_dense2: Optional[eqx.nn.Linear]  # continuous
    patch_position_embeddings: jnp.ndarray
    layers: list
    final_norm: Optional[eqx.Module]

    def __init__(
        self,
        dim_emb,
        dim_ffn,
        n_head,
        n_layer,
        patch_num_output,
        norm_type="rms",
        norm_first=True,
        final_ln=True,
        time_embed_type="learnable",
        max_time_len=32,
        *,
        key,
    ):
        self.dim = dim_emb
        self.n_layer = n_layer
        self.norm_first = norm_first
        self.norm_type = norm_type
        self.final_ln = final_ln
        self.time_embed_type = time_embed_type
        space_len = patch_num_output**2

        keys = jax.random.split(key, n_layer + 3)

        if time_embed_type == "continuous":
            self.time_embed = None
            self.time_proj_dense1 = eqx.nn.Linear(1, dim_emb, key=keys[0])
            self.time_proj_dense2 = eqx.nn.Linear(dim_emb, dim_emb, key=keys[1])
        else:
            self.time_embed = jax.random.normal(keys[0], (1, max_time_len, 1, dim_emb))
            self.time_proj_dense1 = None
            self.time_proj_dense2 = None

        self.patch_position_embeddings = jax.random.normal(
            keys[1], (1, 1, space_len, dim_emb)
        )

        self.layers = [
            OperatorDecoderBlock(
                dim_emb, dim_ffn, n_head, norm_type, norm_first, key=keys[2 + i]
            )
            for i in range(n_layer)
        ]

        if norm_first and final_ln:
            NormCls = (
                RMSNorm1to1 if norm_type == "rms" else lambda d: eqx.nn.LayerNorm(d)
            )
            self.final_norm = NormCls(dim_emb)
        else:
            self.final_norm = None

    def get_query_emb(self, times):
        bs, out_len, _ = times.shape
        if self.time_embed_type == "continuous":
            t = _apply_linear(self.time_proj_dense1, times)
            t = jax.nn.gelu(t)
            t = _apply_linear(self.time_proj_dense2, t)
            t = t[:, :, None, :]
        else:
            t = self.time_embed[:, :out_len, :, :]
        return (t + self.patch_position_embeddings).reshape(bs, -1, self.dim)

    def __call__(self, src, query_emb, src_key_padding_mask=None):
        mask = None
        if src_key_padding_mask is not None:
            valid = (~src_key_padding_mask).astype(jnp.int32)
            q_valid = jnp.ones(
                (query_emb.shape[0], query_emb.shape[1]), dtype=jnp.int32
            )
            mask = jnp.einsum("bi,bj->bij", q_valid, valid)[:, None, :, :]
            mask = mask.astype(jnp.bool_)

        x = query_emb
        for layer in self.layers:
            x = layer(x, src, mask=mask)

        if self.final_norm is not None:
            if isinstance(self.final_norm, RMSNorm1to1):
                x = self.final_norm(x)
            else:
                x = jax.vmap(jax.vmap(self.final_norm))(x)
        return x


# ── Embedders for PROSE1to1 ──


class ConvEmbedder1to1(eqx.Module):
    """ConvEmbedder for PROSE1to1 (from embedder.py)."""

    dim: int
    patch_num: int
    patch_num_output: int
    x_num: int
    data_dim: int
    time_embed_type: str

    patch_position_embeddings: jnp.ndarray
    time_embed: Optional[jnp.ndarray]
    time_proj_dense1: Optional[eqx.nn.Linear]
    time_proj_dense2: Optional[eqx.nn.Linear]

    conv_proj_0: eqx.nn.Conv2d
    conv_proj_1: eqx.nn.Conv2d
    deconv: eqx.nn.ConvTranspose2d
    post_conv_0: eqx.nn.Conv2d
    post_conv_1: eqx.nn.Conv2d

    def __init__(
        self,
        dim,
        patch_num,
        patch_num_output,
        x_num,
        data_dim,
        time_embed_type="learnable",
        max_time_len=32,
        conv_dim=32,
        *,
        key,
    ):
        self.dim = dim
        self.patch_num = patch_num
        self.patch_num_output = patch_num_output
        self.x_num = x_num
        self.data_dim = data_dim
        self.time_embed_type = time_embed_type

        patch_resolution = x_num // patch_num
        patch_resolution_output = x_num // patch_num_output

        keys = jax.random.split(key, 8)
        self.patch_position_embeddings = jax.random.normal(
            keys[0], (1, 1, patch_num * patch_num, dim)
        )

        if time_embed_type == "continuous":
            self.time_embed = None
            self.time_proj_dense1 = eqx.nn.Linear(1, dim, key=keys[1])
            self.time_proj_dense2 = eqx.nn.Linear(dim, dim, key=keys[2])
        else:
            self.time_embed = jax.random.normal(keys[1], (1, max_time_len, 1, dim))
            self.time_proj_dense1 = None
            self.time_proj_dense2 = None

        # Flax Conv NHWC kernel (kH,kW,Cin,Cout) -> eqx Conv2d (Cout,Cin,kH,kW)
        self.conv_proj_0 = eqx.nn.Conv2d(
            data_dim,
            dim,
            kernel_size=patch_resolution,
            stride=patch_resolution,
            key=keys[3],
        )
        self.conv_proj_1 = eqx.nn.Conv2d(dim, dim, kernel_size=1, key=keys[4])
        self.deconv = eqx.nn.ConvTranspose2d(
            dim,
            conv_dim,
            kernel_size=patch_resolution_output,
            stride=patch_resolution_output,
            key=keys[5],
        )
        self.post_conv_0 = eqx.nn.Conv2d(conv_dim, conv_dim, kernel_size=1, key=keys[6])
        self.post_conv_1 = eqx.nn.Conv2d(conv_dim, data_dim, kernel_size=1, key=keys[7])

    def encode(self, data, times):
        bs, t, h, w, c = data.shape
        # NHWC -> NCHW for eqx conv
        x = data.reshape(bs * t, h, w, c).transpose(0, 3, 1, 2)
        x = jax.vmap(self.conv_proj_0)(x)
        x = jax.nn.gelu(x)
        x = jax.vmap(self.conv_proj_1)(x)
        # NCHW -> (bs, t, p*p, dim)
        p = self.patch_num
        x = x.transpose(0, 2, 3, 1)  # (bs*t, pH, pW, dim)
        x = x.reshape(bs, t, p * p, self.dim)

        if self.time_embed_type == "continuous":
            te = _apply_linear(self.time_proj_dense1, times)
            te = jax.nn.gelu(te)
            te = _apply_linear(self.time_proj_dense2, te)
            time_embeddings = te[:, :, None, :]
        else:
            time_embeddings = self.time_embed[:, :t, :, :]

        return (x + time_embeddings + self.patch_position_embeddings).reshape(
            bs, -1, self.dim
        )

    def decode(self, data_output):
        bs, qlen, _ = data_output.shape
        p = self.patch_num_output
        out_t = qlen // (p * p)
        x = data_output.reshape(bs * out_t, p, p, self.dim)
        x = x.transpose(0, 3, 1, 2)  # NCHW
        x = jax.vmap(self.deconv)(x)
        x = jax.nn.gelu(x)
        x = jax.vmap(self.post_conv_0)(x)
        x = jax.nn.gelu(x)
        x = jax.vmap(self.post_conv_1)(x)
        x = x.transpose(0, 2, 3, 1)  # NHWC
        h, w = x.shape[1], x.shape[2]
        return x.reshape(bs, out_t, h, w, self.data_dim)


class PROSE1to1(eqx.Module):
    """PROSE finite-difference 1-to-1 model."""

    embedder: ConvEmbedder1to1
    data_encoder: TransformerDataEncoder
    data_decoder: DataOperatorDecoder
    carry_last_frame: bool

    def __init__(
        self,
        dim_emb=1024,
        dim_ffn=2048,
        n_head=8,
        n_enc_layers=7,
        n_dec_layers=12,
        patch_num=8,
        patch_num_output=16,
        x_num=128,
        max_output_dim=4,
        output_len=1,
        norm_type="rms",
        norm_first=True,
        final_ln=True,
        time_embed_type="learnable",
        max_time_len=32,
        conv_dim=32,
        carry_last_frame=False,
        *,
        key,
    ):
        k1, k2, k3 = jax.random.split(key, 3)
        self.carry_last_frame = carry_last_frame
        self.embedder = ConvEmbedder1to1(
            dim=dim_emb,
            patch_num=patch_num,
            patch_num_output=patch_num_output,
            x_num=x_num,
            data_dim=max_output_dim,
            time_embed_type=time_embed_type,
            max_time_len=max_time_len,
            conv_dim=conv_dim,
            key=k1,
        )
        self.data_encoder = TransformerDataEncoder(
            n_layer=n_enc_layers,
            dim_emb=dim_emb,
            dim_ffn=dim_ffn,
            n_head=n_head,
            norm_type=norm_type,
            norm_first=norm_first,
            key=k2,
        )
        self.data_decoder = DataOperatorDecoder(
            dim_emb=dim_emb,
            dim_ffn=dim_ffn,
            n_head=n_head,
            n_layer=n_dec_layers,
            patch_num_output=patch_num_output,
            norm_type=norm_type,
            norm_first=norm_first,
            final_ln=final_ln,
            time_embed_type=time_embed_type,
            max_time_len=max_time_len,
            key=k3,
        )

    def __call__(self, data_input, input_times, output_times, deterministic=True):
        bs = data_input.shape[0]
        data_tokens = self.embedder.encode(data_input, input_times)
        data_encoded = self.data_encoder(data_tokens)
        query_emb = self.data_decoder.get_query_emb(output_times)
        if query_emb.shape[0] == 1 and bs > 1:
            query_emb = jnp.broadcast_to(
                query_emb, (bs, query_emb.shape[1], query_emb.shape[2])
            )
        decoded = self.data_decoder(data_encoded, query_emb, src_key_padding_mask=None)
        return self.embedder.decode(decoded)


# ═══════════════════════════════════════════════════════════════════════════════
# PROSE2to1 components (prose_fd_2to1.py)
# ═══════════════════════════════════════════════════════════════════════════════


class RMSNormScale(eqx.Module):
    """RMSNorm with learnable scale (prose_fd_2to1)."""

    scale: jnp.ndarray
    eps: float

    def __init__(self, dim, eps=1e-5, *, key=None):
        self.scale = jnp.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        var = jnp.mean(x * x, axis=-1, keepdims=True)
        return x * jnp.reciprocal(jnp.sqrt(var + self.eps)) * self.scale


class CustomMHA(eqx.Module):
    """Custom multi-head attention for PROSE2to1."""

    dim: int
    n_head: int
    linear_q: eqx.nn.Linear
    linear_k: eqx.nn.Linear
    linear_v: eqx.nn.Linear
    out_proj: eqx.nn.Linear

    def __init__(self, dim, n_head, *, key):
        self.dim = dim
        self.n_head = n_head
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.linear_q = eqx.nn.Linear(dim, dim, key=k1)
        self.linear_k = eqx.nn.Linear(dim, dim, key=k2)
        self.linear_v = eqx.nn.Linear(dim, dim, key=k3)
        self.out_proj = eqx.nn.Linear(dim, dim, key=k4)

    def __call__(self, q_in, k_in, v_in, key_padding_mask=None):
        bs, q_len, _ = q_in.shape
        k_len = k_in.shape[1]
        h = self.n_head
        d = self.dim // h
        q = (
            _apply_linear(self.linear_q, q_in)
            .reshape(bs, q_len, h, d)
            .transpose(0, 2, 1, 3)
        )
        k = (
            _apply_linear(self.linear_k, k_in)
            .reshape(bs, k_len, h, d)
            .transpose(0, 2, 1, 3)
        )
        v = (
            _apply_linear(self.linear_v, v_in)
            .reshape(bs, k_len, h, d)
            .transpose(0, 2, 1, 3)
        )
        score = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(float(d))
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]
            score = jnp.where(mask, -1e30, score)
        attn = jax.nn.softmax(score, axis=-1)
        out = (
            jnp.einsum("bhqk,bhkd->bhqd", attn, v)
            .transpose(0, 2, 1, 3)
            .reshape(bs, q_len, self.dim)
        )
        return _apply_linear(self.out_proj, out)


class EncoderLayer2to1(eqx.Module):
    self_attn: CustomMHA
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    norm1: RMSNormScale
    norm2: RMSNormScale

    def __init__(self, dim, dim_ffn, n_head, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.self_attn = CustomMHA(dim, n_head, key=k1)
        self.linear1 = eqx.nn.Linear(dim, dim_ffn, key=k2)
        self.linear2 = eqx.nn.Linear(dim_ffn, dim, key=k3)
        self.norm1 = RMSNormScale(dim)
        self.norm2 = RMSNormScale(dim)

    def __call__(self, x, key_padding_mask=None):
        y = self.norm1(x)
        x = x + self.self_attn(y, y, y, key_padding_mask=key_padding_mask)
        y = self.norm2(x)
        y = _apply_linear(
            self.linear2, jax.nn.gelu(_apply_linear(self.linear1, y), approximate=False)
        )
        return x + y


class Encoder2to1(eqx.Module):
    layers: list
    norm: RMSNormScale

    def __init__(self, n_layer, dim, dim_ffn, n_head, *, key):
        keys = jax.random.split(key, n_layer + 1)
        self.layers = [
            EncoderLayer2to1(dim, dim_ffn, n_head, key=keys[i]) for i in range(n_layer)
        ]
        self.norm = RMSNormScale(dim)

    def __call__(self, x, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.norm(x)


class SymbolEncoder(eqx.Module):
    word_embeddings: eqx.nn.Embedding
    pe: jnp.ndarray
    transformer_encoder: Encoder2to1

    def __init__(self, n_words, dim, dim_ffn, n_head, *, key):
        k1, k2 = jax.random.split(key)
        self.word_embeddings = eqx.nn.Embedding(n_words, dim, key=k1)
        self.pe = _sinusoidal_pe(1024, dim)
        self.transformer_encoder = Encoder2to1(1, dim, dim_ffn, n_head, key=k2)

    def __call__(self, x, key_padding_mask=None):
        emb = jax.vmap(jax.vmap(self.word_embeddings))(x)
        emb = emb + jnp.transpose(self.pe[: emb.shape[1]], (1, 0, 2))
        return self.transformer_encoder(emb, key_padding_mask=key_padding_mask)


class Fusion2to1(eqx.Module):
    type_embeddings: eqx.nn.Embedding
    transformer_encoder: Encoder2to1

    def __init__(self, dim, dim_ffn, n_head, n_layer, *, key):
        k1, k2 = jax.random.split(key)
        self.type_embeddings = eqx.nn.Embedding(2, dim, key=k1)
        self.transformer_encoder = Encoder2to1(n_layer, dim, dim_ffn, n_head, key=k2)

    def __call__(self, x0, x1, key_padding_mask1=None):
        t0 = self.type_embeddings.weight[0][None, None, :]  # (1, 1, dim)
        t1 = self.type_embeddings.weight[1][None, None, :]  # (1, 1, dim)
        x0 = x0 + t0
        x1 = x1 + t1
        x = jnp.concatenate([x0, x1], axis=1)
        fused_mask = None
        if key_padding_mask1 is not None:
            z = jnp.zeros((x0.shape[0], x0.shape[1]), dtype=bool)
            fused_mask = jnp.concatenate([z, key_padding_mask1], axis=1)
        return self.transformer_encoder(x, key_padding_mask=fused_mask), fused_mask


class OperatorDecoderLayer2to1(eqx.Module):
    multihead_attn: CustomMHA
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    norm1: RMSNormScale
    norm2: RMSNormScale

    def __init__(self, dim, dim_ffn, n_head, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.multihead_attn = CustomMHA(dim, n_head, key=k1)
        self.linear1 = eqx.nn.Linear(dim, dim_ffn, key=k2)
        self.linear2 = eqx.nn.Linear(dim_ffn, dim, key=k3)
        self.norm1 = RMSNormScale(dim)
        self.norm2 = RMSNormScale(dim)

    def __call__(self, q, mem, mem_mask=None):
        y = self.norm1(q)
        q = q + self.multihead_attn(y, mem, mem, key_padding_mask=mem_mask)
        y = self.norm2(q)
        y = _apply_linear(
            self.linear2, jax.nn.gelu(_apply_linear(self.linear1, y), approximate=False)
        )
        return q + y


class DataDecoder2to1(eqx.Module):
    dim: int
    patch_num_output: int
    time_embed: jnp.ndarray
    patch_position_embeddings: jnp.ndarray
    layers: list
    norm: RMSNormScale

    def __init__(
        self, dim, dim_ffn, n_head, n_layer, patch_num_output, max_time_len=10, *, key
    ):
        self.dim = dim
        self.patch_num_output = patch_num_output
        keys = jax.random.split(key, n_layer + 1)
        self.time_embed = jax.random.normal(keys[0], (1, max_time_len, 1, dim))
        self.patch_position_embeddings = jax.random.normal(
            keys[0], (1, 1, patch_num_output * patch_num_output, dim)
        )
        self.layers = [
            OperatorDecoderLayer2to1(dim, dim_ffn, n_head, key=keys[1 + i])
            for i in range(n_layer)
        ]
        self.norm = RMSNormScale(dim)

    def get_query_emb(self, output_times):
        bs = output_times.shape[0]
        out_len = output_times.shape[1]
        t = self.time_embed[:, :out_len]
        return (t + self.patch_position_embeddings).reshape(bs, -1, self.dim)

    def __call__(self, src, q, src_mask=None):
        x = q
        for layer in self.layers:
            x = layer(x, src, mem_mask=src_mask)
        return self.norm(x)


class ConvEmbedder2to1(eqx.Module):
    """ConvEmbedder for PROSE2to1."""

    dim: int
    patch_num: int
    patch_num_output: int
    x_num: int
    data_dim: int

    patch_position_embeddings: jnp.ndarray
    time_embed: jnp.ndarray
    conv_proj_0: eqx.nn.Conv2d
    conv_proj_1: eqx.nn.Conv2d
    deconv: eqx.nn.ConvTranspose2d
    post_conv_0: eqx.nn.Conv2d
    post_conv_1: eqx.nn.Conv2d

    def __init__(
        self, dim, patch_num, patch_num_output, x_num, data_dim, max_time_len=10, *, key
    ):
        self.dim = dim
        self.patch_num = patch_num
        self.patch_num_output = patch_num_output
        self.x_num = x_num
        self.data_dim = data_dim
        patch_resolution = x_num // patch_num
        patch_resolution_output = x_num // patch_num_output
        keys = jax.random.split(key, 7)
        self.patch_position_embeddings = jax.random.normal(
            keys[0], (1, 1, patch_num * patch_num, dim)
        )
        self.time_embed = jax.random.normal(keys[1], (1, max_time_len, 1, dim))
        self.conv_proj_0 = eqx.nn.Conv2d(
            data_dim,
            dim,
            kernel_size=patch_resolution,
            stride=patch_resolution,
            key=keys[2],
        )
        self.conv_proj_1 = eqx.nn.Conv2d(dim, dim, kernel_size=1, key=keys[3])
        self.deconv = eqx.nn.ConvTranspose2d(
            dim,
            32,
            kernel_size=patch_resolution_output,
            stride=patch_resolution_output,
            key=keys[4],
        )
        self.post_conv_0 = eqx.nn.Conv2d(32, 32, kernel_size=1, key=keys[5])
        self.post_conv_1 = eqx.nn.Conv2d(32, data_dim, kernel_size=1, key=keys[6])

    def encode(self, data_input, input_times):
        bs, t, h, w, c = data_input.shape
        x = data_input.reshape(bs * t, h, w, c).transpose(0, 3, 1, 2)
        x = jax.vmap(self.conv_proj_0)(x)
        x = jax.nn.gelu(x, approximate=False)
        x = jax.vmap(self.conv_proj_1)(x)
        p = self.patch_num
        x = x.transpose(0, 2, 3, 1).reshape(bs, t, p * p, self.dim)
        time_embeddings = self.time_embed[:, :t]
        return (x + time_embeddings + self.patch_position_embeddings).reshape(
            bs, -1, self.dim
        )

    def decode(self, data_output):
        bs, qlen, _ = data_output.shape
        p = self.patch_num_output
        out_t = qlen // (p * p)
        x = data_output.reshape(bs * out_t, p, p, self.dim).transpose(0, 3, 1, 2)
        x = jax.vmap(self.deconv)(x)
        x = jax.nn.gelu(x, approximate=False)
        x = jax.vmap(self.post_conv_0)(x)
        x = jax.nn.gelu(x, approximate=False)
        x = jax.vmap(self.post_conv_1)(x)
        x = x.transpose(0, 2, 3, 1)
        return x.reshape(bs, out_t, self.x_num, self.x_num, self.data_dim)


class PROSE2to1(eqx.Module):
    """PROSE finite-difference 2-to-1 model."""

    embedder: ConvEmbedder2to1
    data_encoder: Encoder2to1
    symbol_encoder: SymbolEncoder
    fusion: Fusion2to1
    data_decoder: DataDecoder2to1

    def __init__(
        self,
        n_words,
        x_num=128,
        max_output_dim=4,
        dim_emb=1024,
        dim_ffn=2048,
        n_head=8,
        patch_num=8,
        patch_num_output=16,
        data_encoder_layers=2,
        symbol_encoder_layers=1,
        fusion_layers=8,
        data_decoder_layers=8,
        *,
        key,
    ):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.embedder = ConvEmbedder2to1(
            dim=dim_emb,
            patch_num=patch_num,
            patch_num_output=patch_num_output,
            x_num=x_num,
            data_dim=max_output_dim,
            key=k1,
        )
        self.data_encoder = Encoder2to1(
            data_encoder_layers, dim_emb, dim_ffn, n_head, key=k2
        )
        self.symbol_encoder = SymbolEncoder(n_words, dim_emb, dim_ffn, n_head, key=k3)
        self.fusion = Fusion2to1(dim_emb, dim_ffn, n_head, fusion_layers, key=k4)
        self.data_decoder = DataDecoder2to1(
            dim_emb, dim_ffn, n_head, data_decoder_layers, patch_num_output, key=k5
        )

    def __call__(
        self,
        data_input,
        input_times,
        output_times,
        symbol_input,
        symbol_padding_mask=None,
    ):
        bs = data_input.shape[0]
        data_input = self.embedder.encode(data_input, input_times)
        data_encoded = self.data_encoder(data_input)
        symbol_encoded = self.symbol_encoder(
            symbol_input, key_padding_mask=symbol_padding_mask
        )
        fused, fused_mask = self.fusion(
            data_encoded, symbol_encoded, key_padding_mask1=symbol_padding_mask
        )
        q = self.data_decoder.get_query_emb(output_times)
        if q.shape[0] == 1 and bs > 1:
            q = jnp.broadcast_to(q, (bs, q.shape[1], q.shape[2]))
        dec = self.data_decoder(fused, q, src_mask=fused_mask)
        return self.embedder.decode(dec)


# ═══════════════════════════════════════════════════════════════════════════════
# PROSE ODE/PDE 2to1 components (prose_ode_pde_2to1.py)
# ═══════════════════════════════════════════════════════════════════════════════

N_MAX_POSITIONS = 512


class TorchLikeMHA(eqx.Module):
    """Multi-head attention for ODE/PDE models."""

    dim: int
    n_head: int
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear

    def __init__(self, dim, n_head, *, key):
        self.dim = dim
        self.n_head = n_head
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.q_proj = eqx.nn.Linear(dim, dim, key=k1)
        self.k_proj = eqx.nn.Linear(dim, dim, key=k2)
        self.v_proj = eqx.nn.Linear(dim, dim, key=k3)
        self.out_proj = eqx.nn.Linear(dim, dim, key=k4)

    def __call__(self, query, key_value=None, key_padding_mask=None):
        if key_value is None:
            key_value = query
        bs, q_len, _ = query.shape
        k_len = key_value.shape[1]
        h = self.n_head
        d = self.dim // h
        q = (
            _apply_linear(self.q_proj, query)
            .reshape(bs, q_len, h, d)
            .transpose(0, 2, 1, 3)
        )
        k = (
            _apply_linear(self.k_proj, key_value)
            .reshape(bs, k_len, h, d)
            .transpose(0, 2, 1, 3)
        )
        v = (
            _apply_linear(self.v_proj, key_value)
            .reshape(bs, k_len, h, d)
            .transpose(0, 2, 1, 3)
        )
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(float(d))
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]
            scores = jnp.where(mask, -1e30, scores)
        w = jax.nn.softmax(scores, axis=-1)
        out = (
            jnp.einsum("bhqk,bhkd->bhqd", w, v)
            .transpose(0, 2, 1, 3)
            .reshape(bs, q_len, self.dim)
        )
        return _apply_linear(self.out_proj, out)


class TransformerFFN(eqx.Module):
    lin1: eqx.nn.Linear
    mid: list
    lin2: eqx.nn.Linear

    def __init__(self, in_dim, hidden_dim, out_dim, n_hidden_layers, *, key):
        keys = jax.random.split(key, 2 + max(0, n_hidden_layers - 1))
        self.lin1 = eqx.nn.Linear(in_dim, hidden_dim, key=keys[0])
        self.mid = [
            eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1 + i])
            for i in range(max(0, n_hidden_layers - 1))
        ]
        self.lin2 = eqx.nn.Linear(hidden_dim, out_dim, key=keys[-1])

    def __call__(self, x):
        x = jax.nn.gelu(_apply_linear(self.lin1, x), approximate=False)
        for m in self.mid:
            x = jax.nn.gelu(_apply_linear(m, x), approximate=False)
        return _apply_linear(self.lin2, x)


class FusionTransformerModel(eqx.Module):
    dim: int
    n_layers: int
    type_embeddings: Optional[eqx.nn.Embedding]
    layer_norm_emb: eqx.nn.LayerNorm
    attentions: list
    layer_norm1: list
    ffns: list
    layer_norm2: list

    def __init__(
        self, dim, n_layers, n_heads, n_hidden_layers, use_type_embeddings=True, *, key
    ):
        self.dim = dim
        self.n_layers = n_layers
        keys = jax.random.split(key, 2 * n_layers + 3)
        self.type_embeddings = (
            eqx.nn.Embedding(2, dim, key=keys[0]) if use_type_embeddings else None
        )
        self.layer_norm_emb = eqx.nn.LayerNorm(dim, eps=1e-12)
        ki = 1
        self.attentions = [
            TorchLikeMHA(dim, n_heads, key=keys[ki + i]) for i in range(n_layers)
        ]
        ki += n_layers
        self.ffns = [
            TransformerFFN(dim, dim * 4, dim, n_hidden_layers, key=keys[ki + i])
            for i in range(n_layers)
        ]
        ki += n_layers
        self.layer_norm1 = [eqx.nn.LayerNorm(dim, eps=1e-12) for _ in range(n_layers)]
        self.layer_norm2 = [eqx.nn.LayerNorm(dim, eps=1e-12) for _ in range(n_layers)]

    def _ln(self, ln, x):
        return jax.vmap(jax.vmap(ln))(x)

    def __call__(self, x_data, x_text, lengths_data, lengths_text):
        x_data = jnp.transpose(x_data, (1, 0, 2))
        x_text = jnp.transpose(x_text, (1, 0, 2))
        data_len = x_data.shape[1]
        text_len = x_text.shape[1]

        if self.type_embeddings is not None:
            t0 = self.type_embeddings.weight[0][None, None, :]  # (1, 1, dim)
            t1 = self.type_embeddings.weight[1][None, None, :]  # (1, 1, dim)
            x_data = x_data + t0
            x_text = x_text + t1

        x = jnp.concatenate([x_data, x_text], axis=1)
        mask_data = _lengths_to_mask(lengths_data, data_len)
        mask_text = _lengths_to_mask(lengths_text, text_len)
        valid = jnp.concatenate([mask_data, mask_text], axis=1)
        pad_mask = ~valid

        x = self._ln(self.layer_norm_emb, x)
        x = x * valid[:, :, None].astype(x.dtype)

        for i in range(self.n_layers):
            a = self.attentions[i](x, key_padding_mask=pad_mask)
            x = self._ln(self.layer_norm1[i], x + a)
            x = self._ln(self.layer_norm2[i], x + self.ffns[i](x))
            x = x * valid[:, :, None].astype(x.dtype)

        return jnp.transpose(x, (1, 0, 2))


class DataTransformerModel(eqx.Module):
    dim: int
    n_layers: int
    pos_type: Optional[str]
    position_embeddings: Optional[eqx.nn.Embedding]
    pos_sin: Optional[jnp.ndarray]
    layer_norm_emb: eqx.nn.LayerNorm
    attentions: list
    layer_norm1: list
    ffns: list
    layer_norm2: list

    def __init__(
        self,
        dim,
        n_layers,
        n_heads,
        n_hidden_layers,
        positional_embeddings=None,
        *,
        key,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.pos_type = positional_embeddings
        keys = jax.random.split(key, 2 * n_layers + 3)
        if positional_embeddings == "learnable":
            self.position_embeddings = eqx.nn.Embedding(
                N_MAX_POSITIONS, dim, key=keys[0]
            )
            self.pos_sin = None
        elif positional_embeddings == "sinusoidal":
            self.position_embeddings = None
            self.pos_sin = _sinusoidal_embedding(N_MAX_POSITIONS, dim)
        else:
            self.position_embeddings = None
            self.pos_sin = None
        self.layer_norm_emb = eqx.nn.LayerNorm(dim, eps=1e-12)
        ki = 1
        self.attentions = [
            TorchLikeMHA(dim, n_heads, key=keys[ki + i]) for i in range(n_layers)
        ]
        ki += n_layers
        self.ffns = [
            TransformerFFN(dim, dim * 4, dim, n_hidden_layers, key=keys[ki + i])
            for i in range(n_layers)
        ]
        ki += n_layers
        self.layer_norm1 = [eqx.nn.LayerNorm(dim, eps=1e-12) for _ in range(n_layers)]
        self.layer_norm2 = [eqx.nn.LayerNorm(dim, eps=1e-12) for _ in range(n_layers)]

    def _ln(self, ln, x):
        return jax.vmap(jax.vmap(ln))(x)

    def __call__(self, x, lengths):
        slen = x.shape[0]
        x = jnp.transpose(x, (1, 0, 2))
        valid = _lengths_to_mask(lengths, slen)
        pad_mask = ~valid
        if self.position_embeddings is not None:
            pos = jnp.arange(slen, dtype=jnp.int32)[None, :]
            x = x + jax.vmap(jax.vmap(self.position_embeddings))(pos)
        elif self.pos_sin is not None:
            x = x + self.pos_sin[None, :slen, :]
        x = self._ln(self.layer_norm_emb, x)
        x = x * valid[:, :, None].astype(x.dtype)
        for i in range(self.n_layers):
            a = self.attentions[i](x, key_padding_mask=pad_mask)
            x = self._ln(self.layer_norm1[i], x + a)
            x = self._ln(self.layer_norm2[i], x + self.ffns[i](x))
            x = x * valid[:, :, None].astype(x.dtype)
        return jnp.transpose(x, (1, 0, 2))


class TextTransformerModel(eqx.Module):
    n_words: int
    pad_index: int
    dim: int
    n_layers: int
    pos_type: Optional[str]
    embeddings: eqx.nn.Embedding
    position_embeddings: Optional[eqx.nn.Embedding]
    pos_sin: Optional[jnp.ndarray]
    layer_norm_emb: eqx.nn.LayerNorm
    attentions: list
    layer_norm1: list
    ffns: list
    layer_norm2: list

    def __init__(
        self,
        n_words,
        pad_index,
        dim,
        n_layers,
        n_heads,
        n_hidden_layers,
        positional_embeddings="sinusoidal",
        *,
        key,
    ):
        self.n_words = n_words
        self.pad_index = pad_index
        self.dim = dim
        self.n_layers = n_layers
        self.pos_type = positional_embeddings
        keys = jax.random.split(key, 2 * n_layers + 5)
        self.embeddings = eqx.nn.Embedding(n_words, dim, key=keys[0])
        if positional_embeddings == "learnable":
            self.position_embeddings = eqx.nn.Embedding(
                N_MAX_POSITIONS, dim, key=keys[1]
            )
            self.pos_sin = None
        elif positional_embeddings == "sinusoidal":
            self.position_embeddings = None
            self.pos_sin = _sinusoidal_embedding(N_MAX_POSITIONS, dim)
        else:
            self.position_embeddings = None
            self.pos_sin = None
        self.layer_norm_emb = eqx.nn.LayerNorm(dim, eps=1e-12)
        ki = 2
        self.attentions = [
            TorchLikeMHA(dim, n_heads, key=keys[ki + i]) for i in range(n_layers)
        ]
        ki += n_layers
        self.ffns = [
            TransformerFFN(dim, dim * 4, dim, n_hidden_layers, key=keys[ki + i])
            for i in range(n_layers)
        ]
        ki += n_layers
        self.layer_norm1 = [eqx.nn.LayerNorm(dim, eps=1e-12) for _ in range(n_layers)]
        self.layer_norm2 = [eqx.nn.LayerNorm(dim, eps=1e-12) for _ in range(n_layers)]

    def _ln(self, ln, x):
        return jax.vmap(jax.vmap(ln))(x)

    def __call__(self, x, lengths):
        slen, bs = x.shape
        tok = jnp.transpose(x, (1, 0))
        h = jax.vmap(jax.vmap(self.embeddings))(tok)
        valid = _lengths_to_mask(lengths, slen)
        pad_mask = ~valid
        if self.position_embeddings is not None:
            pos = jnp.arange(slen, dtype=jnp.int32)[None, :]
            h = h + jax.vmap(jax.vmap(self.position_embeddings))(pos)
        elif self.pos_sin is not None:
            h = h + self.pos_sin[None, :slen, :]
        h = self._ln(self.layer_norm_emb, h)
        h = h * valid[:, :, None].astype(h.dtype)
        for i in range(self.n_layers):
            a = self.attentions[i](h, key_padding_mask=pad_mask)
            h = self._ln(self.layer_norm1[i], h + a)
            h = self._ln(self.layer_norm2[i], h + self.ffns[i](h))
            h = h * valid[:, :, None].astype(h.dtype)
        return jnp.transpose(h, (1, 0, 2))


class DataOperatorModel(eqx.Module):
    dim: int
    n_layers: int
    max_output_dimension: int
    split_fused_feature_data: bool
    no_text: bool
    data_feature_resnet: bool
    data_decoder_attn: bool
    pos_type: Optional[str]
    x_grid_size: int
    two_layer_proj: bool

    query_embedder: eqx.nn.Linear
    position_embeddings: Optional[eqx.nn.Embedding]
    pos_sin: Optional[jnp.ndarray]
    layer_norm_emb: eqx.nn.LayerNorm
    data_embedder_0: Optional[eqx.nn.Linear]
    data_embedder_2: Optional[eqx.nn.Linear]
    text_embedder_0: Optional[eqx.nn.Linear]
    text_embedder_2: Optional[eqx.nn.Linear]
    attentions: list
    layer_norm1: list
    encoder_attn: Optional[list]
    layer_norm15: Optional[list]
    ffns: list
    layer_norm2: list
    proj: Optional[eqx.nn.Linear]
    proj_0: Optional[eqx.nn.Linear]
    proj_1: Optional[eqx.nn.Linear]

    def __init__(
        self,
        dim,
        n_layers,
        n_heads,
        n_hidden_layers,
        max_output_dimension,
        split_fused_feature_data=True,
        no_text=False,
        data_feature_resnet=False,
        data_decoder_attn=False,
        positional_embeddings=None,
        x_grid_size=1,
        two_layer_proj=False,
        *,
        key,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.max_output_dimension = max_output_dimension
        self.split_fused_feature_data = split_fused_feature_data
        self.no_text = no_text
        self.data_feature_resnet = data_feature_resnet
        self.data_decoder_attn = data_decoder_attn
        self.pos_type = positional_embeddings
        self.x_grid_size = x_grid_size
        self.two_layer_proj = two_layer_proj
        hidden_dim = dim * 4

        keys = jax.random.split(key, 20 + 2 * n_layers)
        ki = 0
        self.query_embedder = eqx.nn.Linear(1, dim, key=keys[ki])
        ki += 1
        if positional_embeddings == "learnable":
            self.position_embeddings = eqx.nn.Embedding(
                N_MAX_POSITIONS, dim, key=keys[ki]
            )
            ki += 1
            self.pos_sin = None
        elif positional_embeddings == "sinusoidal":
            self.position_embeddings = None
            self.pos_sin = _sinusoidal_embedding(N_MAX_POSITIONS, dim)
        else:
            self.position_embeddings = None
            self.pos_sin = None

        self.layer_norm_emb = eqx.nn.LayerNorm(dim, eps=1e-12)

        if data_feature_resnet:
            self.data_embedder_0 = eqx.nn.Linear(dim, dim * 2, key=keys[ki])
            ki += 1
            self.data_embedder_2 = eqx.nn.Linear(dim * 2, dim, key=keys[ki])
            ki += 1
            if not no_text and not split_fused_feature_data:
                self.text_embedder_0 = eqx.nn.Linear(dim, dim * 2, key=keys[ki])
                ki += 1
                self.text_embedder_2 = eqx.nn.Linear(dim * 2, dim, key=keys[ki])
                ki += 1
            else:
                self.text_embedder_0 = None
                self.text_embedder_2 = None
        else:
            self.data_embedder_0 = None
            self.data_embedder_2 = None
            self.text_embedder_0 = None
            self.text_embedder_2 = None

        self.attentions = [
            TorchLikeMHA(dim, n_heads, key=keys[ki + i]) for i in range(n_layers)
        ]
        ki += n_layers
        self.layer_norm1 = [eqx.nn.LayerNorm(dim, eps=1e-12) for _ in range(n_layers)]
        if data_decoder_attn:
            self.encoder_attn = [
                TorchLikeMHA(dim, n_heads, key=keys[ki + i]) for i in range(n_layers)
            ]
            ki += n_layers
            self.layer_norm15 = [
                eqx.nn.LayerNorm(dim, eps=1e-12) for _ in range(n_layers)
            ]
        else:
            self.encoder_attn = None
            self.layer_norm15 = None
        self.ffns = [
            TransformerFFN(dim, hidden_dim, dim, n_hidden_layers, key=keys[ki + i])
            for i in range(n_layers)
        ]
        ki += n_layers
        self.layer_norm2 = [eqx.nn.LayerNorm(dim, eps=1e-12) for _ in range(n_layers)]

        out_dim = max_output_dimension * x_grid_size
        if two_layer_proj:
            self.proj = None
            self.proj_0 = eqx.nn.Linear(dim, hidden_dim, key=keys[ki])
            ki += 1
            self.proj_1 = eqx.nn.Linear(hidden_dim, out_dim, key=keys[ki])
            ki += 1
        else:
            self.proj = eqx.nn.Linear(dim, out_dim, key=keys[ki])
            ki += 1
            self.proj_0 = None
            self.proj_1 = None

    def _ln(self, ln, x):
        return jax.vmap(jax.vmap(ln))(x)

    def get_query_emb(self, query_times):
        return _apply_linear(self.query_embedder, query_times[:, None])

    def _apply_proj(self, x):
        if self.two_layer_proj:
            return _apply_linear(self.proj_1, _apply_linear(self.proj_0, x))
        return _apply_linear(self.proj, x)

    def forward_hidden(self, query_emb, src_enc, src_len):
        bs = src_enc.shape[0]
        q = jnp.broadcast_to(query_emb[None, :, :], (bs, query_emb.shape[0], self.dim))
        src_data_len, src_text_len = src_len
        max_data = src_enc.shape[1] if self.no_text else int(jnp.max(src_data_len))
        src_data_mask = _lengths_to_mask(src_data_len, max_data)

        src = src_enc
        if self.no_text:
            src_mask = src_data_mask
            if self.data_feature_resnet:
                d = jax.nn.gelu(
                    _apply_linear(self.data_embedder_0, src), approximate=False
                )
                src = src + _apply_linear(self.data_embedder_2, d)
        else:
            if self.split_fused_feature_data:
                src = src_enc[:, :max_data, :]
                src_mask = src_data_mask
                if self.data_feature_resnet:
                    d = jax.nn.gelu(
                        _apply_linear(self.data_embedder_0, src), approximate=False
                    )
                    src = src + _apply_linear(self.data_embedder_2, d)
            else:
                text_max = src_enc.shape[1] - max_data
                src_text_mask = _lengths_to_mask(src_text_len, text_max)
                src_mask = jnp.concatenate([src_data_mask, src_text_mask], axis=1)
                if self.data_feature_resnet:
                    src_d = src_enc[:, :max_data, :]
                    src_t = src_enc[:, max_data:, :]
                    src_d = src_d + _apply_linear(
                        self.data_embedder_2,
                        jax.nn.gelu(
                            _apply_linear(self.data_embedder_0, src_d),
                            approximate=False,
                        ),
                    )
                    src_t = src_t + _apply_linear(
                        self.text_embedder_2,
                        jax.nn.gelu(
                            _apply_linear(self.text_embedder_0, src_t),
                            approximate=False,
                        ),
                    )
                    src = jnp.concatenate([src_d, src_t], axis=1)

        src_pad = ~src_mask
        if self.position_embeddings is not None:
            pos = jnp.arange(q.shape[1], dtype=jnp.int32)[None, :]
            q = q + jax.vmap(jax.vmap(self.position_embeddings))(pos)
        elif self.pos_sin is not None:
            q = q + self.pos_sin[None, : q.shape[1], :]
        h = self._ln(self.layer_norm_emb, q)
        for i in range(self.n_layers):
            a = self.attentions[i](h, key_value=src, key_padding_mask=src_pad)
            h = self._ln(self.layer_norm1[i], h + a)
            if self.data_decoder_attn:
                aa = self.encoder_attn[i](h)
                h = self._ln(self.layer_norm15[i], h + aa)
            h = self._ln(self.layer_norm2[i], h + self.ffns[i](h))
        return jnp.transpose(h, (1, 0, 2))

    def generate(self, src_enc, src_len, query_emb):
        hidden = self.forward_hidden(query_emb, src_enc, src_len)
        return self._apply_proj(hidden)


class RevIN(eqx.Module):
    gamma: jnp.ndarray
    beta: jnp.ndarray

    def __init__(self, dim, *, key=None):
        self.gamma = jnp.ones((dim,))
        self.beta = jnp.zeros((dim,))

    def __call__(self, x, eps=1e-6):
        y = x[:, :, 1:]
        mu = jnp.mean(y, axis=0, keepdims=True)
        var = jnp.var(y, axis=0, keepdims=True)
        yhat = (y - mu) / jnp.sqrt(var + eps)
        yout = yhat * self.gamma[None, None, :] + self.beta[None, None, :]
        xout = x.at[:, :, 1:].set(yout)
        return xout, mu, var

    def reverse(self, y, mu, var, eps=1e-6):
        yhat = (y - self.beta[None, None, :]) / self.gamma[None, None, :]
        return yhat * jnp.sqrt(var + eps) + mu


class PROSEODE2to1(eqx.Module):
    """PROSE ODE 2-to-1 model."""

    cfg_no_text: bool
    embedder_0: eqx.nn.Linear
    embedder_2: eqx.nn.Linear
    data_encoder: DataTransformerModel
    text_encoder: TextTransformerModel
    fusion: FusionTransformerModel
    data_decoder: DataOperatorModel

    def __init__(
        self,
        n_words,
        pad_index,
        max_output_dimension=3,
        emb_dim=512,
        n_text_enc_layers=4,
        n_data_enc_layers=2,
        n_data_dec_layers=8,
        n_fusion_layers=8,
        n_text_heads=8,
        n_data_heads=8,
        n_fusion_heads=8,
        n_text_hidden_layers=1,
        n_data_hidden_layers=1,
        n_fusion_hidden_layers=1,
        split_fused_feature_data=True,
        data_feature_resnet=False,
        data_decoder_attn=False,
        no_text=False,
        text_positional_embeddings="sinusoidal",
        data_positional_embeddings=None,
        data_decoder_positional_embeddings=None,
        fusion_type_embeddings=True,
        *,
        key,
    ):
        self.cfg_no_text = no_text
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        self.embedder_0 = eqx.nn.Linear(1 + max_output_dimension, emb_dim, key=k1)
        self.embedder_2 = eqx.nn.Linear(emb_dim, emb_dim, key=k2)
        self.data_encoder = DataTransformerModel(
            emb_dim,
            n_data_enc_layers,
            n_data_heads,
            n_data_hidden_layers,
            positional_embeddings=data_positional_embeddings,
            key=k3,
        )
        self.text_encoder = TextTransformerModel(
            n_words,
            pad_index,
            emb_dim,
            n_text_enc_layers,
            n_text_heads,
            n_text_hidden_layers,
            positional_embeddings=text_positional_embeddings,
            key=k4,
        )
        self.fusion = FusionTransformerModel(
            emb_dim,
            n_fusion_layers,
            n_fusion_heads,
            n_fusion_hidden_layers,
            use_type_embeddings=fusion_type_embeddings,
            key=k5,
        )
        self.data_decoder = DataOperatorModel(
            emb_dim,
            n_data_dec_layers,
            n_data_heads,
            n_data_hidden_layers,
            max_output_dimension,
            split_fused_feature_data=split_fused_feature_data,
            no_text=no_text,
            data_feature_resnet=data_feature_resnet,
            data_decoder_attn=data_decoder_attn,
            positional_embeddings=data_decoder_positional_embeddings,
            x_grid_size=1,
            two_layer_proj=False,
            key=k6,
        )

    def __call__(self, data_input, data_lengths, query_times, text_input, text_lengths):
        x = jax.nn.gelu(_apply_linear(self.embedder_0, data_input), approximate=False)
        x = _apply_linear(self.embedder_2, x)
        data_encoded = self.data_encoder(x, data_lengths)
        if self.cfg_no_text:
            fused = data_encoded
            txt_len = jnp.zeros_like(data_lengths)
        else:
            text_encoded = self.text_encoder(text_input, text_lengths)
            fused = self.fusion(data_encoded, text_encoded, data_lengths, text_lengths)
            txt_len = text_lengths
        fused_bs = jnp.transpose(fused, (1, 0, 2))
        q = self.data_decoder.get_query_emb(query_times)
        return self.data_decoder.generate(fused_bs, (data_lengths, txt_len), q)


class PROSEPDE2to1(eqx.Module):
    """PROSE PDE 2-to-1 model."""

    cfg_no_text: bool
    use_normalizer: bool
    normalizer: Optional[RevIN]
    embedder_0: eqx.nn.Linear
    embedder_2: eqx.nn.Linear
    data_encoder: DataTransformerModel
    text_encoder: TextTransformerModel
    fusion: FusionTransformerModel
    data_decoder: DataOperatorModel

    def __init__(
        self,
        n_words,
        pad_index,
        max_output_dimension=1,
        emb_dim=512,
        n_text_enc_layers=4,
        n_data_enc_layers=2,
        n_data_dec_layers=8,
        n_fusion_layers=8,
        n_text_heads=8,
        n_data_heads=8,
        n_fusion_heads=8,
        n_text_hidden_layers=1,
        n_data_hidden_layers=1,
        n_fusion_hidden_layers=1,
        split_fused_feature_data=True,
        data_feature_resnet=False,
        data_decoder_attn=False,
        no_text=False,
        text_positional_embeddings="sinusoidal",
        data_positional_embeddings=None,
        data_decoder_positional_embeddings=None,
        fusion_type_embeddings=True,
        x_patch_size=1,
        x_grid_size=1,
        normalization=False,
        *,
        key,
    ):
        self.cfg_no_text = no_text
        self.use_normalizer = normalization
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
        in_dim = 1 + max_output_dimension * x_patch_size
        self.normalizer = (
            RevIN(max_output_dimension * x_patch_size) if normalization else None
        )
        self.embedder_0 = eqx.nn.Linear(in_dim, emb_dim, key=k1)
        self.embedder_2 = eqx.nn.Linear(emb_dim, emb_dim, key=k2)
        self.data_encoder = DataTransformerModel(
            emb_dim,
            n_data_enc_layers,
            n_data_heads,
            n_data_hidden_layers,
            positional_embeddings=data_positional_embeddings,
            key=k3,
        )
        self.text_encoder = TextTransformerModel(
            n_words,
            pad_index,
            emb_dim,
            n_text_enc_layers,
            n_text_heads,
            n_text_hidden_layers,
            positional_embeddings=text_positional_embeddings,
            key=k4,
        )
        self.fusion = FusionTransformerModel(
            emb_dim,
            n_fusion_layers,
            n_fusion_heads,
            n_fusion_hidden_layers,
            use_type_embeddings=fusion_type_embeddings,
            key=k5,
        )
        self.data_decoder = DataOperatorModel(
            emb_dim,
            n_data_dec_layers,
            n_data_heads,
            n_data_hidden_layers,
            max_output_dimension,
            split_fused_feature_data=split_fused_feature_data,
            no_text=no_text,
            data_feature_resnet=data_feature_resnet,
            data_decoder_attn=data_decoder_attn,
            positional_embeddings=data_decoder_positional_embeddings,
            x_grid_size=x_grid_size,
            two_layer_proj=True,
            key=k6,
        )

    def __call__(self, data_input, data_lengths, query_times, text_input, text_lengths):
        x = data_input
        mu, var = None, None
        if self.normalizer is not None:
            x, mu, var = self.normalizer(x)
        x = jax.nn.gelu(_apply_linear(self.embedder_0, x), approximate=False)
        x = _apply_linear(self.embedder_2, x)
        data_encoded = self.data_encoder(x, data_lengths)
        if self.cfg_no_text:
            fused = data_encoded
            txt_len = jnp.zeros_like(data_lengths)
        else:
            text_encoded = self.text_encoder(text_input, text_lengths)
            fused = self.fusion(data_encoded, text_encoded, data_lengths, text_lengths)
            txt_len = text_lengths
        fused_bs = jnp.transpose(fused, (1, 0, 2))
        q = self.data_decoder.get_query_emb(query_times)
        out = self.data_decoder.generate(fused_bs, (data_lengths, txt_len), q)
        if self.normalizer is not None:
            out = self.normalizer.reverse(out, mu, var)
        return out
