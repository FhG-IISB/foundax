"""Equinox reimplementation of BCAT (Block Causal Transformer).

Mirrors ``model.py`` but uses ``equinox.Module`` instead of ``flax.linen``.
"""

from __future__ import annotations

from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


def _silu(x):
    return jax.nn.silu(x)


def _gelu(x):
    return jax.nn.gelu(x, approximate=False)


def _swiglu(x, gates):
    return _silu(x) * gates


def _geglu(x, gates):
    return _gelu(x) * gates


# ---------------------------------------------------------------------------
# Helper: apply eqx.nn.Linear to batched inputs
# ---------------------------------------------------------------------------


def _apply_linear(linear: eqx.nn.Linear, x: jnp.ndarray) -> jnp.ndarray:
    """Apply Linear to (..., in_features) -> (..., out_features)."""
    out = x @ linear.weight.T
    if linear.bias is not None:
        out = out + linear.bias
    return out


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(eqx.Module):
    weight: jnp.ndarray
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = jnp.ones(dim)
        self.eps = eps

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        ms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(ms + self.eps)
        return x * self.weight


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------


class FFN(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc_gate: Optional[eqx.nn.Linear]
    activation: Callable = eqx.field(static=True)
    gated: bool = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        activation: Callable = jax.nn.silu,
        gated: bool = True,
        *,
        key,
    ):
        k1, k2, k3 = jax.random.split(key, 3)
        self.fc1 = eqx.nn.Linear(dim, hidden_dim, use_bias=True, key=k1)
        self.fc2 = eqx.nn.Linear(hidden_dim, dim, use_bias=True, key=k2)
        if gated:
            self.fc_gate = eqx.nn.Linear(dim, hidden_dim, use_bias=True, key=k3)
        else:
            self.fc_gate = None
        self.activation = activation
        self.gated = gated

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = _apply_linear(self.fc1, x)

        if self.gated:
            gates = _apply_linear(self.fc_gate, x)
            h = self.activation(h) * gates
        else:
            h = self.activation(h)

        return _apply_linear(self.fc2, h)


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------


class MultiheadAttention(eqx.Module):
    embed_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    qk_norm: bool = eqx.field(static=True)

    linear_q: eqx.nn.Linear
    linear_k: eqx.nn.Linear
    linear_v: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    q_norm: Optional[eqx.nn.LayerNorm]
    k_norm: Optional[eqx.nn.LayerNorm]

    def __init__(self, embed_dim: int, num_heads: int, qk_norm: bool = False, *, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm

        self.linear_q = eqx.nn.Linear(embed_dim, embed_dim, use_bias=True, key=k1)
        self.linear_k = eqx.nn.Linear(embed_dim, embed_dim, use_bias=True, key=k2)
        self.linear_v = eqx.nn.Linear(embed_dim, embed_dim, use_bias=True, key=k3)
        self.out_proj = eqx.nn.Linear(embed_dim, embed_dim, use_bias=True, key=k4)

        head_dim = embed_dim // num_heads
        if qk_norm:
            self.q_norm = eqx.nn.LayerNorm(head_dim, eps=1e-5)
            self.k_norm = eqx.nn.LayerNorm(head_dim, eps=1e-5)
        else:
            self.q_norm = None
            self.k_norm = None

    def __call__(
        self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        bs, seq_len, _ = x.shape
        head_dim = self.embed_dim // self.num_heads

        q = _apply_linear(self.linear_q, x)
        k = _apply_linear(self.linear_k, x)
        v = _apply_linear(self.linear_v, x)

        q = q.reshape(bs, seq_len, self.num_heads, head_dim)
        k = k.reshape(bs, seq_len, self.num_heads, head_dim)
        v = v.reshape(bs, seq_len, self.num_heads, head_dim)

        if self.qk_norm:
            # LayerNorm on last dim (head_dim) — vmap over (bs, seq, heads)
            def _ln(norm, t):
                # t: (bs, seq, heads, head_dim)
                orig_shape = t.shape
                t_flat = t.reshape(-1, head_dim)
                t_flat = jax.vmap(norm)(t_flat)
                return t_flat.reshape(orig_shape)

            q = _ln(self.q_norm, q)
            k = _ln(self.k_norm, k)

        # (bs, n_heads, seq_len, head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        scale = head_dim**-0.5
        attn_weights = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scale

        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        out = jnp.matmul(attn_weights, v)

        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(bs, seq_len, self.embed_dim)
        return _apply_linear(self.out_proj, out)


# ---------------------------------------------------------------------------
# Transformer Encoder Layer & Encoder
# ---------------------------------------------------------------------------


class TransformerEncoderLayer(eqx.Module):
    self_attn: MultiheadAttention
    ffn: FFN
    norm1: eqx.Module  # RMSNorm or eqx.nn.LayerNorm
    norm2: eqx.Module
    norm_first: bool = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        activation: Callable = jax.nn.silu,
        gated: bool = True,
        norm_first: bool = True,
        qk_norm: bool = False,
        norm_type: str = "rms",
        *,
        key,
    ):
        k1, k2 = jax.random.split(key)
        self.self_attn = MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, qk_norm=qk_norm, key=k1
        )
        self.ffn = FFN(
            dim=d_model,
            hidden_dim=dim_feedforward,
            activation=activation,
            gated=gated,
            key=k2,
        )
        self.norm_first = norm_first

        if norm_type == "rms":
            self.norm1 = RMSNorm(dim=d_model, eps=1e-5)
            self.norm2 = RMSNorm(dim=d_model, eps=1e-5)
        else:
            self.norm1 = eqx.nn.LayerNorm(d_model, eps=1e-5)
            self.norm2 = eqx.nn.LayerNorm(d_model, eps=1e-5)

    def _apply_norm(self, norm, x):
        """Apply norm that expects (dim,) to batched (bs, seq, dim)."""
        if isinstance(norm, RMSNorm):
            return norm(x)  # already handles arbitrary batch dims
        # eqx.nn.LayerNorm expects (shape,) input
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        x_flat = jax.vmap(norm)(x_flat)
        return x_flat.reshape(orig_shape)

    def __call__(
        self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        if self.norm_first:
            x = x + self.self_attn(self._apply_norm(self.norm1, x), mask=mask)
            x = x + self.ffn(self._apply_norm(self.norm2, x))
        else:
            x = self._apply_norm(self.norm1, x + self.self_attn(x, mask=mask))
            x = self._apply_norm(self.norm2, x + self.ffn(x))
        return x


class TransformerEncoder(eqx.Module):
    layers: list
    norm: Optional[eqx.Module]  # RMSNorm or eqx.nn.LayerNorm or None
    norm_first: bool = eqx.field(static=True)

    def __init__(
        self,
        n_layer: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        activation: Callable = jax.nn.silu,
        gated: bool = True,
        norm_first: bool = True,
        qk_norm: bool = False,
        norm_type: str = "rms",
        *,
        key,
    ):
        self.norm_first = norm_first
        keys = jax.random.split(key, n_layer)
        self.layers = [
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation,
                gated=gated,
                norm_first=norm_first,
                qk_norm=qk_norm,
                norm_type=norm_type,
                key=keys[i],
            )
            for i in range(n_layer)
        ]

        if norm_first:
            if norm_type == "rms":
                self.norm = RMSNorm(dim=d_model, eps=1e-5)
            else:
                self.norm = eqx.nn.LayerNorm(d_model, eps=1e-5)
        else:
            self.norm = None

    def _apply_norm(self, norm, x):
        if isinstance(norm, RMSNorm):
            return norm(x)
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        x_flat = jax.vmap(norm)(x_flat)
        return x_flat.reshape(orig_shape)

    def __call__(
        self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x, mask=mask)
        if self.norm is not None:
            x = self._apply_norm(self.norm, x)
        return x


# ---------------------------------------------------------------------------
# Convolutional Embedder
# ---------------------------------------------------------------------------


class ConvEmbedder(eqx.Module):
    dim: int = eqx.field(static=True)
    data_dim: int = eqx.field(static=True)
    x_num: int = eqx.field(static=True)
    patch_num: int = eqx.field(static=True)
    patch_num_output: int = eqx.field(static=True)
    conv_dim: int = eqx.field(static=True)
    time_embed: str = eqx.field(static=True)
    max_time_len: int = eqx.field(static=True)

    # Encoder
    in_proj: eqx.nn.Conv2d
    conv_proj: eqx.nn.Conv2d

    # Time embeddings
    time_embeddings: Optional[jnp.ndarray]
    time_proj_0: Optional[eqx.nn.Linear]
    time_proj_1: Optional[eqx.nn.Linear]

    # Patch position embeddings
    patch_position_embeddings: jnp.ndarray

    # Decoder
    post_deconv: eqx.nn.ConvTranspose2d
    post_conv: eqx.nn.Conv2d
    head: eqx.nn.Conv2d

    def __init__(
        self,
        dim: int,
        data_dim: int,
        x_num: int,
        patch_num: int,
        patch_num_output: int,
        conv_dim: int = 32,
        time_embed: str = "learnable",
        max_time_len: int = 20,
        *,
        key,
    ):
        self.dim = dim
        self.data_dim = data_dim
        self.x_num = x_num
        self.patch_num = patch_num
        self.patch_num_output = patch_num_output
        self.conv_dim = conv_dim
        self.time_embed = time_embed
        self.max_time_len = max_time_len

        patch_res = x_num // patch_num
        patch_res_out = x_num // patch_num_output

        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

        # Encoder convs (NCHW)
        self.in_proj = eqx.nn.Conv2d(
            data_dim,
            dim,
            kernel_size=patch_res,
            stride=patch_res,
            padding=0,
            use_bias=True,
            key=k1,
        )
        self.conv_proj = eqx.nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            stride=1,
            padding=0,
            use_bias=True,
            key=k2,
        )

        # Time embeddings
        if time_embed == "learnable":
            self.time_embeddings = jax.random.normal(k3, (1, max_time_len, 1, dim))
            self.time_proj_0 = None
            self.time_proj_1 = None
        else:
            self.time_embeddings = None
            self.time_proj_0 = eqx.nn.Linear(1, dim, use_bias=True, key=k3)
            self.time_proj_1 = eqx.nn.Linear(dim, dim, use_bias=True, key=k4)

        # Patch position embeddings
        self.patch_position_embeddings = jax.random.normal(
            k5, (1, 1, patch_num * patch_num, dim)
        )

        # Decoder
        self.post_deconv = eqx.nn.ConvTranspose2d(
            dim,
            conv_dim,
            kernel_size=patch_res_out,
            stride=patch_res_out,
            padding=0,
            use_bias=True,
            key=k6,
        )
        self.post_conv = eqx.nn.Conv2d(
            conv_dim,
            conv_dim,
            kernel_size=1,
            padding=0,
            use_bias=True,
            key=k7,
        )
        self.head = eqx.nn.Conv2d(
            conv_dim,
            data_dim,
            kernel_size=1,
            padding=0,
            use_bias=True,
            key=k1,
        )

    def encode(self, data: jnp.ndarray, times: jnp.ndarray) -> jnp.ndarray:
        """(bs, t, x_num, x_num, data_dim) -> (bs, t * patch_num^2, dim)"""
        bs = data.shape[0]

        x = rearrange(data, "b t h w c -> (b t) h w c")
        # NHWC -> NCHW for Conv2d
        x = jnp.moveaxis(x, -1, 1)  # (bt, C, H, W)
        x = jax.vmap(self.in_proj)(x)
        x = _gelu(x)
        x = jax.vmap(self.conv_proj)(x)
        # NCHW -> NHWC
        x = jnp.moveaxis(x, 1, -1)  # (bt, H, W, D)

        x = rearrange(x, "(b t) h w d -> b t (h w) d", b=bs)

        if self.time_embed == "learnable":
            t_len = data.shape[1]
            x = x + self.time_embeddings[:, :t_len]
        else:
            te = _apply_linear(self.time_proj_0, times)
            te = _gelu(te)
            te = _apply_linear(self.time_proj_1, te)
            x = x + te[:, :, None, :]

        x = x + self.patch_position_embeddings

        x = x.reshape(bs, -1, self.dim)
        return x

    def decode(self, data_output: jnp.ndarray) -> jnp.ndarray:
        """(bs, t * patch_num_output^2, dim) -> (bs, t, x_num, x_num, data_dim)"""
        bs = data_output.shape[0]

        x = rearrange(
            data_output,
            "b (t h w) d -> (b t) h w d",
            h=self.patch_num_output,
            w=self.patch_num_output,
        )

        # NHWC -> NCHW
        x = jnp.moveaxis(x, -1, 1)
        x = jax.vmap(self.post_deconv)(x)
        x = _gelu(x)
        x = jax.vmap(self.post_conv)(x)
        x = _gelu(x)
        x = jax.vmap(self.head)(x)
        # NCHW -> NHWC
        x = jnp.moveaxis(x, 1, -1)

        x = rearrange(x, "(b t) h w c -> b t h w c", b=bs)
        return x


# ---------------------------------------------------------------------------
# Block Causal Mask
# ---------------------------------------------------------------------------


def block_lower_triangular_mask(block_size: int, block_num: int) -> jnp.ndarray:
    """Create a float attention mask for block-causal attention."""
    matrix_size = block_size * block_num
    idx = jnp.arange(matrix_size)
    lower_tri = idx[:, None] >= idx[None, :]
    block_eq = (idx[:, None] // block_size) == (idx[None, :] // block_size)
    allowed = lower_tri | block_eq
    return jnp.where(allowed, 0.0, -jnp.inf)


# ---------------------------------------------------------------------------
# BCAT
# ---------------------------------------------------------------------------


class BCAT(eqx.Module):
    n_layer: int = eqx.field(static=True)
    dim_emb: int = eqx.field(static=True)
    dim_ffn: int = eqx.field(static=True)
    n_head: int = eqx.field(static=True)
    norm_first: bool = eqx.field(static=True)
    norm_type: str = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    gated: bool = eqx.field(static=True)
    qk_norm: bool = eqx.field(static=True)
    x_num: int = eqx.field(static=True)
    max_output_dim: int = eqx.field(static=True)
    patch_num: int = eqx.field(static=True)
    patch_num_output: int = eqx.field(static=True)
    conv_dim: int = eqx.field(static=True)
    time_embed: str = eqx.field(static=True)
    max_time_len: int = eqx.field(static=True)
    max_data_len: int = eqx.field(static=True)
    deep: bool = eqx.field(static=True)

    embedder: ConvEmbedder
    transformer: TransformerEncoder

    def __init__(
        self,
        n_layer: int = 12,
        dim_emb: int = 1024,
        dim_ffn: int = 2752,
        n_head: int = 8,
        norm_first: bool = True,
        norm_type: str = "rms",
        activation: Callable = jax.nn.silu,
        gated: bool = True,
        qk_norm: bool = True,
        x_num: int = 128,
        max_output_dim: int = 4,
        patch_num: int = 16,
        patch_num_output: int = 16,
        conv_dim: int = 32,
        time_embed: str = "learnable",
        max_time_len: int = 20,
        max_data_len: int = 20,
        deep: bool = False,
        data_dim: int = 1,
        *,
        key,
    ):
        self.n_layer = n_layer
        self.dim_emb = dim_emb
        self.dim_ffn = dim_ffn
        self.n_head = n_head
        self.norm_first = norm_first
        self.norm_type = norm_type
        self.activation = activation
        self.gated = gated
        self.qk_norm = qk_norm
        self.x_num = x_num
        self.max_output_dim = max_output_dim
        self.patch_num = patch_num
        self.patch_num_output = patch_num_output
        self.conv_dim = conv_dim
        self.time_embed = time_embed
        self.max_time_len = max_time_len
        self.max_data_len = max_data_len
        self.deep = deep

        k1, k2 = jax.random.split(key)
        self.embedder = ConvEmbedder(
            dim=dim_emb,
            data_dim=data_dim,
            x_num=x_num,
            patch_num=patch_num,
            patch_num_output=patch_num_output,
            conv_dim=conv_dim,
            time_embed=time_embed,
            max_time_len=max_time_len,
            key=k1,
        )
        self.transformer = TransformerEncoder(
            n_layer=n_layer,
            d_model=dim_emb,
            nhead=n_head,
            dim_feedforward=dim_ffn,
            activation=activation,
            gated=gated,
            norm_first=norm_first,
            qk_norm=qk_norm,
            norm_type=norm_type,
            key=k2,
        )

    def __call__(
        self,
        data: jnp.ndarray,
        times: jnp.ndarray,
        input_len: int = 10,
    ) -> jnp.ndarray:
        """Training-mode forward pass.

        Args:
            data:       (bs, input_len+output_len, x_num, x_num, data_dim)
            times:      (bs, input_len+output_len, 1)
            input_len:  number of input time steps

        Returns:
            data_output: (bs, output_len, x_num, x_num, data_dim)
        """
        seq_len_per_step = self.patch_num**2

        # Autoregressive: drop last timestep
        data = data[:, :-1]
        times = times[:, :-1]

        # Encode
        encoded = self.embedder.encode(data, times)

        # Block-causal mask
        data_len = encoded.shape[1]
        mask = block_lower_triangular_mask(seq_len_per_step, self.max_data_len)
        mask = mask[:data_len, :data_len]

        # Transformer
        encoded = self.transformer(encoded, mask=mask)

        # Decode only the output steps
        input_seq_len = (input_len - 1) * seq_len_per_step
        data_output = encoded[:, input_seq_len:]

        data_output = self.embedder.decode(data_output)
        return data_output
