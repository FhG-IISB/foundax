"""JAX/Flax implementation of the BCAT (Block Causal Transformer) model.

Mirrors the PyTorch implementation in ``ogrepo/bcat/src/models/bcat.py`` and
supporting modules (``attention_utils.py``, ``embedder.py``).
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def _silu(x):
    return jax.nn.silu(x)


def _gelu(x):
    return nn.gelu(x, approximate=False)


# SwiGLU: SiLU(x) * gates   (gates is a second linear projection)
def _swiglu(x, gates):
    return _silu(x) * gates


def _geglu(x, gates):
    return _gelu(x) * gates


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMS Layer Normalisation (matches ``torch.nn.RMSNorm``)."""
    dim: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (self.dim,))
        ms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(ms + self.eps)
        return x * scale


# ---------------------------------------------------------------------------
# Feed-Forward Network (with optional GLU gating)
# ---------------------------------------------------------------------------

class FFN(nn.Module):
    """Position-wise feed-forward network.

    When *activation* is ``'swiglu'`` or ``'geglu'`` an extra gating linear
    projection is used (matching the PyTorch ``GLU`` family).
    """
    dim: int
    hidden_dim: int
    activation: str = "gelu"

    @nn.compact
    def __call__(self, x):
        use_glu = self.activation in ("swiglu", "geglu")

        h = nn.Dense(self.hidden_dim, use_bias=True, name="fc1")(x)

        if use_glu:
            gates = nn.Dense(self.hidden_dim, use_bias=True, name="fc_gate")(x)
            if self.activation == "swiglu":
                h = _swiglu(h, gates)
            else:
                h = _geglu(h, gates)
        else:
            if self.activation == "gelu":
                h = _gelu(h)
            elif self.activation == "silu":
                h = _silu(h)
            elif self.activation == "relu":
                h = jax.nn.relu(h)
            else:
                raise ValueError(f"Unknown activation: {self.activation}")

        return nn.Dense(self.dim, use_bias=True, name="fc2")(h)


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiheadAttention(nn.Module):
    """Standard multi-head attention with optional QK normalisation.

    Follows the PyTorch ``MultiheadAttention`` in ``attention_utils.py``.
    """
    embed_dim: int
    num_heads: int
    qk_norm: bool = False

    @nn.compact
    def __call__(self, x, mask=None):
        bs, seq_len, _ = x.shape
        head_dim = self.embed_dim // self.num_heads

        q = nn.Dense(self.embed_dim, use_bias=True, name="linear_q")(x)
        k = nn.Dense(self.embed_dim, use_bias=True, name="linear_k")(x)
        v = nn.Dense(self.embed_dim, use_bias=True, name="linear_v")(x)

        # (bs, seq_len, n_heads, head_dim)
        q = q.reshape(bs, seq_len, self.num_heads, head_dim)
        k = k.reshape(bs, seq_len, self.num_heads, head_dim)
        v = v.reshape(bs, seq_len, self.num_heads, head_dim)

        if self.qk_norm:
            q = nn.LayerNorm(epsilon=1e-5, name="q_norm")(q)
            k = nn.LayerNorm(epsilon=1e-5, name="k_norm")(k)

        # (bs, n_heads, seq_len, head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        scale = head_dim ** -0.5
        attn_weights = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scale

        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        out = jnp.matmul(attn_weights, v)  # (bs, n_heads, seq_len, head_dim)

        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(bs, seq_len, self.embed_dim)
        return nn.Dense(self.embed_dim, use_bias=True, name="out_proj")(out)


# ---------------------------------------------------------------------------
# Transformer Encoder Layer & Encoder
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    """Pre-norm or post-norm transformer encoder layer.

    Matches ``CustomTransformerEncoderLayer`` from the PyTorch codebase.
    """
    d_model: int
    nhead: int
    dim_feedforward: int
    activation: str = "gelu"
    norm_first: bool = True
    qk_norm: bool = False
    norm_type: str = "rms"

    def _norm(self, name: str):
        if self.norm_type == "rms":
            return RMSNorm(dim=self.d_model, eps=1e-5, name=name)
        return nn.LayerNorm(epsilon=1e-5, name=name)

    @nn.compact
    def __call__(self, x, mask=None):
        attn = MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.nhead,
            qk_norm=self.qk_norm,
            name="self_attn",
        )
        ffn = FFN(
            dim=self.d_model,
            hidden_dim=self.dim_feedforward,
            activation=self.activation,
            name="ffn",
        )
        norm1 = self._norm("norm1")
        norm2 = self._norm("norm2")

        if self.norm_first:
            x = x + attn(norm1(x), mask=mask)
            x = x + ffn(norm2(x))
        else:
            x = norm1(x + attn(x, mask=mask))
            x = norm2(x + ffn(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of ``TransformerEncoderLayer`` with optional final norm."""
    n_layer: int
    d_model: int
    nhead: int
    dim_feedforward: int
    activation: str = "gelu"
    norm_first: bool = True
    qk_norm: bool = False
    norm_type: str = "rms"

    @nn.compact
    def __call__(self, x, mask=None):
        for i in range(self.n_layer):
            x = TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                activation=self.activation,
                norm_first=self.norm_first,
                qk_norm=self.qk_norm,
                norm_type=self.norm_type,
                name=f"layers_{i}",
            )(x, mask=mask)

        if self.norm_first:
            if self.norm_type == "rms":
                x = RMSNorm(dim=self.d_model, eps=1e-5, name="norm")(x)
            else:
                x = nn.LayerNorm(epsilon=1e-5, name="norm")(x)
        return x


# ---------------------------------------------------------------------------
# Convolutional Embedder (default for bcat_auto)
# ---------------------------------------------------------------------------

class ConvEmbedder(nn.Module):
    """Convolutional patch embedder matching ``ConvEmbedder`` from PyTorch.

    Uses ``setup()`` so that ``encode`` / ``decode`` work as plain methods.
    """
    dim: int
    data_dim: int
    x_num: int
    patch_num: int
    patch_num_output: int
    conv_dim: int = 32
    time_embed: str = "learnable"
    max_time_len: int = 20

    def setup(self):
        patch_res = self.x_num // self.patch_num
        patch_res_out = self.x_num // self.patch_num_output

        # --- Encoder ---
        self.in_proj = nn.Conv(
            features=self.dim,
            kernel_size=(patch_res, patch_res),
            strides=(patch_res, patch_res),
            padding="VALID",
            use_bias=True,
            name="in_proj",
        )
        self.conv_proj = nn.Conv(
            features=self.dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=True,
            name="conv_proj",
        )

        if self.time_embed == "learnable":
            self.time_embeddings = self.param(
                "time_embeddings",
                nn.initializers.normal(stddev=1.0),
                (1, self.max_time_len, 1, self.dim),
            )
        else:
            self.time_proj_0 = nn.Dense(self.dim, use_bias=True, name="time_proj_0")
            self.time_proj_1 = nn.Dense(self.dim, use_bias=True, name="time_proj_1")

        self.patch_position_embeddings = self.param(
            "patch_position_embeddings",
            nn.initializers.normal(stddev=1.0),
            (1, 1, self.patch_num * self.patch_num, self.dim),
        )

        # --- Decoder ---
        self.post_deconv = nn.ConvTranspose(
            features=self.conv_dim,
            kernel_size=(patch_res_out, patch_res_out),
            strides=(patch_res_out, patch_res_out),
            padding="VALID",
            use_bias=True,
            transpose_kernel=True,
            name="post_deconv",
        )
        self.post_conv = nn.Conv(
            features=self.conv_dim,
            kernel_size=(1, 1),
            padding="VALID",
            use_bias=True,
            name="post_conv",
        )
        self.head = nn.Conv(
            features=self.data_dim,
            kernel_size=(1, 1),
            padding="VALID",
            use_bias=True,
            name="head",
        )

    def __call__(self, data, times):
        """Not used directly — call ``encode`` / ``decode``."""
        raise NotImplementedError("Use encode / decode directly")

    def encode(self, data, times):
        """(bs, t, x_num, x_num, data_dim) -> (bs, t * patch_num^2, dim)"""
        bs = data.shape[0]

        x = rearrange(data, "b t h w c -> (b t) h w c")
        x = self.in_proj(x)
        x = _gelu(x)
        x = self.conv_proj(x)

        x = rearrange(x, "(b t) h w d -> b t (h w) d", b=bs)

        if self.time_embed == "learnable":
            t_len = data.shape[1]
            x = x + self.time_embeddings[:, :t_len]
        else:
            te = self.time_proj_0(times)
            te = _gelu(te)
            te = self.time_proj_1(te)
            x = x + te[:, :, None, :]

        x = x + self.patch_position_embeddings

        x = x.reshape(bs, -1, self.dim)
        return x

    def decode(self, data_output):
        """(bs, t * patch_num_output^2, dim) -> (bs, t, x_num, x_num, data_dim)"""
        bs = data_output.shape[0]

        x = rearrange(
            data_output,
            "b (t h w) d -> (b t) h w d",
            h=self.patch_num_output,
            w=self.patch_num_output,
        )

        x = self.post_deconv(x)
        x = _gelu(x)
        x = self.post_conv(x)
        x = _gelu(x)
        x = self.head(x)

        x = rearrange(x, "(b t) h w c -> b t h w c", b=bs)
        return x


# ---------------------------------------------------------------------------
# Block Causal Mask
# ---------------------------------------------------------------------------

def block_lower_triangular_mask(block_size: int, block_num: int) -> jnp.ndarray:
    """Create a float attention mask for block-causal attention.

    Positions that should be **ignored** are ``-inf``; others are ``0``.
    This matches ``block_lower_triangular_mask(..., use_float=True)`` in PyTorch.
    """
    matrix_size = block_size * block_num
    idx = jnp.arange(matrix_size)
    # lower triangular: q_idx >= kv_idx
    lower_tri = idx[:, None] >= idx[None, :]
    # block diagonal: same block
    block_eq = (idx[:, None] // block_size) == (idx[None, :] // block_size)
    allowed = lower_tri | block_eq
    return jnp.where(allowed, 0.0, -jnp.inf)


# ---------------------------------------------------------------------------
# BCAT  (autoregressive block-causal transformer)
# ---------------------------------------------------------------------------

class BCAT(nn.Module):
    """Block Causal Autoregressive Transformer for PDE prediction.

    Default configuration (from ``bcat.yaml``):
        n_layer=12, dim_emb=1024, dim_ffn=2752, n_head=8,
        norm_first=True, norm='rms', activation='swiglu', qk_norm=True,
        patch_num=16, patch_num_output=16, x_num=128, max_output_dim=4,
        max_data_len=20, conv_dim=32, time_embed='learnable', max_time_len=20,
    """
    # Transformer hyper-params
    n_layer: int = 12
    dim_emb: int = 1024
    dim_ffn: int = 2752
    n_head: int = 8
    norm_first: bool = True
    norm_type: str = "rms"
    activation: str = "swiglu"
    qk_norm: bool = True

    # Spatial / embedder params
    x_num: int = 128
    max_output_dim: int = 4
    patch_num: int = 16
    patch_num_output: int = 16
    conv_dim: int = 32
    time_embed: str = "learnable"
    max_time_len: int = 20
    max_data_len: int = 20
    deep: bool = False

    @nn.compact
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
        data_dim = data.shape[-1]
        seq_len_per_step = self.patch_num ** 2

        # Autoregressive: drop last timestep
        data = data[:, :-1]
        times = times[:, :-1]

        # Encode
        embedder = ConvEmbedder(
            dim=self.dim_emb,
            data_dim=data_dim,
            x_num=self.x_num,
            patch_num=self.patch_num,
            patch_num_output=self.patch_num_output,
            conv_dim=self.conv_dim,
            time_embed=self.time_embed,
            max_time_len=self.max_time_len,
            name="embedder",
        )
        encoded = embedder.encode(data, times)  # (bs, data_len, dim)

        # Block-causal mask
        data_len = encoded.shape[1]
        mask = block_lower_triangular_mask(seq_len_per_step, self.max_data_len)
        mask = mask[:data_len, :data_len]

        # Transformer
        encoded = TransformerEncoder(
            n_layer=self.n_layer,
            d_model=self.dim_emb,
            nhead=self.n_head,
            dim_feedforward=self.dim_ffn,
            activation=self.activation,
            norm_first=self.norm_first,
            qk_norm=self.qk_norm,
            norm_type=self.norm_type,
            name="transformer",
        )(encoded, mask=mask)

        # Decode only the output steps
        input_seq_len = (input_len - 1) * seq_len_per_step
        data_output = encoded[:, input_seq_len:]

        data_output = embedder.decode(data_output)
        return data_output
