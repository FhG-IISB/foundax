"""Flax linen Transformer for neural operator applications.

Port of jNO's Equinox Transformer to Flax ``nn.Module``.
Provides encoder-only and full encoder-decoder variants.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    max_len: int
    embed_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (seq_len, embed_dim)
        seq_len = x.shape[0]
        pos = jnp.arange(self.max_len, dtype=jnp.float32)[:, None]
        div = jnp.exp(
            jnp.arange(0, self.embed_dim, 2, dtype=jnp.float32)
            * -(jnp.log(10000.0) / self.embed_dim)
        )
        pe = jnp.zeros((self.max_len, self.embed_dim))
        pe = pe.at[:, 0::2].set(jnp.sin(pos * div))
        pe = pe.at[:, 1::2].set(jnp.cos(pos * div))
        return x + pe[:seq_len].astype(x.dtype)


class EncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block."""

    num_heads: int
    qkv_features: int
    mlp_features: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        # Pre-norm self-attention
        y = nn.LayerNorm(name="ln1")(x)
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
            name="sa",
        )(y, mask=mask)
        x = x + y

        # Pre-norm FFN
        y = nn.LayerNorm(name="ln2")(x)
        y = nn.Dense(self.mlp_features, name="ff1")(y)
        y = nn.relu(y)
        y = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(y)
        y = nn.Dense(x.shape[-1], name="ff2")(y)
        return x + y


class DecoderBlock(nn.Module):
    """Pre-norm Transformer decoder block (self-attn + cross-attn + FFN)."""

    num_heads: int
    qkv_features: int
    mlp_features: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        enc_out: jnp.ndarray,
        self_mask: Optional[jnp.ndarray] = None,
        cross_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        # Self-attention
        y = nn.LayerNorm(name="ln1")(x)
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
            name="sa",
        )(y, mask=self_mask)
        x = x + y

        # Cross-attention
        y = nn.LayerNorm(name="ln2")(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
            name="ca",
        )(y, enc_out, mask=cross_mask)
        x = x + y

        # FFN
        y = nn.LayerNorm(name="ln3")(x)
        y = nn.Dense(self.mlp_features, name="ff1")(y)
        y = nn.relu(y)
        y = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(y)
        y = nn.Dense(x.shape[-1], name="ff2")(y)
        return x + y


class Transformer(nn.Module):
    """Full encoder-decoder Transformer.

    Designed for sequence-to-sequence tasks in PDE operator learning.

    Attributes
    ----------
    encoder_num_layers, decoder_num_layers : int
        Number of encoder / decoder blocks.
    embed_dim : int
        Token embedding dimension.
    num_heads : int
        Number of attention heads.
    qkv_features : int
        QKV projection width.
    mlp_features : int
        FFN hidden width.
    vocab_size : int
        Token vocabulary size for embeddings.
    max_len : int
        Maximum sequence length for positional encoding.
    dropout_rate : float
        Dropout probability.
    """

    encoder_num_layers: int
    decoder_num_layers: int
    embed_dim: int
    num_heads: int
    qkv_features: int
    mlp_features: int
    vocab_size: int
    max_len: int = 512
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        encoder_input_tokens: jnp.ndarray,
        decoder_input_tokens: jnp.ndarray,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        decoder_self_attention_mask: Optional[jnp.ndarray] = None,
        cross_attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        # Encoder
        enc = nn.Embed(self.vocab_size, self.embed_dim, name="enc_embed")(
            encoder_input_tokens
        )
        enc = PositionalEncoding(self.max_len, self.embed_dim, name="enc_pe")(enc)
        enc = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(enc)
        for i in range(self.encoder_num_layers):
            enc = EncoderBlock(
                num_heads=self.num_heads,
                qkv_features=self.qkv_features,
                mlp_features=self.mlp_features,
                dropout_rate=self.dropout_rate,
                name=f"enc_{i}",
            )(enc, mask=encoder_attention_mask, deterministic=deterministic)
        enc = nn.LayerNorm(name="enc_ln")(enc)

        # Decoder
        dec = nn.Embed(self.vocab_size, self.embed_dim, name="dec_embed")(
            decoder_input_tokens
        )
        dec = PositionalEncoding(self.max_len, self.embed_dim, name="dec_pe")(dec)
        dec = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(dec)
        for i in range(self.decoder_num_layers):
            dec = DecoderBlock(
                num_heads=self.num_heads,
                qkv_features=self.qkv_features,
                mlp_features=self.mlp_features,
                dropout_rate=self.dropout_rate,
                name=f"dec_{i}",
            )(
                dec,
                enc,
                self_mask=decoder_self_attention_mask,
                cross_mask=cross_attention_mask,
                deterministic=deterministic,
            )
        dec = nn.LayerNorm(name="dec_ln")(dec)

        # Logits
        return nn.Dense(self.vocab_size, name="logits")(dec)
