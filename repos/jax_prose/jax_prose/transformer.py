from typing import Optional

import flax.linen as nn
import jax.numpy as jnp

from .config import DataEncoderConfig, DataDecoderConfig


def _build_norm(norm_type: str, dim: int):
    if norm_type == "rms":
        return nn.RMSNorm()
    return nn.LayerNorm()


class EncoderBlock(nn.Module):
    config: DataEncoderConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        norm1 = _build_norm(self.config.norm, self.config.dim_emb)
        norm2 = _build_norm(self.config.norm, self.config.dim_emb)

        y = norm1(x) if self.config.norm_first else x
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.config.n_head,
            dropout_rate=self.config.dropout,
            deterministic=deterministic,
        )(y, y)
        x = x + attn
        if not self.config.norm_first:
            x = norm1(x)

        y = norm2(x) if self.config.norm_first else x
        y = nn.Dense(self.config.dim_ffn)(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.config.dropout)(y, deterministic=deterministic)
        y = nn.Dense(self.config.dim_emb)(y)
        y = nn.Dropout(rate=self.config.dropout)(y, deterministic=deterministic)

        x = x + y
        if not self.config.norm_first:
            x = norm2(x)
        return x


class TransformerDataEncoder(nn.Module):
    config: DataEncoderConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        for i in range(self.config.n_layer):
            x = EncoderBlock(self.config, name=f"layer_{i}")(
                x, deterministic=deterministic
            )
        return x


class OperatorDecoderBlock(nn.Module):
    config: DataDecoderConfig

    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,
        memory: jnp.ndarray,
        src_key_padding_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        norm1 = _build_norm(self.config.norm, self.config.dim_emb)
        norm2 = _build_norm(self.config.norm, self.config.dim_emb)

        y = norm1(query) if self.config.norm_first else query

        attn_mask = None
        if src_key_padding_mask is not None:
            valid = (~src_key_padding_mask).astype(jnp.int32)
            q_valid = jnp.ones((query.shape[0], query.shape[1]), dtype=jnp.int32)
            attn_mask = nn.make_attention_mask(q_valid, valid)

        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.config.n_head,
            dropout_rate=self.config.dropout,
            deterministic=deterministic,
        )(y, memory, mask=attn_mask)
        x = query + attn_out
        if not self.config.norm_first:
            x = norm1(x)

        y = norm2(x) if self.config.norm_first else x
        y = nn.Dense(self.config.dim_ffn)(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.config.dropout)(y, deterministic=deterministic)
        y = nn.Dense(self.config.dim_emb)(y)
        y = nn.Dropout(rate=self.config.dropout)(y, deterministic=deterministic)
        x = x + y
        if not self.config.norm_first:
            x = norm2(x)
        return x


class DataOperatorDecoder(nn.Module):
    config: DataDecoderConfig
    output_len: int = 1
    space_len: Optional[int] = None

    def setup(self):
        self.dim = self.config.dim_emb
        self._space_len = self.space_len or (self.config.patch_num_output**2)

        if self.config.time_embed == "continuous":
            self.time_proj = nn.Sequential(
                [nn.Dense(self.dim), nn.gelu, nn.Dense(self.dim)]
            )
        else:
            self.time_embed = self.param(
                "time_embed",
                nn.initializers.normal(),
                (1, self.config.max_time_len, 1, self.dim),
            )

        self.patch_position_embeddings = self.param(
            "patch_position_embeddings",
            nn.initializers.normal(),
            (1, 1, self._space_len, self.dim),
        )

    def get_query_emb(self, times: jnp.ndarray) -> jnp.ndarray:
        bs, out_len, _ = times.shape
        if self.config.time_embed == "continuous":
            t = self.time_proj(times)[:, :, None, :]
        else:
            t = self.time_embed[:, :out_len, :, :]
        return (t + self.patch_position_embeddings).reshape(bs, -1, self.dim)

    @nn.compact
    def __call__(
        self,
        src: jnp.ndarray,
        query_emb: jnp.ndarray,
        src_key_padding_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        x = query_emb
        for i in range(self.config.n_layer):
            x = OperatorDecoderBlock(self.config, name=f"layer_{i}")(
                x,
                src,
                src_key_padding_mask=src_key_padding_mask,
                deterministic=deterministic,
            )

        if self.config.norm_first and self.config.final_ln:
            x = _build_norm(self.config.norm, self.config.dim_emb)(x)

        return x
