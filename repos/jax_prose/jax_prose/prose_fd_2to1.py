from dataclasses import dataclass, field

import flax.linen as nn
import jax.numpy as jnp


def _sinusoidal_pe(max_len: int, dim: int) -> jnp.ndarray:
    position = jnp.arange(max_len, dtype=jnp.float32)[:, None]
    div_term = jnp.exp(
        jnp.arange(0, dim, 2, dtype=jnp.float32) * (-jnp.log(10000.0) / dim)
    )
    pe = jnp.zeros((max_len, 1, dim), dtype=jnp.float32)
    pe = pe.at[:, 0, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 0, 1::2].set(jnp.cos(position * div_term))
    return pe


class RMSNormScale(nn.Module):
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        var = jnp.mean(x * x, axis=-1, keepdims=True)
        return x * jax_lax_rsqrt(var + self.eps) * scale


def jax_lax_rsqrt(x):
    return jnp.reciprocal(jnp.sqrt(x))


class CustomMHA(nn.Module):
    dim: int
    n_head: int

    def setup(self):
        self.linear_q = nn.Dense(self.dim, use_bias=True, name="linear_q")
        self.linear_k = nn.Dense(self.dim, use_bias=True, name="linear_k")
        self.linear_v = nn.Dense(self.dim, use_bias=True, name="linear_v")
        self.out_proj = nn.Dense(self.dim, use_bias=True, name="out_proj")

    def __call__(
        self,
        q_in: jnp.ndarray,
        k_in: jnp.ndarray,
        v_in: jnp.ndarray,
        key_padding_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        bs, q_len, _ = q_in.shape
        k_len = k_in.shape[1]
        h = self.n_head
        d = self.dim // h

        q = self.linear_q(q_in).reshape(bs, q_len, h, d).transpose(0, 2, 1, 3)
        k = self.linear_k(k_in).reshape(bs, k_len, h, d).transpose(0, 2, 1, 3)
        v = self.linear_v(v_in).reshape(bs, k_len, h, d).transpose(0, 2, 1, 3)

        score = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(d)
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]
            score = jnp.where(mask, -1e30, score)

        attn = nn.softmax(score, axis=-1)
        out = (
            jnp.einsum("bhqk,bhkd->bhqd", attn, v)
            .transpose(0, 2, 1, 3)
            .reshape(bs, q_len, self.dim)
        )
        return self.out_proj(out)


class EncoderLayer(nn.Module):
    dim: int
    dim_ffn: int
    n_head: int

    def setup(self):
        self.self_attn = CustomMHA(self.dim, self.n_head, name="self_attn")
        self.linear1 = nn.Dense(self.dim_ffn, use_bias=True, name="linear1")
        self.linear2 = nn.Dense(self.dim, use_bias=True, name="linear2")
        self.norm1 = RMSNormScale(name="norm1")
        self.norm2 = RMSNormScale(name="norm2")

    def __call__(
        self, x: jnp.ndarray, key_padding_mask: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        y = self.norm1(x)
        x = x + self.self_attn(y, y, y, key_padding_mask=key_padding_mask)
        y = self.norm2(x)
        y = self.linear2(nn.gelu(self.linear1(y), approximate=False))
        x = x + y
        return x


class Encoder(nn.Module):
    n_layer: int
    dim: int
    dim_ffn: int
    n_head: int

    def setup(self):
        self.layers = [
            EncoderLayer(self.dim, self.dim_ffn, self.n_head, name=f"layers_{i}")
            for i in range(self.n_layer)
        ]
        self.norm = RMSNormScale(name="norm")

    def __call__(
        self, x: jnp.ndarray, key_padding_mask: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.norm(x)


class SymbolEncoder(nn.Module):
    n_words: int
    dim: int
    dim_ffn: int
    n_head: int

    def setup(self):
        self.word_embeddings = nn.Embed(self.n_words, self.dim, name="word_embeddings")
        self.pe = self.param(
            "pe",
            lambda rng, shape: _sinusoidal_pe(shape[0], shape[2]),
            (1024, 1, self.dim),
        )
        self.transformer_encoder = Encoder(
            n_layer=1,
            dim=self.dim,
            dim_ffn=self.dim_ffn,
            n_head=self.n_head,
            name="transformer_encoder",
        )

    def __call__(
        self, x: jnp.ndarray, key_padding_mask: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        x = self.word_embeddings(x)
        x = x + jnp.transpose(self.pe[: x.shape[1]], (1, 0, 2))
        return self.transformer_encoder(x, key_padding_mask=key_padding_mask)


class Fusion(nn.Module):
    dim: int
    dim_ffn: int
    n_head: int
    n_layer: int

    def setup(self):
        self.type_embeddings = nn.Embed(2, self.dim, name="type_embeddings")
        self.transformer_encoder = Encoder(
            self.n_layer,
            self.dim,
            self.dim_ffn,
            self.n_head,
            name="transformer_encoder",
        )

    def __call__(
        self,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        key_padding_mask1: jnp.ndarray | None = None,
    ):
        t0 = self.type_embeddings(jnp.zeros((1, 1), dtype=jnp.int32))
        t1 = self.type_embeddings(jnp.ones((1, 1), dtype=jnp.int32))
        x0 = x0 + t0
        x1 = x1 + t1
        x = jnp.concatenate([x0, x1], axis=1)

        fused_mask = None
        if key_padding_mask1 is not None:
            z = jnp.zeros((x0.shape[0], x0.shape[1]), dtype=bool)
            fused_mask = jnp.concatenate([z, key_padding_mask1], axis=1)

        return self.transformer_encoder(x, key_padding_mask=fused_mask), fused_mask


class OperatorDecoderLayer(nn.Module):
    dim: int
    dim_ffn: int
    n_head: int

    def setup(self):
        self.multihead_attn = CustomMHA(self.dim, self.n_head, name="multihead_attn")
        self.linear1 = nn.Dense(self.dim_ffn, use_bias=True, name="linear1")
        self.linear2 = nn.Dense(self.dim, use_bias=True, name="linear2")
        self.norm1 = RMSNormScale(name="norm1")
        self.norm2 = RMSNormScale(name="norm2")

    def __call__(
        self, q: jnp.ndarray, mem: jnp.ndarray, mem_mask: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        y = self.norm1(q)
        q = q + self.multihead_attn(y, mem, mem, key_padding_mask=mem_mask)
        y = self.norm2(q)
        y = self.linear2(nn.gelu(self.linear1(y), approximate=False))
        return q + y


class DataDecoder(nn.Module):
    dim: int
    dim_ffn: int
    n_head: int
    n_layer: int
    patch_num_output: int
    max_time_len: int = 10

    def setup(self):
        self.time_embed = self.param(
            "time_embed", nn.initializers.normal(), (1, self.max_time_len, 1, self.dim)
        )
        self.patch_position_embeddings = self.param(
            "patch_position_embeddings",
            nn.initializers.normal(),
            (1, 1, self.patch_num_output * self.patch_num_output, self.dim),
        )
        self.layers = [
            OperatorDecoderLayer(
                self.dim, self.dim_ffn, self.n_head, name=f"layers_{i}"
            )
            for i in range(self.n_layer)
        ]
        self.norm = RMSNormScale(name="norm")

    def get_query_emb(self, output_times: jnp.ndarray) -> jnp.ndarray:
        bs = output_times.shape[0]
        out_len = output_times.shape[1]
        t = self.time_embed[:, :out_len]
        return (t + self.patch_position_embeddings).reshape(bs, -1, self.dim)

    def __call__(
        self, src: jnp.ndarray, q: jnp.ndarray, src_mask: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        x = q
        for layer in self.layers:
            x = layer(x, src, mem_mask=src_mask)
        return self.norm(x)


class ConvEmbedder(nn.Module):
    dim: int
    patch_num: int
    patch_num_output: int
    x_num: int
    data_dim: int
    max_time_len: int = 10

    def setup(self):
        self.patch_position_embeddings = self.param(
            "patch_position_embeddings",
            nn.initializers.normal(),
            (1, 1, self.patch_num * self.patch_num, self.dim),
        )
        self.time_embed = self.param(
            "time_embed", nn.initializers.normal(), (1, self.max_time_len, 1, self.dim)
        )
        patch_resolution = self.x_num // self.patch_num
        patch_resolution_output = self.x_num // self.patch_num_output

        self.conv_proj_0 = nn.Conv(
            self.dim,
            (patch_resolution, patch_resolution),
            (patch_resolution, patch_resolution),
            padding="VALID",
            name="conv_proj_0",
        )
        self.conv_proj_1 = nn.Conv(
            self.dim, (1, 1), padding="VALID", name="conv_proj_1"
        )

        self.deconv = nn.ConvTranspose(
            32,
            (patch_resolution_output, patch_resolution_output),
            (patch_resolution_output, patch_resolution_output),
            padding="VALID",
            name="deconv",
        )
        self.post_conv_0 = nn.Conv(32, (1, 1), padding="VALID", name="post_conv_0")
        self.post_conv_1 = nn.Conv(
            self.data_dim, (1, 1), padding="VALID", name="post_conv_1"
        )

    def encode(self, data_input: jnp.ndarray, input_times: jnp.ndarray) -> jnp.ndarray:
        bs, t, h, w, c = data_input.shape
        x = data_input.reshape(bs * t, h, w, c)
        x = self.conv_proj_0(x)
        x = nn.gelu(x, approximate=False)
        x = self.conv_proj_1(x)
        p = self.patch_num
        x = x.reshape(bs, t, p * p, self.dim)
        time_embeddings = self.time_embed[:, :t]
        return (x + time_embeddings + self.patch_position_embeddings).reshape(
            bs, -1, self.dim
        )

    def decode(self, data_output: jnp.ndarray) -> jnp.ndarray:
        bs, qlen, _ = data_output.shape
        p = self.patch_num_output
        out_t = qlen // (p * p)
        x = data_output.reshape(bs, out_t, p, p, self.dim).reshape(
            bs * out_t, p, p, self.dim
        )
        x = self.deconv(x)
        x = nn.gelu(x, approximate=False)
        x = self.post_conv_0(x)
        x = nn.gelu(x, approximate=False)
        x = self.post_conv_1(x)
        return x.reshape(bs, out_t, self.x_num, self.x_num, self.data_dim)


@dataclass
class Prose2to1Config:
    dim_emb: int = 1024
    dim_ffn: int = 2048
    n_head: int = 8
    patch_num: int = 8
    patch_num_output: int = 16
    data_encoder_layers: int = 2
    symbol_encoder_layers: int = 1
    fusion_layers: int = 8
    data_decoder_layers: int = 8


class PROSE2to1(nn.Module):
    n_words: int
    x_num: int = 128
    max_output_dim: int = 4
    cfg: Prose2to1Config = field(default_factory=Prose2to1Config)

    def setup(self):
        c = self.cfg
        self.embedder = ConvEmbedder(
            dim=c.dim_emb,
            patch_num=c.patch_num,
            patch_num_output=c.patch_num_output,
            x_num=self.x_num,
            data_dim=self.max_output_dim,
            name="embedder",
        )
        self.data_encoder = Encoder(
            c.data_encoder_layers, c.dim_emb, c.dim_ffn, c.n_head, name="data_encoder"
        )
        self.symbol_encoder = SymbolEncoder(
            self.n_words, c.dim_emb, c.dim_ffn, c.n_head, name="symbol_encoder"
        )
        self.fusion = Fusion(
            c.dim_emb, c.dim_ffn, c.n_head, c.fusion_layers, name="fusion"
        )
        self.data_decoder = DataDecoder(
            c.dim_emb,
            c.dim_ffn,
            c.n_head,
            c.data_decoder_layers,
            c.patch_num_output,
            name="data_decoder",
        )

    def __call__(
        self,
        data_input: jnp.ndarray,
        input_times: jnp.ndarray,
        output_times: jnp.ndarray,
        symbol_input: jnp.ndarray,
        symbol_padding_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
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
