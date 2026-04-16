from dataclasses import dataclass, field

import flax.linen as nn
import jax.numpy as jnp


N_MAX_POSITIONS = 512


def _sinusoidal_embedding(n_pos: int, dim: int) -> jnp.ndarray:
    pos = jnp.arange(n_pos, dtype=jnp.float32)[:, None]
    i = jnp.arange(dim, dtype=jnp.float32)[None, :]
    angle = pos / jnp.power(10000.0, 2.0 * jnp.floor(i / 2.0) / float(dim))
    out = jnp.zeros((n_pos, dim), dtype=jnp.float32)
    out = out.at[:, 0::2].set(jnp.sin(angle[:, 0::2]))
    out = out.at[:, 1::2].set(jnp.cos(angle[:, 1::2]))
    return out


def _lengths_to_mask(lengths: jnp.ndarray, max_len: int) -> jnp.ndarray:
    ar = jnp.arange(max_len, dtype=jnp.int32)[None, :]
    return ar < lengths[:, None]


class TorchLikeMHA(nn.Module):
    dim: int
    n_head: int

    def setup(self):
        self.q_proj = nn.Dense(self.dim, use_bias=True, name="q_proj")
        self.k_proj = nn.Dense(self.dim, use_bias=True, name="k_proj")
        self.v_proj = nn.Dense(self.dim, use_bias=True, name="v_proj")
        self.out_proj = nn.Dense(self.dim, use_bias=True, name="out_proj")

    def __call__(
        self,
        query: jnp.ndarray,
        key_value: jnp.ndarray | None = None,
        key_padding_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if key_value is None:
            key_value = query

        bs, q_len, _ = query.shape
        k_len = key_value.shape[1]
        h = self.n_head
        d = self.dim // h

        q = self.q_proj(query).reshape(bs, q_len, h, d).transpose(0, 2, 1, 3)
        k = self.k_proj(key_value).reshape(bs, k_len, h, d).transpose(0, 2, 1, 3)
        v = self.v_proj(key_value).reshape(bs, k_len, h, d).transpose(0, 2, 1, 3)

        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(float(d))
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]
            scores = jnp.where(mask, -1e30, scores)

        w = nn.softmax(scores, axis=-1)
        out = jnp.einsum("bhqk,bhkd->bhqd", w, v)
        out = out.transpose(0, 2, 1, 3).reshape(bs, q_len, self.dim)
        return self.out_proj(out)


class TransformerFFN(nn.Module):
    in_dim: int
    hidden_dim: int
    out_dim: int
    n_hidden_layers: int

    def setup(self):
        self.lin1 = nn.Dense(self.hidden_dim, use_bias=True, name="lin1")
        self.mid = [
            nn.Dense(self.hidden_dim, use_bias=True, name=f"midlin_{i}")
            for i in range(max(0, self.n_hidden_layers - 1))
        ]
        self.lin2 = nn.Dense(self.out_dim, use_bias=True, name="lin2")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.gelu(self.lin1(x), approximate=False)
        for m in self.mid:
            x = nn.gelu(m(x), approximate=False)
        return self.lin2(x)


class FusionTransformerModel(nn.Module):
    dim: int
    n_layers: int
    n_heads: int
    n_hidden_layers: int
    use_type_embeddings: bool = True

    def setup(self):
        self.type_embeddings = (
            nn.Embed(2, self.dim, name="type_embeddings")
            if self.use_type_embeddings
            else None
        )
        self.layer_norm_emb = nn.LayerNorm(epsilon=1e-12, name="layer_norm_emb")

        self.attentions = [
            TorchLikeMHA(self.dim, self.n_heads, name=f"attentions_{i}")
            for i in range(self.n_layers)
        ]
        self.layer_norm1 = [
            nn.LayerNorm(epsilon=1e-12, name=f"layer_norm1_{i}")
            for i in range(self.n_layers)
        ]
        self.ffns = [
            TransformerFFN(
                self.dim,
                self.dim * 4,
                self.dim,
                self.n_hidden_layers,
                name=f"ffns_{i}",
            )
            for i in range(self.n_layers)
        ]
        self.layer_norm2 = [
            nn.LayerNorm(epsilon=1e-12, name=f"layer_norm2_{i}")
            for i in range(self.n_layers)
        ]

    def __call__(
        self,
        x_data: jnp.ndarray,
        x_text: jnp.ndarray,
        lengths_data: jnp.ndarray,
        lengths_text: jnp.ndarray,
    ) -> jnp.ndarray:
        # inputs are (slen, bs, dim)
        x_data = jnp.transpose(x_data, (1, 0, 2))
        x_text = jnp.transpose(x_text, (1, 0, 2))
        bs = x_data.shape[0]
        data_len = x_data.shape[1]
        text_len = x_text.shape[1]

        if self.type_embeddings is not None:
            t0 = self.type_embeddings(jnp.zeros((1, 1), dtype=jnp.int32))
            t1 = self.type_embeddings(jnp.ones((1, 1), dtype=jnp.int32))
            x_data = x_data + t0
            x_text = x_text + t1

        x = jnp.concatenate([x_data, x_text], axis=1)
        mask_data = _lengths_to_mask(lengths_data, data_len)
        mask_text = _lengths_to_mask(lengths_text, text_len)
        valid = jnp.concatenate([mask_data, mask_text], axis=1)
        pad_mask = ~valid

        x = self.layer_norm_emb(x)
        x = x * valid[:, :, None].astype(x.dtype)

        for i in range(self.n_layers):
            a = self.attentions[i](x, key_padding_mask=pad_mask)
            x = self.layer_norm1[i](x + a)
            x = self.layer_norm2[i](x + self.ffns[i](x))
            x = x * valid[:, :, None].astype(x.dtype)

        return jnp.transpose(x, (1, 0, 2))


class DataTransformerModel(nn.Module):
    dim: int
    n_layers: int
    n_heads: int
    n_hidden_layers: int
    positional_embeddings: str | None = None

    def setup(self):
        if self.positional_embeddings == "learnable":
            self.position_embeddings = nn.Embed(
                N_MAX_POSITIONS, self.dim, name="position_embeddings"
            )
            self.pos_sin = None
        elif self.positional_embeddings == "sinusoidal":
            self.position_embeddings = None
            self.pos_sin = _sinusoidal_embedding(N_MAX_POSITIONS, self.dim)
        else:
            self.position_embeddings = None
            self.pos_sin = None

        self.layer_norm_emb = nn.LayerNorm(epsilon=1e-12, name="layer_norm_emb")
        self.attentions = [
            TorchLikeMHA(self.dim, self.n_heads, name=f"attentions_{i}")
            for i in range(self.n_layers)
        ]
        self.layer_norm1 = [
            nn.LayerNorm(epsilon=1e-12, name=f"layer_norm1_{i}")
            for i in range(self.n_layers)
        ]
        self.ffns = [
            TransformerFFN(
                self.dim,
                self.dim * 4,
                self.dim,
                self.n_hidden_layers,
                name=f"ffns_{i}",
            )
            for i in range(self.n_layers)
        ]
        self.layer_norm2 = [
            nn.LayerNorm(epsilon=1e-12, name=f"layer_norm2_{i}")
            for i in range(self.n_layers)
        ]

    def __call__(
        self,
        x: jnp.ndarray,
        lengths: jnp.ndarray,
    ) -> jnp.ndarray:
        # x is (slen, bs, dim)
        slen = x.shape[0]
        x = jnp.transpose(x, (1, 0, 2))
        valid = _lengths_to_mask(lengths, slen)
        pad_mask = ~valid

        if self.position_embeddings is not None:
            pos = jnp.arange(slen, dtype=jnp.int32)[None, :]
            x = x + self.position_embeddings(pos)
        elif self.pos_sin is not None:
            x = x + self.pos_sin[None, :slen, :]

        x = self.layer_norm_emb(x)
        x = x * valid[:, :, None].astype(x.dtype)

        for i in range(self.n_layers):
            a = self.attentions[i](x, key_padding_mask=pad_mask)
            x = self.layer_norm1[i](x + a)
            x = self.layer_norm2[i](x + self.ffns[i](x))
            x = x * valid[:, :, None].astype(x.dtype)

        return jnp.transpose(x, (1, 0, 2))


class TextTransformerModel(nn.Module):
    n_words: int
    pad_index: int
    dim: int
    n_layers: int
    n_heads: int
    n_hidden_layers: int
    positional_embeddings: str | None = "sinusoidal"

    def setup(self):
        self.embeddings = nn.Embed(
            self.n_words, self.dim, embedding_init=nn.initializers.normal(self.dim**-0.5), name="embeddings"
        )
        if self.positional_embeddings == "learnable":
            self.position_embeddings = nn.Embed(
                N_MAX_POSITIONS, self.dim, name="position_embeddings"
            )
            self.pos_sin = None
        elif self.positional_embeddings == "sinusoidal":
            self.position_embeddings = None
            self.pos_sin = _sinusoidal_embedding(N_MAX_POSITIONS, self.dim)
        else:
            self.position_embeddings = None
            self.pos_sin = None

        self.layer_norm_emb = nn.LayerNorm(epsilon=1e-12, name="layer_norm_emb")
        self.attentions = [
            TorchLikeMHA(self.dim, self.n_heads, name=f"attentions_{i}")
            for i in range(self.n_layers)
        ]
        self.layer_norm1 = [
            nn.LayerNorm(epsilon=1e-12, name=f"layer_norm1_{i}")
            for i in range(self.n_layers)
        ]
        self.ffns = [
            TransformerFFN(
                self.dim,
                self.dim * 4,
                self.dim,
                self.n_hidden_layers,
                name=f"ffns_{i}",
            )
            for i in range(self.n_layers)
        ]
        self.layer_norm2 = [
            nn.LayerNorm(epsilon=1e-12, name=f"layer_norm2_{i}")
            for i in range(self.n_layers)
        ]

    def __call__(self, x: jnp.ndarray, lengths: jnp.ndarray) -> jnp.ndarray:
        # x is (slen, bs)
        slen, bs = x.shape
        tok = jnp.transpose(x, (1, 0))
        h = self.embeddings(tok)

        valid = _lengths_to_mask(lengths, slen)
        pad_mask = ~valid

        if self.position_embeddings is not None:
            pos = jnp.arange(slen, dtype=jnp.int32)[None, :]
            h = h + self.position_embeddings(pos)
        elif self.pos_sin is not None:
            h = h + self.pos_sin[None, :slen, :]

        h = self.layer_norm_emb(h)
        h = h * valid[:, :, None].astype(h.dtype)

        for i in range(self.n_layers):
            a = self.attentions[i](h, key_padding_mask=pad_mask)
            h = self.layer_norm1[i](h + a)
            h = self.layer_norm2[i](h + self.ffns[i](h))
            h = h * valid[:, :, None].astype(h.dtype)

        return jnp.transpose(h, (1, 0, 2))


class DataOperatorModel(nn.Module):
    dim: int
    n_layers: int
    n_heads: int
    n_hidden_layers: int
    max_output_dimension: int
    split_fused_feature_data: bool = True
    no_text: bool = False
    data_feature_resnet: bool = False
    data_decoder_attn: bool = False
    positional_embeddings: str | None = None
    x_grid_size: int = 1
    two_layer_proj: bool = False

    def setup(self):
        self.hidden_dim = self.dim * 4
        self.query_embedder = nn.Dense(self.dim, use_bias=True, name="query_embedder")
        if self.positional_embeddings == "learnable":
            self.position_embeddings = nn.Embed(
                N_MAX_POSITIONS, self.dim, name="position_embeddings"
            )
            self.pos_sin = None
        elif self.positional_embeddings == "sinusoidal":
            self.position_embeddings = None
            self.pos_sin = _sinusoidal_embedding(N_MAX_POSITIONS, self.dim)
        else:
            self.position_embeddings = None
            self.pos_sin = None

        self.layer_norm_emb = nn.LayerNorm(epsilon=1e-12, name="layer_norm_emb")

        if self.data_feature_resnet:
            self.data_embedder_0 = nn.Dense(self.dim * 2, use_bias=True, name="data_embedder_0")
            self.data_embedder_2 = nn.Dense(self.dim, use_bias=True, name="data_embedder_2")
            if not self.no_text and not self.split_fused_feature_data:
                self.text_embedder_0 = nn.Dense(self.dim * 2, use_bias=True, name="text_embedder_0")
                self.text_embedder_2 = nn.Dense(self.dim, use_bias=True, name="text_embedder_2")

        self.attentions = [
            TorchLikeMHA(self.dim, self.n_heads, name=f"attentions_{i}")
            for i in range(self.n_layers)
        ]
        self.layer_norm1 = [
            nn.LayerNorm(epsilon=1e-12, name=f"layer_norm1_{i}")
            for i in range(self.n_layers)
        ]
        if self.data_decoder_attn:
            self.encoder_attn = [
                TorchLikeMHA(self.dim, self.n_heads, name=f"encoder_attn_{i}")
                for i in range(self.n_layers)
            ]
            self.layer_norm15 = [
                nn.LayerNorm(epsilon=1e-12, name=f"layer_norm15_{i}")
                for i in range(self.n_layers)
            ]
        self.ffns = [
            TransformerFFN(
                self.dim,
                self.hidden_dim,
                self.dim,
                self.n_hidden_layers,
                name=f"ffns_{i}",
            )
            for i in range(self.n_layers)
        ]
        self.layer_norm2 = [
            nn.LayerNorm(epsilon=1e-12, name=f"layer_norm2_{i}")
            for i in range(self.n_layers)
        ]

        out_dim = self.max_output_dimension * self.x_grid_size
        if self.two_layer_proj:
            self.proj_0 = nn.Dense(self.hidden_dim, use_bias=True, name="proj_0")
            self.proj_1 = nn.Dense(out_dim, use_bias=True, name="proj_1")
        else:
            self.proj = nn.Dense(out_dim, use_bias=True, name="proj")

    def get_query_emb(self, query_times: jnp.ndarray) -> jnp.ndarray:
        # query_times is (slen,)
        return self.query_embedder(query_times[:, None])

    def _apply_proj(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.two_layer_proj:
            return self.proj_1(self.proj_0(x))
        return self.proj(x)

    def forward_hidden(
        self,
        query_emb: jnp.ndarray,
        src_enc: jnp.ndarray,
        src_len: tuple[jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        # query_emb (slen, dim), src_enc (bs, slen, dim)
        bs = src_enc.shape[0]
        q = jnp.broadcast_to(query_emb[None, :, :], (bs, query_emb.shape[0], self.dim))

        src_data_len, src_text_len = src_len
        max_data = src_enc.shape[1] if self.no_text else int(jnp.max(src_data_len))
        src_data_mask = _lengths_to_mask(src_data_len, max_data)

        src = src_enc
        if self.no_text:
            src_mask = src_data_mask
            if self.data_feature_resnet:
                d = nn.gelu(self.data_embedder_0(src), approximate=False)
                src = src + self.data_embedder_2(d)
        else:
            if self.split_fused_feature_data:
                src = src_enc[:, :max_data, :]
                src_mask = src_data_mask
                if self.data_feature_resnet:
                    d = nn.gelu(self.data_embedder_0(src), approximate=False)
                    src = src + self.data_embedder_2(d)
            else:
                text_max = src_enc.shape[1] - max_data
                src_text_mask = _lengths_to_mask(src_text_len, text_max)
                src_mask = jnp.concatenate([src_data_mask, src_text_mask], axis=1)
                if self.data_feature_resnet:
                    src_d = src_enc[:, :max_data, :]
                    src_t = src_enc[:, max_data:, :]
                    src_d = src_d + self.data_embedder_2(
                        nn.gelu(self.data_embedder_0(src_d), approximate=False)
                    )
                    src_t = src_t + self.text_embedder_2(
                        nn.gelu(self.text_embedder_0(src_t), approximate=False)
                    )
                    src = jnp.concatenate([src_d, src_t], axis=1)

        src_pad = ~src_mask
        if self.position_embeddings is not None:
            pos = jnp.arange(q.shape[1], dtype=jnp.int32)[None, :]
            q = q + self.position_embeddings(pos)
        elif self.pos_sin is not None:
            q = q + self.pos_sin[None, : q.shape[1], :]

        h = self.layer_norm_emb(q)
        for i in range(self.n_layers):
            a = self.attentions[i](h, key_value=src, key_padding_mask=src_pad)
            h = self.layer_norm1[i](h + a)
            if self.data_decoder_attn:
                aa = self.encoder_attn[i](h)
                h = self.layer_norm15[i](h + aa)
            h = self.layer_norm2[i](h + self.ffns[i](h))

        return jnp.transpose(h, (1, 0, 2))

    def generate(
        self,
        src_enc: jnp.ndarray,
        src_len: tuple[jnp.ndarray, jnp.ndarray],
        query_emb: jnp.ndarray,
    ) -> jnp.ndarray:
        hidden = self.forward_hidden(query_emb, src_enc, src_len)
        return self._apply_proj(hidden)


class RevIN(nn.Module):
    dim: int

    def setup(self):
        self.gamma = self.param("gamma", nn.initializers.ones, (self.dim,))
        self.beta = self.param("beta", nn.initializers.zeros, (self.dim,))

    def __call__(self, x: jnp.ndarray, eps: float = 1e-6):
        # x: (slen, bs, 1 + dim)
        y = x[:, :, 1:]
        mu = jnp.mean(y, axis=0, keepdims=True)
        var = jnp.var(y, axis=0, keepdims=True)
        yhat = (y - mu) / jnp.sqrt(var + eps)
        yout = yhat * self.gamma[None, None, :] + self.beta[None, None, :]
        xout = x.at[:, :, 1:].set(yout)
        return xout, mu, var

    def reverse(self, y: jnp.ndarray, mu: jnp.ndarray, var: jnp.ndarray, eps: float = 1e-6):
        yhat = (y - self.beta[None, None, :]) / self.gamma[None, None, :]
        return yhat * jnp.sqrt(var + eps) + mu


@dataclass
class ProseTextData2to1Config:
    emb_dim: int = 512
    n_text_enc_layers: int = 4
    n_data_enc_layers: int = 2
    n_data_dec_layers: int = 8
    n_fusion_layers: int = 8
    n_text_heads: int = 8
    n_data_heads: int = 8
    n_fusion_heads: int = 8
    n_text_hidden_layers: int = 1
    n_data_hidden_layers: int = 1
    n_fusion_hidden_layers: int = 1
    split_fused_feature_data: bool = True
    data_feature_resnet: bool = False
    data_decoder_attn: bool = False
    no_text: bool = False
    text_positional_embeddings: str | None = "sinusoidal"
    data_positional_embeddings: str | None = None
    data_decoder_positional_embeddings: str | None = None
    fusion_type_embeddings: bool = True
    x_patch_size: int = 1
    x_grid_size: int = 1
    normalization: bool = False


class PROSEODE2to1(nn.Module):
    n_words: int
    pad_index: int
    max_output_dimension: int
    cfg: ProseTextData2to1Config = field(default_factory=ProseTextData2to1Config)

    def setup(self):
        self.embedder_0 = nn.Dense(self.cfg.emb_dim, use_bias=True, name="embedder_0")
        self.embedder_2 = nn.Dense(self.cfg.emb_dim, use_bias=True, name="embedder_2")

        self.data_encoder = DataTransformerModel(
            dim=self.cfg.emb_dim,
            n_layers=self.cfg.n_data_enc_layers,
            n_heads=self.cfg.n_data_heads,
            n_hidden_layers=self.cfg.n_data_hidden_layers,
            positional_embeddings=self.cfg.data_positional_embeddings,
            name="data_encoder",
        )
        self.text_encoder = TextTransformerModel(
            n_words=self.n_words,
            pad_index=self.pad_index,
            dim=self.cfg.emb_dim,
            n_layers=self.cfg.n_text_enc_layers,
            n_heads=self.cfg.n_text_heads,
            n_hidden_layers=self.cfg.n_text_hidden_layers,
            positional_embeddings=self.cfg.text_positional_embeddings,
            name="text_encoder",
        )
        self.fusion = FusionTransformerModel(
            dim=self.cfg.emb_dim,
            n_layers=self.cfg.n_fusion_layers,
            n_heads=self.cfg.n_fusion_heads,
            n_hidden_layers=self.cfg.n_fusion_hidden_layers,
            use_type_embeddings=self.cfg.fusion_type_embeddings,
            name="fusion",
        )
        self.data_decoder = DataOperatorModel(
            dim=self.cfg.emb_dim,
            n_layers=self.cfg.n_data_dec_layers,
            n_heads=self.cfg.n_data_heads,
            n_hidden_layers=self.cfg.n_data_hidden_layers,
            max_output_dimension=self.max_output_dimension,
            split_fused_feature_data=self.cfg.split_fused_feature_data,
            no_text=self.cfg.no_text,
            data_feature_resnet=self.cfg.data_feature_resnet,
            data_decoder_attn=self.cfg.data_decoder_attn,
            positional_embeddings=self.cfg.data_decoder_positional_embeddings,
            x_grid_size=1,
            two_layer_proj=False,
            name="data_decoder",
        )

    def __call__(
        self,
        data_input: jnp.ndarray,
        data_lengths: jnp.ndarray,
        query_times: jnp.ndarray,
        text_input: jnp.ndarray,
        text_lengths: jnp.ndarray,
    ) -> jnp.ndarray:
        x = nn.gelu(self.embedder_0(data_input), approximate=False)
        x = self.embedder_2(x)
        data_encoded = self.data_encoder(x, data_lengths)

        if self.cfg.no_text:
            fused = data_encoded
            txt_len = jnp.zeros_like(data_lengths)
        else:
            text_encoded = self.text_encoder(text_input, text_lengths)
            fused = self.fusion(data_encoded, text_encoded, data_lengths, text_lengths)
            txt_len = text_lengths

        fused_bs = jnp.transpose(fused, (1, 0, 2))
        q = self.data_decoder.get_query_emb(query_times)
        return self.data_decoder.generate(fused_bs, (data_lengths, txt_len), q)


class PROSEPDE2to1(nn.Module):
    n_words: int
    pad_index: int
    max_output_dimension: int
    cfg: ProseTextData2to1Config = field(default_factory=ProseTextData2to1Config)

    def setup(self):
        self.embedder_0 = nn.Dense(self.cfg.emb_dim, use_bias=True, name="embedder_0")
        self.embedder_2 = nn.Dense(self.cfg.emb_dim, use_bias=True, name="embedder_2")
        if self.cfg.normalization:
            self.normalizer = RevIN(
                dim=self.max_output_dimension * self.cfg.x_patch_size,
                name="normalizer",
            )
        else:
            self.normalizer = None

        self.data_encoder = DataTransformerModel(
            dim=self.cfg.emb_dim,
            n_layers=self.cfg.n_data_enc_layers,
            n_heads=self.cfg.n_data_heads,
            n_hidden_layers=self.cfg.n_data_hidden_layers,
            positional_embeddings=self.cfg.data_positional_embeddings,
            name="data_encoder",
        )
        self.text_encoder = TextTransformerModel(
            n_words=self.n_words,
            pad_index=self.pad_index,
            dim=self.cfg.emb_dim,
            n_layers=self.cfg.n_text_enc_layers,
            n_heads=self.cfg.n_text_heads,
            n_hidden_layers=self.cfg.n_text_hidden_layers,
            positional_embeddings=self.cfg.text_positional_embeddings,
            name="text_encoder",
        )
        self.fusion = FusionTransformerModel(
            dim=self.cfg.emb_dim,
            n_layers=self.cfg.n_fusion_layers,
            n_heads=self.cfg.n_fusion_heads,
            n_hidden_layers=self.cfg.n_fusion_hidden_layers,
            use_type_embeddings=self.cfg.fusion_type_embeddings,
            name="fusion",
        )
        self.data_decoder = DataOperatorModel(
            dim=self.cfg.emb_dim,
            n_layers=self.cfg.n_data_dec_layers,
            n_heads=self.cfg.n_data_heads,
            n_hidden_layers=self.cfg.n_data_hidden_layers,
            max_output_dimension=self.max_output_dimension,
            split_fused_feature_data=self.cfg.split_fused_feature_data,
            no_text=self.cfg.no_text,
            data_feature_resnet=self.cfg.data_feature_resnet,
            data_decoder_attn=self.cfg.data_decoder_attn,
            positional_embeddings=self.cfg.data_decoder_positional_embeddings,
            x_grid_size=self.cfg.x_grid_size,
            two_layer_proj=True,
            name="data_decoder",
        )

    def __call__(
        self,
        data_input: jnp.ndarray,
        data_lengths: jnp.ndarray,
        query_times: jnp.ndarray,
        text_input: jnp.ndarray,
        text_lengths: jnp.ndarray,
    ) -> jnp.ndarray:
        x = data_input
        mu = None
        var = None
        if self.normalizer is not None:
            x, mu, var = self.normalizer(x)

        x = nn.gelu(self.embedder_0(x), approximate=False)
        x = self.embedder_2(x)
        data_encoded = self.data_encoder(x, data_lengths)

        if self.cfg.no_text:
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
