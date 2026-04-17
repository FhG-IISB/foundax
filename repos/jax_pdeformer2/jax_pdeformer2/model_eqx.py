"""Equinox reimplementation of PDEformer-2.

Mirrors the Flax modules in ``basic_block.py``, ``function_encoder.py``,
``graphormer.py``, ``inr_with_hypernet.py``, and ``pdeformer.py``.
"""

from __future__ import annotations

import math
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Helper: apply eqx.nn.Linear to batched inputs
# ---------------------------------------------------------------------------


def _apply_linear(linear: eqx.nn.Linear, x: jnp.ndarray) -> jnp.ndarray:
    """Apply Linear to (..., in_features) -> (..., out_features)."""
    out = x @ linear.weight.T
    if linear.bias is not None:
        out = out + linear.bias
    return out


def _apply_embedding(emb: eqx.nn.Embedding, ids: jnp.ndarray) -> jnp.ndarray:
    """Apply Embedding to integer ids of arbitrary batch shape."""
    return emb.weight[ids]


# ---------------------------------------------------------------------------
# basic_block equivalents
# ---------------------------------------------------------------------------


class MLP(eqx.Module):
    layers: list  # list of eqx.nn.Linear

    def __init__(
        self, dim_in: int, dim_out: int, dim_hidden: int, num_layers: int = 3, *, key
    ):
        keys = jax.random.split(key, num_layers)
        self.layers = []
        if num_layers > 1:
            self.layers.append(
                eqx.nn.Linear(dim_in, dim_hidden, use_bias=True, key=keys[0])
            )
            for i in range(1, num_layers - 1):
                self.layers.append(
                    eqx.nn.Linear(dim_hidden, dim_hidden, use_bias=True, key=keys[i])
                )
            self.layers.append(
                eqx.nn.Linear(dim_hidden, dim_out, use_bias=True, key=keys[-1])
            )
        elif num_layers == 1:
            self.layers.append(
                eqx.nn.Linear(dim_in, dim_out, use_bias=True, key=keys[0])
            )
        else:
            raise ValueError(f"num_layers should be > 0, got {num_layers}")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers[:-1]:
            x = _apply_linear(layer, x)
            x = jax.nn.relu(x)
        x = _apply_linear(self.layers[-1], x)
        return x


class Sine(eqx.Module):
    omega_0: float = eqx.field(static=True)

    def __init__(self, omega_0: float = 1.0):
        self.omega_0 = omega_0

    def __call__(self, x):
        return jnp.sin(self.omega_0 * x)


class Scale(eqx.Module):
    a: float = eqx.field(static=True)

    def __init__(self, a: float = 1.0):
        self.a = a

    def __call__(self, x):
        return self.a * x


class Clamp(eqx.Module):
    threshold: float = eqx.field(static=True)

    def __init__(self, threshold: float = 256.0):
        self.threshold = threshold

    def __call__(self, x):
        return jnp.clip(x, -self.threshold, self.threshold)


# ---------------------------------------------------------------------------
# function_encoder
# ---------------------------------------------------------------------------


class Conv2dFuncEncoderV3(eqx.Module):
    in_dim: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    resolution: int = eqx.field(static=True)
    input_txyz: bool = eqx.field(static=True)
    keep_nchw: bool = eqx.field(static=True)

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d

    def __init__(
        self,
        in_dim: int = 5,
        out_dim: int = 256,
        resolution: int = 128,
        input_txyz: bool = True,
        keep_nchw: bool = True,
        *,
        key,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.resolution = resolution
        self.input_txyz = input_txyz
        self.keep_nchw = keep_nchw

        in_ch = in_dim if input_txyz else 1
        k1, k2, k3 = jax.random.split(key, 3)
        self.conv1 = eqx.nn.Conv2d(
            in_ch, 32, kernel_size=4, stride=4, padding=0, use_bias=True, key=k1
        )
        self.conv2 = eqx.nn.Conv2d(
            32, 128, kernel_size=4, stride=4, padding=0, use_bias=True, key=k2
        )
        self.conv3 = eqx.nn.Conv2d(
            128, out_dim, kernel_size=4, stride=4, padding=0, use_bias=True, key=k3
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        bsz, _, dim_in = x.shape
        x = x.reshape(bsz, self.resolution, self.resolution, dim_in)
        # NHWC -> NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))
        f_values = x[:, -1:, :, :]
        net_in = x if self.input_txyz else f_values

        # Conv layers (eqx Conv2d expects (C,H,W), vmap over batch)
        x = jax.vmap(self.conv1)(net_in)
        x = jax.nn.relu(x)
        x = jax.vmap(self.conv2)(x)
        x = jax.nn.relu(x)
        x = jax.vmap(self.conv3)(x)

        if not self.keep_nchw:
            x = jnp.transpose(x, (0, 2, 3, 1))
        return x


# ---------------------------------------------------------------------------
# graphormer
# ---------------------------------------------------------------------------


class GraphNodeFeature(eqx.Module):
    num_heads: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)

    node_encoder: eqx.nn.Embedding
    in_degree_encoder: eqx.nn.Embedding
    out_degree_encoder: eqx.nn.Embedding

    def __init__(
        self,
        num_heads: int,
        num_node_type: int,
        num_in_degree: int,
        num_out_degree: int,
        embed_dim: int,
        *,
        key,
    ):
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        k1, k2, k3 = jax.random.split(key, 3)
        self.node_encoder = eqx.nn.Embedding(num_node_type + 1, embed_dim, key=k1)
        self.in_degree_encoder = eqx.nn.Embedding(num_in_degree, embed_dim, key=k2)
        self.out_degree_encoder = eqx.nn.Embedding(num_out_degree, embed_dim, key=k3)

    def __call__(self, node_type, in_degree, out_degree):
        # node_type: [n_graph, n_node, 1]
        node_feature = _apply_embedding(self.node_encoder, node_type).sum(axis=-2)
        node_feature = node_feature + _apply_embedding(
            self.in_degree_encoder, in_degree
        )
        node_feature = node_feature + _apply_embedding(
            self.out_degree_encoder, out_degree
        )
        return node_feature


class GraphAttnBias(eqx.Module):
    num_heads: int = eqx.field(static=True)

    spatial_pos_encoder: eqx.nn.Embedding
    spatial_pos_encoder_rev: eqx.nn.Embedding

    def __init__(self, num_heads: int, num_spatial: int, *, key):
        self.num_heads = num_heads
        k1, k2 = jax.random.split(key)
        self.spatial_pos_encoder = eqx.nn.Embedding(num_spatial, num_heads, key=k1)
        self.spatial_pos_encoder_rev = eqx.nn.Embedding(num_spatial, num_heads, key=k2)

    def __call__(self, attn_bias, spatial_pos):
        # spatial_pos: [n_graph, n_node, n_node]
        spatial_bias = _apply_embedding(self.spatial_pos_encoder, spatial_pos)
        spatial_bias_rev = _apply_embedding(
            self.spatial_pos_encoder_rev, jnp.transpose(spatial_pos, (0, 2, 1))
        )
        spatial_bias = spatial_bias + spatial_bias_rev
        spatial_bias = jnp.transpose(spatial_bias, (0, 3, 1, 2))
        attn_bias = jnp.expand_dims(attn_bias, axis=1)
        return spatial_bias + attn_bias


class MultiheadAttention(eqx.Module):
    embed_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear

    def __init__(self, embed_dim: int, num_heads: int, *, key):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.q_proj = eqx.nn.Linear(embed_dim, embed_dim, use_bias=True, key=k1)
        self.k_proj = eqx.nn.Linear(embed_dim, embed_dim, use_bias=True, key=k2)
        self.v_proj = eqx.nn.Linear(embed_dim, embed_dim, use_bias=True, key=k3)
        self.out_proj = eqx.nn.Linear(embed_dim, embed_dim, use_bias=True, key=k4)

    def __call__(self, x, attn_bias=None, key_padding_mask=None, attn_mask=None):
        n_node, n_graph, embed_dim = x.shape
        head_dim = embed_dim // self.num_heads
        scaling = head_dim**-0.5

        query = _apply_linear(self.q_proj, x) * scaling
        key = _apply_linear(self.k_proj, x)
        value = _apply_linear(self.v_proj, x)

        query = query.reshape(n_node, n_graph * self.num_heads, head_dim).transpose(
            (1, 0, 2)
        )
        key = key.reshape(n_node, n_graph * self.num_heads, head_dim).transpose(
            (1, 0, 2)
        )
        value = value.reshape(n_node, n_graph * self.num_heads, head_dim).transpose(
            (1, 0, 2)
        )

        attn_weights = jnp.matmul(query, key.transpose((0, 2, 1)))

        if attn_bias is not None:
            attn_bias = attn_bias.reshape(n_graph * self.num_heads, n_node, n_node)
            attn_weights = attn_weights + attn_bias

        if attn_mask is not None:
            attn_weights = attn_weights + jnp.expand_dims(attn_mask, axis=0)

        if key_padding_mask is not None:
            key_padding_mask = jnp.expand_dims(
                jnp.expand_dims(key_padding_mask, axis=1), axis=2
            )
            attn_weights = attn_weights.reshape(n_graph, self.num_heads, n_node, n_node)
            attn_weights = jnp.where(key_padding_mask, float("-inf"), attn_weights)
            attn_weights = attn_weights.reshape(
                n_graph * self.num_heads, n_node, n_node
            )

        attn_probs = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1)
        attn = jnp.matmul(attn_probs, value)
        attn = attn.transpose((1, 0, 2)).reshape(n_node, n_graph, embed_dim)
        return _apply_linear(self.out_proj, attn)


class GraphormerEncoderLayer(eqx.Module):
    embed_dim: int = eqx.field(static=True)
    pre_layernorm: bool = eqx.field(static=True)
    activation_fn: str = eqx.field(static=True)

    multihead_attn: MultiheadAttention
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    attn_layer_norm: eqx.nn.LayerNorm
    ffn_layer_norm: eqx.nn.LayerNorm

    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        num_heads: int = 8,
        pre_layernorm: bool = False,
        activation_fn: str = "gelu",
        *,
        key,
    ):
        self.embed_dim = embed_dim
        self.pre_layernorm = pre_layernorm
        self.activation_fn = activation_fn

        k1, k2, k3 = jax.random.split(key, 3)
        self.multihead_attn = MultiheadAttention(embed_dim, num_heads, key=k1)
        self.fc1 = eqx.nn.Linear(embed_dim, ffn_embed_dim, use_bias=True, key=k2)
        self.fc2 = eqx.nn.Linear(ffn_embed_dim, embed_dim, use_bias=True, key=k3)
        self.attn_layer_norm = eqx.nn.LayerNorm(embed_dim, eps=1e-5)
        self.ffn_layer_norm = eqx.nn.LayerNorm(embed_dim, eps=1e-5)

    def _apply_norm(self, norm, x):
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        x_flat = jax.vmap(norm)(x_flat)
        return x_flat.reshape(orig_shape)

    def __call__(self, x, attn_bias=None, attn_mask=None, attn_padding_mask=None):
        residual = x
        if self.pre_layernorm:
            x = self._apply_norm(self.attn_layer_norm, x)
        x = self.multihead_attn(
            x,
            attn_bias=attn_bias,
            key_padding_mask=attn_padding_mask,
            attn_mask=attn_mask,
        )
        x = residual + x
        if not self.pre_layernorm:
            x = self._apply_norm(self.attn_layer_norm, x)

        residual = x
        if self.pre_layernorm:
            x = self._apply_norm(self.ffn_layer_norm, x)
        x = _apply_linear(self.fc1, x)
        if self.activation_fn.lower() == "gelu":
            x = jax.nn.gelu(x, approximate=False)
        else:
            x = jax.nn.relu(x)
        x = _apply_linear(self.fc2, x)
        x = residual + x
        if not self.pre_layernorm:
            x = self._apply_norm(self.ffn_layer_norm, x)

        return x


class GraphormerEncoder(eqx.Module):
    embed_dim: int = eqx.field(static=True)
    num_encoder_layers: int = eqx.field(static=True)
    encoder_normalize_before: bool = eqx.field(static=True)

    graph_node_feature: GraphNodeFeature
    graph_attn_bias: GraphAttnBias
    layers: list  # list of GraphormerEncoderLayer
    emb_layer_norm: Optional[eqx.nn.LayerNorm]

    def __init__(
        self,
        num_node_type: int,
        num_in_degree: int,
        num_out_degree: int,
        num_spatial: int,
        num_encoder_layers: int = 12,
        embed_dim: int = 768,
        ffn_embed_dim: int = 768,
        num_heads: int = 32,
        encoder_normalize_before: bool = False,
        pre_layernorm: bool = False,
        activation_fn: str = "gelu",
        *,
        key,
    ):
        self.embed_dim = embed_dim
        self.num_encoder_layers = num_encoder_layers
        self.encoder_normalize_before = encoder_normalize_before

        keys = jax.random.split(key, 3 + num_encoder_layers)
        self.graph_node_feature = GraphNodeFeature(
            num_heads=num_heads,
            num_node_type=num_node_type,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            embed_dim=embed_dim,
            key=keys[0],
        )
        self.graph_attn_bias = GraphAttnBias(
            num_heads=num_heads, num_spatial=num_spatial, key=keys[1]
        )
        self.layers = [
            GraphormerEncoderLayer(
                embed_dim=embed_dim,
                ffn_embed_dim=ffn_embed_dim,
                num_heads=num_heads,
                pre_layernorm=pre_layernorm,
                activation_fn=activation_fn,
                key=keys[2 + i],
            )
            for i in range(num_encoder_layers)
        ]
        if encoder_normalize_before:
            self.emb_layer_norm = eqx.nn.LayerNorm(embed_dim, eps=1e-5)
        else:
            self.emb_layer_norm = None

    def __call__(
        self,
        node_type,
        node_input_feature,
        in_degree,
        out_degree,
        attn_bias,
        spatial_pos,
        token_embeddings=None,
        attn_mask=None,
    ):
        node_type_ = node_type.squeeze(-1)
        padding_mask = node_type_ == 0

        if token_embeddings is not None:
            x = token_embeddings
        else:
            x = self.graph_node_feature(node_type, in_degree, out_degree)

        x = x + node_input_feature
        attn_bias_out = self.graph_attn_bias(attn_bias, spatial_pos)

        if self.emb_layer_norm is not None:
            orig_shape = x.shape
            x_flat = x.reshape(-1, orig_shape[-1])
            x_flat = jax.vmap(self.emb_layer_norm)(x_flat)
            x = x_flat.reshape(orig_shape)

        x = jnp.transpose(x, (1, 0, 2))

        for layer in self.layers:
            x = layer(
                x,
                attn_bias=attn_bias_out,
                attn_mask=attn_mask,
                attn_padding_mask=padding_mask,
            )

        return x


# ---------------------------------------------------------------------------
# PolyINR + Hypernet
# ---------------------------------------------------------------------------


class PolyINR(eqx.Module):
    dim_in: int = eqx.field(static=True)
    dim_out: int = eqx.field(static=True)
    dim_hidden: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    activation_fn: str = eqx.field(static=True)
    affine_act_fn: str = eqx.field(static=True)

    affines: list  # list of eqx.nn.Linear
    dense_layers: list  # list of eqx.nn.Linear
    last_layer: eqx.nn.Linear

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_hidden: int,
        num_layers: int,
        activation_fn: str = "lrelu",
        affine_act_fn: str = "identity",
        *,
        key,
    ):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.activation_fn = activation_fn
        self.affine_act_fn = affine_act_fn

        n = num_layers - 1
        keys = jax.random.split(key, 2 * n + 1)
        self.affines = [
            eqx.nn.Linear(dim_in, dim_hidden, use_bias=True, key=keys[i])
            for i in range(n)
        ]
        self.dense_layers = [
            eqx.nn.Linear(dim_hidden, dim_hidden, use_bias=True, key=keys[n + i])
            for i in range(n)
        ]
        self.last_layer = eqx.nn.Linear(
            dim_hidden, dim_out, use_bias=True, key=keys[2 * n]
        )

    def __call__(
        self, x, affine_modulations=None, scale_modulations=None, shift_modulations=None
    ):
        x_pad = jnp.concatenate(
            [x, jnp.ones((*x.shape[:-1], 1), dtype=x.dtype)], axis=-1
        )
        hidden_state = 1.0

        for layer_idx in range(self.num_layers - 1):
            if scale_modulations is not None:
                scale = 1.0 + jnp.expand_dims(scale_modulations[layer_idx], axis=1)
            else:
                scale = 1.0

            if shift_modulations is not None:
                shift = jnp.expand_dims(shift_modulations[layer_idx], axis=1)
            else:
                shift = 0.0

            affine = _apply_linear(self.affines[layer_idx], x)

            if self.affine_act_fn.lower() in ("none", "identity"):
                tmp = affine
            elif self.affine_act_fn.lower() in ("lrelu", "leakyrelu"):
                tmp = jax.nn.leaky_relu(affine, negative_slope=0.2)
                tmp = jnp.clip(tmp, -256.0, 256.0)
            elif self.affine_act_fn.lower() in ("sin", "sine"):
                tmp = jnp.sin(affine) * math.sqrt(2.0)
            else:
                tmp = affine

            if affine_modulations is not None:
                tmp2 = jnp.matmul(x_pad, affine_modulations[layer_idx])
                tmp = tmp + tmp2

            hidden_state = hidden_state * tmp
            hidden_state = _apply_linear(self.dense_layers[layer_idx], hidden_state)
            hidden_state = scale * hidden_state + shift

            if self.activation_fn.lower() in ("lrelu", "leakyrelu"):
                hidden_state = jax.nn.leaky_relu(hidden_state, negative_slope=0.2)
                hidden_state = jnp.clip(hidden_state, -256.0, 256.0)
            elif self.activation_fn.lower() in ("sin", "sine"):
                hidden_state = jnp.sin(hidden_state)

        return _apply_linear(self.last_layer, hidden_state)


class PolyINRWithHypernet(eqx.Module):
    inr_dim_in: int = eqx.field(static=True)
    inr_dim_out: int = eqx.field(static=True)
    inr_dim_hidden: int = eqx.field(static=True)
    inr_num_layers: int = eqx.field(static=True)
    share_hypernet: bool = eqx.field(static=True)
    enable_affine: bool = eqx.field(static=True)
    enable_shift: bool = eqx.field(static=True)
    enable_scale: bool = eqx.field(static=True)

    inr: PolyINR
    shift_hypernets: Optional[list]  # list of MLP or single MLP
    scale_hypernets: Optional[list]
    affine_hypernets: Optional[list]

    def __init__(
        self,
        inr_dim_in: int,
        inr_dim_out: int,
        inr_dim_hidden: int,
        inr_num_layers: int,
        hyper_dim_in: int,
        hyper_dim_hidden: int,
        hyper_num_layers: int,
        share_hypernet: bool = False,
        enable_affine: bool = False,
        enable_shift: bool = True,
        enable_scale: bool = True,
        activation_fn: str = "lrelu",
        affine_act_fn: str = "identity",
        *,
        key,
    ):
        self.inr_dim_in = inr_dim_in
        self.inr_dim_out = inr_dim_out
        self.inr_dim_hidden = inr_dim_hidden
        self.inr_num_layers = inr_num_layers
        self.share_hypernet = share_hypernet
        self.enable_affine = enable_affine
        self.enable_shift = enable_shift
        self.enable_scale = enable_scale

        num_hypernet = inr_num_layers - 1
        keys = jax.random.split(key, 4)

        self.inr = PolyINR(
            dim_in=inr_dim_in,
            dim_out=inr_dim_out,
            dim_hidden=inr_dim_hidden,
            num_layers=inr_num_layers,
            activation_fn=activation_fn,
            affine_act_fn=affine_act_fn,
            key=keys[0],
        )

        # Shift hypernets
        if enable_shift:
            shift_keys = jax.random.split(keys[1], num_hypernet)
            if share_hypernet:
                shared = MLP(
                    hyper_dim_in,
                    inr_dim_hidden,
                    hyper_dim_hidden,
                    hyper_num_layers,
                    key=shift_keys[0],
                )
                self.shift_hypernets = [shared] * num_hypernet
            else:
                self.shift_hypernets = [
                    MLP(
                        hyper_dim_in,
                        inr_dim_hidden,
                        hyper_dim_hidden,
                        hyper_num_layers,
                        key=shift_keys[i],
                    )
                    for i in range(num_hypernet)
                ]
        else:
            self.shift_hypernets = None

        # Scale hypernets
        if enable_scale:
            scale_keys = jax.random.split(keys[2], num_hypernet)
            if share_hypernet:
                shared = MLP(
                    hyper_dim_in,
                    inr_dim_hidden,
                    hyper_dim_hidden,
                    hyper_num_layers,
                    key=scale_keys[0],
                )
                self.scale_hypernets = [shared] * num_hypernet
            else:
                self.scale_hypernets = [
                    MLP(
                        hyper_dim_in,
                        inr_dim_hidden,
                        hyper_dim_hidden,
                        hyper_num_layers,
                        key=scale_keys[i],
                    )
                    for i in range(num_hypernet)
                ]
        else:
            self.scale_hypernets = None

        # Affine hypernets
        if enable_affine:
            affine_out = (inr_dim_in + 1) * inr_dim_hidden
            affine_keys = jax.random.split(keys[3], num_hypernet)
            if share_hypernet:
                shared = MLP(
                    hyper_dim_in,
                    affine_out,
                    hyper_dim_hidden,
                    hyper_num_layers,
                    key=affine_keys[0],
                )
                self.affine_hypernets = [shared] * num_hypernet
            else:
                self.affine_hypernets = [
                    MLP(
                        hyper_dim_in,
                        affine_out,
                        hyper_dim_hidden,
                        hyper_num_layers,
                        key=affine_keys[i],
                    )
                    for i in range(num_hypernet)
                ]
        else:
            self.affine_hypernets = None

    def __call__(self, coordinate, hyper_in):
        affine_modulations, scale_modulations, shift_modulations = (
            self._get_modulations(hyper_in)
        )
        return self.inr(
            coordinate, affine_modulations, scale_modulations, shift_modulations
        )

    def _get_modulations(self, hyper_in):
        num_hypernet = self.inr_num_layers - 1

        if self.enable_affine:
            affine_list = []
            for idx in range(num_hypernet):
                out = self.affine_hypernets[idx](hyper_in[idx])
                affine_list.append(out)
            affine_modulations = jnp.stack(affine_list, axis=0)
            affine_modulations = affine_modulations.reshape(
                num_hypernet, -1, self.inr_dim_in + 1, self.inr_dim_hidden
            )
        else:
            affine_modulations = None

        if self.enable_shift:
            shift_list = []
            for idx in range(num_hypernet):
                out = self.shift_hypernets[idx](hyper_in[idx])
                shift_list.append(out)
            shift_modulations = jnp.stack(shift_list, axis=0)
        else:
            shift_modulations = None

        if self.enable_scale:
            scale_list = []
            for idx in range(num_hypernet):
                out = self.scale_hypernets[idx](hyper_in[idx])
                scale_list.append(out)
            scale_modulations = jnp.stack(scale_list, axis=0)
        else:
            scale_modulations = None

        return affine_modulations, scale_modulations, shift_modulations


# ---------------------------------------------------------------------------
# PDEEncoder & PDEformer
# ---------------------------------------------------------------------------

SPACE_DIM = 3


class PDEEncoder(eqx.Module):
    embed_dim: int = eqx.field(static=True)
    func_enc_resolution: int = eqx.field(static=True)
    func_enc_input_txyz: bool = eqx.field(static=True)
    func_enc_keep_nchw: bool = eqx.field(static=True)

    scalar_encoder: MLP
    function_encoder: Conv2dFuncEncoderV3
    graphormer: GraphormerEncoder

    def __init__(
        self,
        num_node_type: int = 128,
        num_in_degree: int = 32,
        num_out_degree: int = 32,
        num_spatial: int = 16,
        num_encoder_layers: int = 12,
        embed_dim: int = 768,
        ffn_embed_dim: int = 1536,
        num_heads: int = 32,
        pre_layernorm: bool = True,
        scalar_dim_hidden: int = 256,
        scalar_num_layers: int = 3,
        func_enc_resolution: int = 128,
        func_enc_input_txyz: bool = False,
        func_enc_keep_nchw: bool = True,
        *,
        key,
    ):
        self.embed_dim = embed_dim
        self.func_enc_resolution = func_enc_resolution
        self.func_enc_input_txyz = func_enc_input_txyz
        self.func_enc_keep_nchw = func_enc_keep_nchw

        k1, k2, k3 = jax.random.split(key, 3)
        self.scalar_encoder = MLP(
            dim_in=1,
            dim_out=embed_dim,
            dim_hidden=scalar_dim_hidden,
            num_layers=scalar_num_layers,
            key=k1,
        )
        self.function_encoder = Conv2dFuncEncoderV3(
            in_dim=1 + SPACE_DIM + 1,
            out_dim=embed_dim,
            resolution=func_enc_resolution,
            input_txyz=func_enc_input_txyz,
            keep_nchw=func_enc_keep_nchw,
            key=k2,
        )
        self.graphormer = GraphormerEncoder(
            num_node_type=num_node_type,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            num_spatial=num_spatial,
            num_encoder_layers=num_encoder_layers,
            embed_dim=embed_dim,
            ffn_embed_dim=ffn_embed_dim,
            num_heads=num_heads,
            pre_layernorm=pre_layernorm,
            key=k3,
        )

    def __call__(
        self,
        node_type,
        node_scalar,
        node_function,
        in_degree,
        out_degree,
        attn_bias,
        spatial_pos,
    ):
        node_scalar_feature = self.scalar_encoder(node_scalar)

        n_graph, num_function = node_function.shape[0], node_function.shape[1]
        num_points_function = node_function.shape[2]
        node_function_flat = node_function.reshape(
            n_graph * num_function, num_points_function, 1 + SPACE_DIM + 1
        )
        node_function_feature = self.function_encoder(node_function_flat)
        node_function_feature = node_function_feature.reshape(
            n_graph, -1, self.embed_dim
        )

        node_input_feature = jnp.concatenate(
            [node_scalar_feature, node_function_feature], axis=1
        )

        return self.graphormer(
            node_type,
            node_input_feature,
            in_degree,
            out_degree,
            attn_bias,
            spatial_pos,
        )


class PDEformer(eqx.Module):
    n_inr_nodes: int = eqx.field(static=True)
    multi_inr: bool = eqx.field(static=True)
    separate_latent: bool = eqx.field(static=True)

    pde_encoder: PDEEncoder
    inr: PolyINRWithHypernet
    inr2: Optional[PolyINRWithHypernet]

    def __init__(
        self,
        # Graphormer
        num_node_type: int = 128,
        num_in_degree: int = 32,
        num_out_degree: int = 32,
        num_spatial: int = 16,
        num_encoder_layers: int = 12,
        embed_dim: int = 768,
        ffn_embed_dim: int = 1536,
        num_heads: int = 32,
        pre_layernorm: bool = True,
        # Scalar encoder
        scalar_dim_hidden: int = 256,
        scalar_num_layers: int = 3,
        # Function encoder
        func_enc_resolution: int = 128,
        func_enc_input_txyz: bool = False,
        func_enc_keep_nchw: bool = True,
        # INR
        inr_dim_hidden: int = 768,
        inr_num_layers: int = 12,
        enable_affine: bool = False,
        enable_shift: bool = True,
        enable_scale: bool = True,
        activation_fn: str = "sin",
        affine_act_fn: str = "identity",
        # Hypernet
        hyper_dim_hidden: int = 512,
        hyper_num_layers: int = 2,
        share_hypernet: bool = False,
        # Multi-INR
        multi_inr: bool = False,
        separate_latent: bool = False,
        *,
        key,
    ):
        self.n_inr_nodes = inr_num_layers - 1
        self.multi_inr = multi_inr
        self.separate_latent = separate_latent

        k1, k2, k3 = jax.random.split(key, 3)

        self.pde_encoder = PDEEncoder(
            num_node_type=num_node_type,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            num_spatial=num_spatial,
            num_encoder_layers=num_encoder_layers,
            embed_dim=embed_dim,
            ffn_embed_dim=ffn_embed_dim,
            num_heads=num_heads,
            pre_layernorm=pre_layernorm,
            scalar_dim_hidden=scalar_dim_hidden,
            scalar_num_layers=scalar_num_layers,
            func_enc_resolution=func_enc_resolution,
            func_enc_input_txyz=func_enc_input_txyz,
            func_enc_keep_nchw=func_enc_keep_nchw,
            key=k1,
        )

        self.inr = PolyINRWithHypernet(
            inr_dim_in=1 + SPACE_DIM,
            inr_dim_out=1,
            inr_dim_hidden=inr_dim_hidden,
            inr_num_layers=inr_num_layers,
            hyper_dim_in=embed_dim,
            hyper_dim_hidden=hyper_dim_hidden,
            hyper_num_layers=hyper_num_layers,
            share_hypernet=share_hypernet,
            enable_affine=enable_affine,
            enable_shift=enable_shift,
            enable_scale=enable_scale,
            activation_fn=activation_fn,
            affine_act_fn=affine_act_fn,
            key=k2,
        )

        if multi_inr:
            self.inr2 = PolyINRWithHypernet(
                inr_dim_in=1 + SPACE_DIM,
                inr_dim_out=1,
                inr_dim_hidden=inr_dim_hidden,
                inr_num_layers=inr_num_layers,
                hyper_dim_in=embed_dim,
                hyper_dim_hidden=hyper_dim_hidden,
                hyper_num_layers=hyper_num_layers,
                share_hypernet=share_hypernet,
                enable_affine=enable_affine,
                enable_shift=enable_shift,
                enable_scale=enable_scale,
                activation_fn=activation_fn,
                affine_act_fn=affine_act_fn,
                key=k3,
            )
        else:
            self.inr2 = None

    def __call__(
        self,
        node_type,
        node_scalar,
        node_function,
        in_degree,
        out_degree,
        attn_bias,
        spatial_pos,
        coordinate,
    ):
        pde_feature = self.pde_encoder(
            node_type,
            node_scalar,
            node_function,
            in_degree,
            out_degree,
            attn_bias,
            spatial_pos,
        )

        out = self.inr(coordinate, pde_feature[: self.n_inr_nodes])

        if self.multi_inr and self.inr2 is not None:
            if self.separate_latent:
                out2 = self.inr2(coordinate, pde_feature[self.n_inr_nodes :])
            else:
                out2 = self.inr2(coordinate, pde_feature[: self.n_inr_nodes])
            return out + out2

        return out
