"""Graphormer layers implemented in JAX/Flax."""
from typing import Optional
import math

import jax
import jax.numpy as jnp
from flax import linen as nn


class GraphNodeFeature(nn.Module):
    """Compute node features for each node in the graph."""
    num_heads: int
    num_node_type: int
    num_in_degree: int
    num_out_degree: int
    embed_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, node_type, in_degree, out_degree):
        """
        Args:
            node_type: [n_graph, n_node, 1]
            in_degree: [n_graph, n_node]
            out_degree: [n_graph, n_node]
        Returns:
            [n_graph, n_node, embed_dim]
        """
        # Node type embedding (1 for graph token)
        node_encoder = nn.Embed(
            num_embeddings=self.num_node_type + 1,
            features=self.embed_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="node_encoder"
        )
        
        in_degree_encoder = nn.Embed(
            num_embeddings=self.num_in_degree,
            features=self.embed_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="in_degree_encoder"
        )
        
        out_degree_encoder = nn.Embed(
            num_embeddings=self.num_out_degree,
            features=self.embed_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="out_degree_encoder"
        )
        
        # node_type shape: [n_graph, n_node, 1] -> sum over last dim
        node_feature = node_encoder(node_type).sum(axis=-2)  # [n_graph, n_node, embed_dim]
        node_feature = node_feature + in_degree_encoder(in_degree)  # [n_graph, n_node, embed_dim]
        node_feature = node_feature + out_degree_encoder(out_degree)  # [n_graph, n_node, embed_dim]
        
        return node_feature


class GraphAttnBias(nn.Module):
    """Compute attention bias for each head."""
    num_heads: int
    num_spatial: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, attn_bias, spatial_pos):
        """
        Args:
            attn_bias: [n_graph, n_node, n_node]
            spatial_pos: [n_graph, n_node, n_node]
        Returns:
            [n_graph, n_head, n_node, n_node]
        """
        spatial_pos_encoder = nn.Embed(
            num_embeddings=self.num_spatial,
            features=self.num_heads,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="spatial_pos_encoder"
        )
        
        spatial_pos_encoder_rev = nn.Embed(
            num_embeddings=self.num_spatial,
            features=self.num_heads,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="spatial_pos_encoder_rev"
        )
        
        # spatial_pos: [n_graph, n_node, n_node] -> [n_graph, n_node, n_node, num_heads]
        spatial_bias = spatial_pos_encoder(spatial_pos)
        # Reverse encoder uses transposed spatial_pos (j→i instead of i→j)
        spatial_bias_rev = spatial_pos_encoder_rev(jnp.transpose(spatial_pos, (0, 2, 1)))
        
        # Combine forward and reverse spatial biases
        spatial_bias = spatial_bias + spatial_bias_rev
        
        # Transpose to [n_graph, num_heads, n_node, n_node]
        spatial_bias = jnp.transpose(spatial_bias, (0, 3, 1, 2))
        
        # Add attn_bias: [n_graph, n_node, n_node] -> [n_graph, 1, n_node, n_node]
        attn_bias = jnp.expand_dims(attn_bias, axis=1)
        
        return spatial_bias + attn_bias


class MultiheadAttention(nn.Module):
    """Multi-headed attention."""
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, attn_bias=None, key_padding_mask=None, attn_mask=None, deterministic=True):
        """
        Args:
            x: [n_node, n_graph, embed_dim]
            attn_bias: [n_graph, num_heads, n_node, n_node]
            key_padding_mask: [n_graph, n_node]
            attn_mask: [n_node, n_node]
        Returns:
            [n_node, n_graph, embed_dim]
        """
        n_node, n_graph, embed_dim = x.shape
        head_dim = embed_dim // self.num_heads
        scaling = head_dim ** -0.5
        
        # Xavier uniform initialization for projections
        kernel_init = nn.initializers.xavier_uniform()
        scale = math.sqrt(1 / embed_dim)
        bias_init = nn.initializers.uniform(scale=scale)
        
        q_proj = nn.Dense(embed_dim, kernel_init=kernel_init, bias_init=bias_init, dtype=self.dtype, name="q_proj")
        k_proj = nn.Dense(embed_dim, kernel_init=kernel_init, bias_init=bias_init, dtype=self.dtype, name="k_proj")
        v_proj = nn.Dense(embed_dim, kernel_init=kernel_init, bias_init=bias_init, dtype=self.dtype, name="v_proj")
        out_proj = nn.Dense(embed_dim, kernel_init=nn.initializers.xavier_uniform(), 
                           bias_init=nn.initializers.zeros, dtype=self.dtype, name="out_proj")
        
        # Compute Q, K, V
        query = q_proj(x) * scaling  # [n_node, n_graph, embed_dim]
        key = k_proj(x)  # [n_node, n_graph, embed_dim]
        value = v_proj(x)  # [n_node, n_graph, embed_dim]
        
        # Reshape for multi-head attention
        # [n_node, n_graph, embed_dim] -> [n_graph*num_heads, n_node, head_dim]
        query = query.reshape(n_node, n_graph * self.num_heads, head_dim).transpose((1, 0, 2))
        key = key.reshape(n_node, n_graph * self.num_heads, head_dim).transpose((1, 0, 2))
        value = value.reshape(n_node, n_graph * self.num_heads, head_dim).transpose((1, 0, 2))
        
        # Compute attention weights
        # [n_graph*num_heads, n_node, head_dim] @ [n_graph*num_heads, head_dim, n_node]
        # -> [n_graph*num_heads, n_node, n_node]
        attn_weights = jnp.matmul(query, key.transpose((0, 2, 1)))
        
        # Add attention bias (Graphormer-specific)
        if attn_bias is not None:
            # [n_graph, num_heads, n_node, n_node] -> [n_graph*num_heads, n_node, n_node]
            attn_bias = attn_bias.reshape(n_graph * self.num_heads, n_node, n_node)
            attn_weights = attn_weights + attn_bias
        
        # Add attention mask if provided
        if attn_mask is not None:
            attn_mask = jnp.expand_dims(attn_mask, axis=0)  # [1, n_node, n_node]
            attn_weights = attn_weights + attn_mask
        
        # Apply key padding mask
        if key_padding_mask is not None:
            # [n_graph, n_node] -> [n_graph, 1, 1, n_node]
            key_padding_mask = jnp.expand_dims(jnp.expand_dims(key_padding_mask, axis=1), axis=2)
            # [n_graph*num_heads, n_node, n_node] -> [n_graph, num_heads, n_node, n_node]
            attn_weights = attn_weights.reshape(n_graph, self.num_heads, n_node, n_node)
            attn_weights = jnp.where(key_padding_mask, float("-inf"), attn_weights)
            attn_weights = attn_weights.reshape(n_graph * self.num_heads, n_node, n_node)
        
        # Softmax
        attn_probs = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(self.dtype)
        
        # Apply attention to values
        # [n_graph*num_heads, n_node, n_node] @ [n_graph*num_heads, n_node, head_dim]
        # -> [n_graph*num_heads, n_node, head_dim]
        attn = jnp.matmul(attn_probs, value)
        
        # Reshape back
        # [n_graph*num_heads, n_node, head_dim] -> [n_node, n_graph, embed_dim]
        attn = attn.transpose((1, 0, 2)).reshape(n_node, n_graph, embed_dim)
        
        # Output projection
        attn = out_proj(attn)
        
        return attn


class GraphormerEncoderLayer(nn.Module):
    """Basic module in Transformer encoder."""
    embed_dim: int = 768
    ffn_embed_dim: int = 3072
    num_heads: int = 8
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    activation_fn: str = "gelu"
    pre_layernorm: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, attn_bias=None, attn_mask=None, attn_padding_mask=None, deterministic=True):
        """
        Args:
            x: [n_node, n_graph, embed_dim]
        Returns:
            [n_node, n_graph, embed_dim]
        """
        residual = x
        
        if self.pre_layernorm:
            x = nn.LayerNorm(epsilon=1e-5, dtype=jnp.float32, name="attn_layer_norm")(x)
        
        # Multi-head attention
        x = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.attention_dropout,
            dtype=self.dtype,
            name="multihead_attn"
        )(x, attn_bias=attn_bias, key_padding_mask=attn_padding_mask, 
          attn_mask=attn_mask, deterministic=deterministic)
        
        x = residual + x
        
        if not self.pre_layernorm:
            x = nn.LayerNorm(epsilon=1e-5, dtype=jnp.float32, name="attn_layer_norm")(x)
        
        # Feed-forward network
        residual = x
        
        if self.pre_layernorm:
            x = nn.LayerNorm(epsilon=1e-5, dtype=jnp.float32, name="ffn_layer_norm")(x)
        
        # FFN
        scale_fc1 = math.sqrt(1.0 / self.embed_dim)
        x = nn.Dense(
            self.ffn_embed_dim,
            kernel_init=nn.initializers.uniform(scale=scale_fc1),
            bias_init=nn.initializers.uniform(scale=scale_fc1),
            dtype=self.dtype,
            name="fc1"
        )(x)
        
        if self.activation_fn.lower() == "gelu":
            x = nn.gelu(x)
        else:
            x = nn.relu(x)
        
        scale_fc2 = math.sqrt(1.0 / self.ffn_embed_dim)
        x = nn.Dense(
            self.embed_dim,
            kernel_init=nn.initializers.uniform(scale=scale_fc2),
            bias_init=nn.initializers.uniform(scale=scale_fc2),
            dtype=self.dtype,
            name="fc2"
        )(x)
        
        x = residual + x
        
        if not self.pre_layernorm:
            x = nn.LayerNorm(epsilon=1e-5, dtype=jnp.float32, name="ffn_layer_norm")(x)
        
        return x


class GraphormerEncoder(nn.Module):
    """Graphormer encoder."""
    num_node_type: int
    num_in_degree: int
    num_out_degree: int
    num_spatial: int
    num_encoder_layers: int = 12
    embed_dim: int = 768
    ffn_embed_dim: int = 768
    num_heads: int = 32
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    encoder_normalize_before: bool = False
    pre_layernorm: bool = False
    activation_fn: str = "gelu"
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, node_type, node_input_feature, in_degree, out_degree, 
                 attn_bias, spatial_pos, token_embeddings=None, attn_mask=None, deterministic=True):
        """
        Args:
            node_type: [n_graph, n_node, 1]
            node_input_feature: [n_graph, n_node, embed_dim]
            in_degree: [n_graph, n_node]
            out_degree: [n_graph, n_node]
            attn_bias: [n_graph, n_node, n_node]
            spatial_pos: [n_graph, n_node, n_node]
        Returns:
            [n_node, n_graph, embed_dim]
        """
        # Compute padding mask
        node_type_ = node_type.squeeze(-1)  # [n_graph, n_node]
        padding_mask = (node_type_ == 0)  # [n_graph, n_node]
        
        if token_embeddings is not None:
            x = token_embeddings
        else:
            x = GraphNodeFeature(
                num_heads=self.num_heads,
                num_node_type=self.num_node_type,
                num_in_degree=self.num_in_degree,
                num_out_degree=self.num_out_degree,
                embed_dim=self.embed_dim,
                dtype=self.dtype,
                name="graph_node_feature"
            )(node_type, in_degree, out_degree)
        
        x = x + node_input_feature  # [n_graph, n_node, embed_dim]
        
        # Compute attention bias
        attn_bias = GraphAttnBias(
            num_heads=self.num_heads,
            num_spatial=self.num_spatial,
            dtype=self.dtype,
            name="graph_attn_bias"
        )(attn_bias, spatial_pos)  # [n_graph, n_head, n_node, n_node]
        
        if self.encoder_normalize_before:
            x = nn.LayerNorm(epsilon=1e-5, dtype=jnp.float32, name="emb_layer_norm")(x)
        
        # Transpose: [n_graph, n_node, embed_dim] -> [n_node, n_graph, embed_dim]
        x = jnp.transpose(x, (1, 0, 2))
        
        # Apply encoder layers
        for i in range(self.num_encoder_layers):
            x = GraphormerEncoderLayer(
                embed_dim=self.embed_dim,
                ffn_embed_dim=self.ffn_embed_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                attention_dropout=self.attention_dropout,
                activation_dropout=self.activation_dropout,
                activation_fn=self.activation_fn,
                pre_layernorm=self.pre_layernorm,
                dtype=self.dtype,
                name=f"layers_{i}"
            )(x, attn_bias=attn_bias, attn_mask=attn_mask, 
              attn_padding_mask=padding_mask, deterministic=deterministic)
        
        return x  # [n_node, n_graph, embed_dim]
