"""
JAX/Flax implementation of ScOT (Swin Transformer V2 based encoder-decoder).
Translated from PyTorch while maintaining exact architecture.

Github repository of original implementation: https://github.com/camlab-ethz/poseidon
Arxiv paper: https://arxiv.org/abs/2405.19101
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from typing import Optional, Tuple, List, Any, Union, Sequence
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class ScOTConfig:
    """Configuration class for ScOT model."""

    name: str = "poseidonT"
    image_size: int = 224
    patch_size: int = 4
    num_channels: int = 3
    num_out_channels: int = 1
    embed_dim: int = 96
    depths: Tuple[int, ...] = (2, 2, 6, 2)
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)
    skip_connections: Tuple[int, ...] = (True, True, True)
    window_size: int = 7
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    drop_path_rate: float = 0.1
    hidden_act: str = "gelu"
    use_absolute_embeddings: bool = False
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    p: int = 1
    channel_slice_list_normalized_loss: Optional[List[int]] = None
    residual_model: str = "convnext"
    use_conditioning: bool = False
    learn_residual: bool = False
    pretrained_window_sizes: Tuple[int, ...] = (0, 0, 0, 0)
    chunk_size_feed_forward: int = 0
    output_attentions: bool = False
    output_hidden_states: bool = False
    use_return_dict: bool = True

    def __post_init__(self):
        self.num_layers = len(self.depths)
        self.hidden_size = int(self.embed_dim * 2 ** (len(self.depths) - 1))
        if not self.use_conditioning:
            self.learn_residual = False


@dataclass
class ScOTOutput:
    """Output dataclass for ScOT model."""

    loss: Optional[jnp.ndarray] = None
    output: Optional[jnp.ndarray] = None
    hidden_states: Optional[Tuple[jnp.ndarray, ...]] = None
    attentions: Optional[Tuple[jnp.ndarray, ...]] = None
    reshaped_hidden_states: Optional[Tuple[jnp.ndarray, ...]] = None


@dataclass
class Swinv2EncoderOutput:
    """Encoder output dataclass."""

    last_hidden_state: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray, ...]] = None
    attentions: Optional[Tuple[jnp.ndarray, ...]] = None
    reshaped_hidden_states: Optional[Tuple[jnp.ndarray, ...]] = None


def window_partition(input_feature: jnp.ndarray, window_size: int) -> jnp.ndarray:
    """
    Partitions the given input into windows.
    Args:
        input_feature: (batch_size, height, width, channels)
        window_size: window size
    Returns:
        windows: (num_windows * batch_size, window_size, window_size, channels)
    """
    batch_size, height, width, channels = input_feature.shape
    num_windows_h = height // window_size
    num_windows_w = width // window_size

    input_feature = input_feature.reshape(
        batch_size, num_windows_h, window_size, num_windows_w, window_size, channels
    )
    windows = input_feature.transpose(0, 1, 3, 2, 4, 5)
    windows = windows.reshape(-1, window_size, window_size, channels)
    return windows


def window_reverse(
    windows: jnp.ndarray, window_size: int, height: int, width: int
) -> jnp.ndarray:
    """
    Merges windows back to feature map.
    Args:
        windows: (num_windows * batch_size, window_size, window_size, channels)
        window_size: window size
        height: height of image
        width: width of image
    Returns:
        output: (batch_size, height, width, channels)
    """
    num_windows_h = height // window_size
    num_windows_w = width // window_size
    batch_size = windows.shape[0] // (num_windows_h * num_windows_w)
    channels = windows.shape[-1]

    windows = windows.reshape(
        batch_size, num_windows_h, num_windows_w, window_size, window_size, channels
    )
    output = windows.transpose(0, 1, 3, 2, 4, 5)
    output = output.reshape(batch_size, height, width, channels)
    return output


class DropPath(nn.Module):
    drop_prob: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic=True):
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        # rng = self.make_prng("dropout")
        rng = jax.random.PRNGKey(12)

        random_tensor = jax.random.bernoulli(rng, keep_prob, shape).astype(x.dtype)
        dropped = x / keep_prob * random_tensor

        return jnp.where((self.drop_prob == 0.0) | deterministic, x, dropped)


class LayerNormWithTime(nn.Module):
    """Standard LayerNorm that ignores time parameter for API compatibility."""

    epsilon: float = 1e-5
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, time: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        return nn.LayerNorm(epsilon=self.epsilon, dtype=self.dtype)(x)


class ConditionalLayerNorm(nn.Module):
    """Conditional LayerNorm that modulates based on time."""

    dim: int
    epsilon: float = 1e-5
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, time: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(x**2, axis=-1, keepdims=True) - mean**2
        x_norm = (x - mean) / jnp.sqrt(var + self.epsilon)

        time = time.reshape(-1, 1).astype(self.dtype)
        weight = nn.Dense(self.dim, use_bias=True, dtype=self.dtype, name="weight")(
            time
        )
        bias = nn.Dense(self.dim, use_bias=True, dtype=self.dtype, name="bias")(time)

        # Expand dimensions based on input shape
        # PyTorch: unsqueeze(1) adds dimension at position 1
        # For 3D: (batch, dim) -> (batch, 1, dim) via unsqueeze(1)
        # For 4D: (batch, 1, dim) -> (batch, 1, 1, dim) via another unsqueeze(1)
        weight = weight[:, None, :]  # (batch, 1, dim)
        bias = bias[:, None, :]  # (batch, 1, dim)

        if x.ndim == 4:
            weight = weight[:, :, None, :]  # (batch, 1, 1, dim)
            bias = bias[:, :, None, :]  # (batch, 1, 1, dim)

        result = weight * x_norm + bias

        return result


def get_layer_norm(config: ScOTConfig, dim: int):
    """Factory function to get appropriate layer norm based on config."""
    if config.use_conditioning:
        return ConditionalLayerNorm(dim=dim, epsilon=config.layer_norm_eps)
    else:
        return LayerNormWithTime(epsilon=config.layer_norm_eps)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block implementation in JAX/Flax."""

    config: ScOTConfig
    dim: int
    drop_path: float = 0.0
    layer_scale_init_value: float = 1e-6
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        time: Optional[jnp.ndarray] = None,
        deterministic: Optional[bool] = None,
    ) -> jnp.ndarray:
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        batch_size, sequence_length, hidden_size = x.shape
        input_dim = int(math.floor(sequence_length**0.5))

        input_x = x

        # Reshape to image format
        x = x.reshape(batch_size, input_dim, input_dim, hidden_size)

        # Depthwise conv (groups=dim)
        x = nn.Conv(
            features=self.dim,
            kernel_size=(7, 7),
            padding=((3, 3), (3, 3)),
            feature_group_count=self.dim,
            name="dwconv",
        )(x)

        # LayerNorm
        if self.config.use_conditioning:
            x = ConditionalLayerNorm(
                dim=self.dim, epsilon=self.config.layer_norm_eps, name="norm"
            )(x, time)
        else:
            x = LayerNormWithTime(epsilon=self.config.layer_norm_eps, name="norm")(
                x, time
            )

        # Pointwise convs as linear layers
        x = nn.Dense(4 * self.dim, name="pwconv1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim, name="pwconv2")(x)

        # Layer scale
        if self.layer_scale_init_value > 0:
            gamma = self.param(
                "weight",
                lambda rng, shape: jnp.ones(shape) * self.layer_scale_init_value,
                (self.dim,),
            )
            x = gamma * x

        x = x.reshape(batch_size, sequence_length, hidden_size)

        # Residual with drop path
        x = input_x + DropPath(drop_prob=self.drop_path, name="drop_path")(
            x, deterministic=deterministic
        )
        return x


class ResNetBlock(nn.Module):
    """ResNet Block implementation in JAX/Flax."""

    config: ScOTConfig
    dim: int
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        time: Optional[jnp.ndarray] = None,
        deterministic: Optional[bool] = None,
    ) -> jnp.ndarray:
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        batch_size, sequence_length, hidden_size = x.shape
        input_dim = int(math.floor(sequence_length**0.5))

        input_x = x

        # Reshape to image format (B, H, W, C)
        x = x.reshape(batch_size, input_dim, input_dim, hidden_size)

        # Conv1 + BN + LeakyReLU
        x = nn.Conv(
            features=self.dim, kernel_size=(3, 3), padding="SAME", name="conv1"
        )(x)
        x = nn.BatchNorm(use_running_average=deterministic, name="bn1")(x)
        x = nn.leaky_relu(x)

        # Conv2 + BN
        x = nn.Conv(
            features=self.dim, kernel_size=(3, 3), padding="SAME", name="conv2"
        )(x)
        x = nn.BatchNorm(use_running_average=deterministic, name="bn2")(x)

        x = x.reshape(batch_size, sequence_length, hidden_size)

        # Residual connection
        x = x + input_x
        return x


class ScOTPatchEmbeddings(nn.Module):
    """Patch embedding layer."""

    config: ScOTConfig

    def setup(self):
        image_size = self.config.image_size
        patch_size = self.config.patch_size

        self.image_size = (
            (image_size, image_size) if isinstance(image_size, int) else image_size
        )
        self.patch_size = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (
            self.image_size[1] // self.patch_size[1]
        )
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        self.num_channels = self.config.num_channels

        self.projection = nn.Conv(
            features=self.config.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            name="projection",
        )

    def maybe_pad(
        self, pixel_values: jnp.ndarray, height: int, width: int
    ) -> jnp.ndarray:
        """Pad input if not divisible by patch size."""
        pad_w = (self.patch_size[1] - width % self.patch_size[1]) % self.patch_size[1]
        pad_h = (self.patch_size[0] - height % self.patch_size[0]) % self.patch_size[0]

        if pad_w > 0 or pad_h > 0:
            # pixel_values is (B, H, W, C) in JAX convention
            pixel_values = jnp.pad(
                pixel_values, ((0, 0), (0, pad_h), (0, pad_w), (0, 0))
            )
        return pixel_values

    def __call__(
        self, pixel_values: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Tuple[int, int]]:
        # pixel_values: (B, H, W, C) in JAX convention

        batch_size, height, width, num_channels = pixel_values.shape

        # Pad if needed
        pixel_values = self.maybe_pad(pixel_values, height, width)

        # Project patches
        embeddings = self.projection(pixel_values)

        _, new_height, new_width, _ = embeddings.shape
        output_dimensions = (new_height, new_width)

        # Flatten spatial dimensions
        embeddings = embeddings.reshape(batch_size, -1, self.config.embed_dim)

        return embeddings, output_dimensions


class ScOTEmbeddings(nn.Module):
    """Construct patch and position embeddings."""

    config: ScOTConfig
    use_mask_token: bool = False

    def setup(self):
        self.patch_embeddings = ScOTPatchEmbeddings(self.config)
        num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size

        if self.use_mask_token:
            self.mask_token = self.param(
                "mask_token", initializers.zeros, (1, 1, self.config.embed_dim)
            )

        if self.config.use_absolute_embeddings:
            self.position_embeddings = self.param(
                "position_embeddings",
                initializers.zeros,
                (1, num_patches, self.config.embed_dim),
            )

        if self.config.use_conditioning:
            self.norm = ConditionalLayerNorm(
                dim=self.config.embed_dim, epsilon=self.config.layer_norm_eps
            )
        else:
            self.norm = LayerNormWithTime(epsilon=self.config.layer_norm_eps)

        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(
        self,
        pixel_values: jnp.ndarray,
        bool_masked_pos: Optional[jnp.ndarray] = None,
        time: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, Tuple[int, int]]:
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings, time)
        batch_size, seq_len, _ = embeddings.shape

        if bool_masked_pos is not None and self.use_mask_token:
            mask_tokens = jnp.broadcast_to(
                self.mask_token, (batch_size, seq_len, self.config.embed_dim)
            )
            mask = bool_masked_pos[:, :, None].astype(embeddings.dtype)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        if self.config.use_absolute_embeddings:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings, deterministic=deterministic)

        return embeddings, output_dimensions


class Swinv2RelativePositionBias(nn.Module):
    """Relative position bias for Swin Transformer V2."""

    config: ScOTConfig
    window_size: Tuple[int, int]
    num_heads: int
    pretrained_window_size: Tuple[int, int] = (0, 0)

    def setup(self):
        # MLP for continuous relative position bias
        self.cpb_mlp = [
            nn.Dense(512, use_bias=True, name="cpb_mlp_0"),
            nn.Dense(self.num_heads, use_bias=False, name="cpb_mlp_1"),
        ]

        # Use NUMPY (not jax.numpy) for static computations in setup()
        relative_coords_h = np.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=np.float32
        )
        relative_coords_w = np.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=np.float32
        )
        relative_coords_table = np.stack(
            np.meshgrid(relative_coords_h, relative_coords_w, indexing="ij")
        )
        relative_coords_table = relative_coords_table.transpose(1, 2, 0).reshape(-1, 2)

        # Normalize
        if self.pretrained_window_size[0] > 0:
            relative_coords_table = relative_coords_table / np.array(
                [
                    self.pretrained_window_size[0] - 1,
                    self.pretrained_window_size[1] - 1,
                ],
                dtype=np.float32,
            )
        else:
            # Avoid division by zero
            denom_h = max(self.window_size[0] - 1, 1)
            denom_w = max(self.window_size[1] - 1, 1)
            relative_coords_table = relative_coords_table / np.array(
                [denom_h, denom_w], dtype=np.float32
            )

        relative_coords_table = relative_coords_table * 8
        relative_coords_table = (
            np.sign(relative_coords_table)
            * np.log2(np.abs(relative_coords_table) + 1.0)
            / np.log2(8.0)
        )

        # Store as a regular numpy array (will be converted to JAX array when used)
        self.relative_coords_table = relative_coords_table

        # Create relative position index using numpy
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = coords.reshape(2, -1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose(1, 2, 0)
        relative_coords = relative_coords + np.array(
            [self.window_size[0] - 1, self.window_size[1] - 1]
        )
        relative_coords = relative_coords * np.array([2 * self.window_size[1] - 1, 1])

        # Store as numpy array
        self.relative_position_index = relative_coords.sum(-1).astype(np.int32)

    def __call__(self) -> jnp.ndarray:
        # Convert to JAX arrays here (at call time, not setup time)
        relative_coords_table = jnp.asarray(self.relative_coords_table)
        relative_position_index = jnp.asarray(self.relative_position_index)

        # Apply MLP to get relative position bias
        relative_position_bias = relative_coords_table
        for i, layer in enumerate(self.cpb_mlp):
            relative_position_bias = layer(relative_position_bias)
            if i == 0:
                relative_position_bias = nn.relu(relative_position_bias)

        # Index into the bias table
        relative_position_bias = relative_position_bias.reshape(-1, self.num_heads)
        relative_position_bias = relative_position_bias[
            relative_position_index.reshape(-1)
        ]
        relative_position_bias = relative_position_bias.reshape(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.transpose(2, 0, 1)
        relative_position_bias = 16 * nn.sigmoid(relative_position_bias)

        return relative_position_bias


class Swinv2Attention(nn.Module):
    """Swin Transformer V2 attention."""

    config: ScOTConfig
    dim: int
    num_heads: int
    window_size: Tuple[int, int]
    pretrained_window_size: Tuple[int, int] = (0, 0)

    def setup(self):
        self.head_dim = self.dim // self.num_heads

        # CORRECTION: Split qkv into separate layers to match PyTorch structure
        # PyTorch implementation uses bias=False for Key, but True for Query/Value
        self.query = nn.Dense(self.dim, use_bias=self.config.qkv_bias, name="query")
        self.key = nn.Dense(self.dim, use_bias=False, name="key")
        self.value = nn.Dense(self.dim, use_bias=self.config.qkv_bias, name="value")

        self.attn_drop = nn.Dropout(rate=self.config.attention_probs_dropout_prob)
        self.proj = nn.Dense(self.dim, name="proj")
        self.proj_drop = nn.Dropout(rate=self.config.hidden_dropout_prob)

        self.relative_position_bias = Swinv2RelativePositionBias(
            config=self.config,
            window_size=self.window_size,
            num_heads=self.num_heads,
            pretrained_window_size=self.pretrained_window_size,
        )

        self.logit_scale = self.param(
            "logit_scale",
            lambda rng, shape: jnp.log(10 * jnp.ones(shape)),
            (self.num_heads, 1, 1),
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        head_mask: Optional[jnp.ndarray] = None,
        output_attentions: bool = False,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, ...]:
        batch_size, seq_len, _ = hidden_states.shape

        # CORRECTION: Apply separate layers
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        query = query.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        value = value.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        # Cosine attention (Swin V2)
        query = query / jnp.maximum(
            jnp.linalg.norm(query, axis=-1, keepdims=True), 1e-6
        )
        key = key / jnp.maximum(jnp.linalg.norm(key, axis=-1, keepdims=True), 1e-6)

        logit_scale = jnp.exp(jnp.clip(self.logit_scale))
        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", query, key) * logit_scale

        # Add relative position bias
        relative_position_bias = self.relative_position_bias()
        attn_weights = attn_weights + relative_position_bias[None, :, :, :]

        if attention_mask is not None:
            # attention_mask: (num_windows, window_size*window_size, window_size*window_size)
            num_windows = attention_mask.shape[0]
            attn_weights = attn_weights.reshape(
                batch_size // num_windows, num_windows, self.num_heads, seq_len, seq_len
            )
            attn_weights = attn_weights + attention_mask[None, :, None, :, :]
            attn_weights = attn_weights.reshape(-1, self.num_heads, seq_len, seq_len)

        attn_weights = nn.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_drop(attn_weights, deterministic=deterministic)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, value)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.dim
        )

        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output, deterministic=deterministic)

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class Swinv2Intermediate(nn.Module):
    """Intermediate layer (MLP first half)."""

    config: ScOTConfig
    dim: int

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        hidden_states = nn.Dense(int(self.dim * self.config.mlp_ratio), name="dense")(
            hidden_states
        )
        hidden_states = nn.gelu(hidden_states)
        return hidden_states


class Swinv2Output(nn.Module):
    """Output layer (MLP second half)."""

    config: ScOTConfig
    dim: int

    @nn.compact
    def __call__(
        self, hidden_states: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        hidden_states = nn.Dense(self.dim, name="dense")(hidden_states)
        hidden_states = nn.Dropout(rate=self.config.hidden_dropout_prob)(
            hidden_states, deterministic=deterministic
        )
        return hidden_states


class ScOTLayer(nn.Module):
    """Single Swin Transformer layer."""

    config: ScOTConfig
    dim: int
    input_resolution: Tuple[int, int]
    num_heads: int
    shift_size: int = 0
    drop_path: float = 0.0
    pretrained_window_size: int = 0

    def get_attn_mask(
        self, height: int, width: int, shift_size: int, window_size: int
    ) -> Optional[jnp.ndarray]:
        """Generate attention mask for shifted window attention."""
        if shift_size > 0:
            img_mask = jnp.zeros((1, height, width, 1))
            h_slices = [
                (0, height - window_size),
                (height - window_size, height - shift_size),
                (height - shift_size, height),
            ]
            w_slices = [
                (0, width - window_size),
                (width - window_size, width - shift_size),
                (width - shift_size, width),
            ]
            count = 0
            for h_start, h_end in h_slices:
                for w_start, w_end in w_slices:
                    img_mask = img_mask.at[:, h_start:h_end, w_start:w_end, :].set(
                        count
                    )
                    count += 1
            mask_windows = window_partition(img_mask, window_size)
            mask_windows = mask_windows.reshape(-1, window_size * window_size)
            attn_mask = mask_windows[:, :, None] - mask_windows[:, None, :]
            attn_mask = jnp.where(attn_mask != 0, -100.0, 0.0)
            return attn_mask
        return None

    def maybe_pad(
        self, hidden_states: jnp.ndarray, height: int, width: int, window_size: int
    ) -> Tuple[jnp.ndarray, Tuple[int, ...]]:
        """Pad hidden states to be divisible by window size."""
        pad_right = (window_size - width % window_size) % window_size
        pad_bottom = (window_size - height % window_size) % window_size
        pad_values = ((0, 0), (0, pad_bottom), (0, pad_right), (0, 0))
        if pad_right > 0 or pad_bottom > 0:
            hidden_states = jnp.pad(hidden_states, pad_values)
        return hidden_states, (0, 0, 0, pad_right, 0, pad_bottom)

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        input_dimensions: Tuple[int, int],
        time: Optional[jnp.ndarray] = None,
        head_mask: Optional[jnp.ndarray] = None,
        output_attentions: bool = False,
        always_partition: bool = False,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, ...]:
        height, width = input_dimensions
        batch_size, seq_len, channels = hidden_states.shape
        shortcut = hidden_states

        # Get config window size
        config_window_size = self.config.window_size
        if isinstance(config_window_size, int):
            config_window_size = (config_window_size, config_window_size)

        shift_size_tuple = (
            self.shift_size
            if isinstance(self.shift_size, (list, tuple))
            else (self.shift_size, self.shift_size)
        )

        pretrained_window_size = self.pretrained_window_size
        if isinstance(pretrained_window_size, int):
            pretrained_window_size = (pretrained_window_size, pretrained_window_size)

        # Determine effective window and shift sizes
        window_size = min(config_window_size[0], height, width)
        shift_size = 0 if min(height, width) <= window_size else shift_size_tuple[0]

        # Reshape to spatial
        hidden_states = hidden_states.reshape(batch_size, height, width, channels)
        hidden_states, pad_values = self.maybe_pad(
            hidden_states, height, width, window_size
        )
        _, height_pad, width_pad, _ = hidden_states.shape

        # Cyclic shift
        if shift_size > 0:
            shifted_hidden_states = jnp.roll(
                hidden_states, shift=(-shift_size, -shift_size), axis=(1, 2)
            )
        else:
            shifted_hidden_states = hidden_states

        # Partition windows
        hidden_states_windows = window_partition(shifted_hidden_states, window_size)
        hidden_states_windows = hidden_states_windows.reshape(
            -1, window_size * window_size, channels
        )

        # Attention mask
        attn_mask = self.get_attn_mask(height_pad, width_pad, shift_size, window_size)

        # Create attention with EFFECTIVE window size (inside @nn.compact, this is allowed)
        attention = Swinv2Attention(
            config=self.config,
            dim=self.dim,
            num_heads=self.num_heads,
            window_size=(window_size, window_size),
            pretrained_window_size=pretrained_window_size,
            name="attention",
        )  # Use effective size

        attention_outputs = attention(
            hidden_states_windows,
            attention_mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            deterministic=deterministic,
        )
        attention_output = attention_outputs[0]

        # Merge windows
        attention_windows = attention_output.reshape(
            -1, window_size, window_size, channels
        )
        shifted_windows = window_reverse(
            attention_windows, window_size, height_pad, width_pad
        )

        # Reverse cyclic shift
        if shift_size > 0:
            attention_windows = jnp.roll(
                shifted_windows, shift=(shift_size, shift_size), axis=(1, 2)
            )
        else:
            attention_windows = shifted_windows

        # Remove padding
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :]

        attention_windows = attention_windows.reshape(
            batch_size, height * width, channels
        )

        # Layer norms
        if self.config.use_conditioning:
            layernorm_before = ConditionalLayerNorm(
                dim=self.dim,
                epsilon=self.config.layer_norm_eps,
                name="layernorm_before",
            )
            layernorm_after = ConditionalLayerNorm(
                dim=self.dim, epsilon=self.config.layer_norm_eps, name="layernorm_after"
            )
        else:
            layernorm_before = LayerNormWithTime(
                epsilon=self.config.layer_norm_eps, name="layernorm_before"
            )
            layernorm_after = LayerNormWithTime(
                epsilon=self.config.layer_norm_eps, name="layernorm_after"
            )

        drop_path_layer = DropPath(drop_prob=self.drop_path, name="drop_path")

        # First residual
        hidden_states = shortcut + drop_path_layer(
            layernorm_before(attention_windows, time), deterministic=deterministic
        )

        # MLP
        intermediate = Swinv2Intermediate(
            config=self.config, dim=self.dim, name="intermediate"
        )
        output_layer = Swinv2Output(config=self.config, dim=self.dim, name="output")

        residual = hidden_states
        layer_output = output_layer(
            intermediate(hidden_states), deterministic=deterministic
        )
        layer_output = residual + drop_path_layer(
            layernorm_after(layer_output, time), deterministic=deterministic
        )

        outputs = (
            (layer_output, attention_outputs[1])
            if output_attentions
            else (layer_output,)
        )
        return outputs


class ScOTPatchMerging(nn.Module):
    """Patch merging layer for downsampling."""

    config: ScOTConfig
    input_resolution: Tuple[int, int]
    dim: int

    def setup(self):
        self.reduction = nn.Dense(2 * self.dim, use_bias=False, name="reduction")
        if self.config.use_conditioning:
            self.norm = ConditionalLayerNorm(
                dim=2 * self.dim, epsilon=self.config.layer_norm_eps
            )
        else:
            self.norm = LayerNormWithTime(epsilon=self.config.layer_norm_eps)

    def maybe_pad(
        self, input_feature: jnp.ndarray, height: int, width: int
    ) -> jnp.ndarray:
        """Pad if height or width is odd."""
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_values = ((0, 0), (0, height % 2), (0, width % 2), (0, 0))
            input_feature = jnp.pad(input_feature, pad_values)
        return input_feature

    def __call__(
        self,
        input_feature: jnp.ndarray,
        input_dimensions: Tuple[int, int],
        time: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        height, width = input_dimensions
        batch_size, dim, num_channels = input_feature.shape

        input_feature = input_feature.reshape(batch_size, height, width, num_channels)
        input_feature = self.maybe_pad(input_feature, height, width)

        # Merge 2x2 patches
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]

        input_feature = jnp.concatenate(
            [input_feature_0, input_feature_1, input_feature_2, input_feature_3],
            axis=-1,
        )
        input_feature = input_feature.reshape(batch_size, -1, 4 * num_channels)

        input_feature = self.reduction(input_feature)
        input_feature = self.norm(input_feature, time)

        return input_feature


class ScOTPatchUnmerging(nn.Module):
    """Patch unmerging layer for upsampling."""

    config: ScOTConfig
    input_resolution: Tuple[int, int]
    dim: int

    def setup(self):
        self.upsample = nn.Dense(2 * self.dim, use_bias=False, name="upsample")
        self.mixup = nn.Dense(self.dim // 2, use_bias=False, name="mixup")
        if self.config.use_conditioning:
            self.norm = ConditionalLayerNorm(
                dim=self.dim // 2, epsilon=self.config.layer_norm_eps
            )
        else:
            self.norm = LayerNormWithTime(epsilon=self.config.layer_norm_eps)

    def maybe_crop(
        self, input_feature: jnp.ndarray, height: int, width: int
    ) -> jnp.ndarray:
        """Crop if dimensions exceed target."""
        height_in, width_in = input_feature.shape[1], input_feature.shape[2]
        if height_in > height:
            input_feature = input_feature[:, :height, :, :]
        if width_in > width:
            input_feature = input_feature[:, :, :width, :]
        return input_feature

    def __call__(
        self,
        input_feature: jnp.ndarray,
        output_dimensions: Tuple[int, int],
        time: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        output_height, output_width = output_dimensions
        batch_size, seq_len, hidden_size = input_feature.shape
        input_height = input_width = int(math.floor(seq_len**0.5))

        input_feature = self.upsample(input_feature)
        input_feature = input_feature.reshape(
            batch_size, input_height, input_width, 2, 2, hidden_size // 2
        )
        input_feature = input_feature.transpose(0, 1, 3, 2, 4, 5)
        input_feature = input_feature.reshape(
            batch_size, 2 * input_height, 2 * input_width, hidden_size // 2
        )

        input_feature = self.maybe_crop(input_feature, output_height, output_width)
        input_feature = input_feature.reshape(batch_size, -1, hidden_size // 2)

        input_feature = self.norm(input_feature, time)
        return self.mixup(input_feature)


class ScOTPatchRecovery(nn.Module):
    """Patch recovery layer to reconstruct output image."""

    config: ScOTConfig

    def setup(self):
        image_size = self.config.image_size
        patch_size = self.config.patch_size

        self.image_size = (
            (image_size, image_size) if isinstance(image_size, int) else image_size
        )
        self.patch_size = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )

        self.projection = nn.ConvTranspose(
            features=self.config.num_out_channels,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            name="projection",
        )
        self.mixup = nn.Conv(
            features=self.config.num_out_channels,
            kernel_size=(5, 5),
            padding="SAME",
            use_bias=False,
            name="mixup",
        )

    def maybe_crop(
        self, pixel_values: jnp.ndarray, height: int, width: int
    ) -> jnp.ndarray:
        """Crop output to target size."""
        if pixel_values.shape[1] > height:
            pixel_values = pixel_values[:, :height, :, :]
        if pixel_values.shape[2] > width:
            pixel_values = pixel_values[:, :, :width, :]
        return pixel_values

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Reshape to spatial (B, H, W, C)
        hidden_states = hidden_states.reshape(batch_size, *self.grid_size, hidden_size)

        # Transposed convolution
        output = self.projection(hidden_states)
        output = self.maybe_crop(output, self.image_size[0], self.image_size[1])

        # Mixup conv
        return self.mixup(output)


class ScOTEncodeStage(nn.Module):
    """Single encoder stage."""

    config: ScOTConfig
    dim: int
    input_resolution: Tuple[int, int]
    depth: int
    num_heads: int
    drop_path: Sequence[float]
    downsample: bool
    pretrained_window_size: int = 0

    def setup(self):
        window_size = self.config.window_size
        if isinstance(window_size, int):
            window_size = (window_size, window_size)

        self.blocks = [
            ScOTLayer(
                config=self.config,
                dim=self.dim,
                input_resolution=self.input_resolution,
                num_heads=self.num_heads,
                shift_size=(
                    [0, 0]
                    if (i % 2 == 0)
                    else [window_size[0] // 2, window_size[1] // 2]
                ),
                drop_path=self.drop_path[i],
                pretrained_window_size=self.pretrained_window_size,
                name=f"block_{i}",
            )
            for i in range(self.depth)
        ]

        if self.downsample:
            self.downsample_layer = ScOTPatchMerging(
                config=self.config,
                input_resolution=self.input_resolution,
                dim=self.dim,
                name="downsample",
            )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        input_dimensions: Tuple[int, int],
        time: Optional[jnp.ndarray] = None,
        head_mask: Optional[jnp.ndarray] = None,
        output_attentions: bool = False,
        always_partition: bool = False,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, ...]:
        height, width = input_dimensions
        inputs = hidden_states

        for i, block in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = block(
                hidden_states,
                input_dimensions,
                time,
                layer_head_mask,
                output_attentions,
                always_partition,
                deterministic,
            )
            hidden_states = layer_outputs[0]

        hidden_states_before_downsampling = hidden_states

        if self.downsample:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample_layer(
                hidden_states_before_downsampling + inputs, input_dimensions, time
            )
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (
            hidden_states,
            hidden_states_before_downsampling,
            output_dimensions,
        )

        if output_attentions:
            stage_outputs = stage_outputs + (layer_outputs[1],)

        return stage_outputs


class ScOTDecodeStage(nn.Module):
    """Single decoder stage."""

    config: ScOTConfig
    dim: int
    input_resolution: Tuple[int, int]
    depth: int
    num_heads: int
    drop_path: Sequence[float]
    upsample: bool
    upsampled_size: Tuple[int, int]
    pretrained_window_size: int = 0

    def setup(self):
        window_size = self.config.window_size
        if isinstance(window_size, int):
            window_size = (window_size, window_size)

        self.blocks = [
            ScOTLayer(
                config=self.config,
                dim=self.dim,
                input_resolution=self.input_resolution,
                num_heads=self.num_heads,
                shift_size=(
                    [0, 0]
                    if (i % 2 == 0)
                    else [window_size[0] // 2, window_size[1] // 2]
                ),
                drop_path=self.drop_path[self.depth - 1 - i],
                pretrained_window_size=self.pretrained_window_size,
                name=f"block_{i}",
            )
            for i in reversed(range(self.depth))
        ]

        if self.upsample:
            self.upsample_layer = ScOTPatchUnmerging(
                config=self.config,
                input_resolution=self.input_resolution,
                dim=self.dim,
                name="upsample",
            )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        input_dimensions: Tuple[int, int],
        time: Optional[jnp.ndarray] = None,
        head_mask: Optional[jnp.ndarray] = None,
        output_attentions: bool = False,
        always_partition: bool = False,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, ...]:
        height, width = input_dimensions

        for i, block in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = block(
                hidden_states,
                input_dimensions,
                time,
                layer_head_mask,
                output_attentions,
                always_partition,
                deterministic,
            )
            hidden_states = layer_outputs[0]

        hidden_states_before_upsampling = hidden_states

        if self.upsample:
            height_upsampled, width_upsampled = self.upsampled_size
            output_dimensions = (height, width, height_upsampled, width_upsampled)
            hidden_states = self.upsample_layer(
                hidden_states_before_upsampling,
                (height_upsampled, width_upsampled),
                time,
            )
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (
            hidden_states,
            hidden_states_before_upsampling,
            output_dimensions,
        )

        if output_attentions:
            stage_outputs = stage_outputs + (layer_outputs[1],)

        return stage_outputs


class ScOTEncoder(nn.Module):
    """Full encoder."""

    config: ScOTConfig
    grid_size: Tuple[int, int]

    def setup(self):
        num_layers = len(self.config.depths)
        pretrained_window_sizes = self.config.pretrained_window_sizes

        # Compute drop path rates
        total_depth = 2 * sum(self.config.depths)
        drop_rates = jnp.linspace(0, self.config.drop_path_rate, total_depth)
        dpr = [x.astype(float) for x in drop_rates[: total_depth // 2]]

        self.layers = [
            ScOTEncodeStage(
                config=self.config,
                dim=int(self.config.embed_dim * 2**i_layer),
                input_resolution=(
                    self.grid_size[0] // (2**i_layer),
                    self.grid_size[1] // (2**i_layer),
                ),
                depth=self.config.depths[i_layer],
                num_heads=self.config.num_heads[i_layer],
                drop_path=dpr[
                    sum(self.config.depths[:i_layer]) : sum(
                        self.config.depths[: i_layer + 1]
                    )
                ],
                downsample=(i_layer < num_layers - 1),
                pretrained_window_size=pretrained_window_sizes[i_layer],
                name=f"layer_{i_layer}",
            )
            for i_layer in range(num_layers)
        ]

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        input_dimensions: Tuple[int, int],
        time: Optional[jnp.ndarray] = None,
        head_mask: Optional[jnp.ndarray] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_hidden_states_before_downsampling: bool = False,
        always_partition: bool = False,
        deterministic: bool = True,
        return_dict: bool = True,
    ) -> Union[Tuple, Swinv2EncoderOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            reshaped_hidden_state = hidden_states.reshape(
                batch_size, *input_dimensions, hidden_size
            )
            reshaped_hidden_state = reshaped_hidden_state.transpose(0, 3, 1, 2)
            all_hidden_states = all_hidden_states + (hidden_states,)
            all_reshaped_hidden_states = all_reshaped_hidden_states + (
                reshaped_hidden_state,
            )

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                time,
                layer_head_mask,
                output_attentions,
                always_partition,
                deterministic,
            )

            hidden_states = layer_outputs[0]
            hidden_states_before_downsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            if output_hidden_states and output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states_before_downsampling.shape
                reshaped_hidden_state = hidden_states_before_downsampling.reshape(
                    batch_size, output_dimensions[0], output_dimensions[1], hidden_size
                )
                reshaped_hidden_state = reshaped_hidden_state.transpose(0, 3, 1, 2)
                all_hidden_states = all_hidden_states + (
                    hidden_states_before_downsampling,
                )
                all_reshaped_hidden_states = all_reshaped_hidden_states + (
                    reshaped_hidden_state,
                )
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states.shape
                reshaped_hidden_state = hidden_states.reshape(
                    batch_size, *input_dimensions, hidden_size
                )
                reshaped_hidden_state = reshaped_hidden_state.transpose(0, 3, 1, 2)
                all_hidden_states = all_hidden_states + (hidden_states,)
                all_reshaped_hidden_states = all_reshaped_hidden_states + (
                    reshaped_hidden_state,
                )

            if output_attentions and len(layer_outputs) > 3:
                all_self_attentions = all_self_attentions + (layer_outputs[3],)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return Swinv2EncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )


class ScOTDecoder(nn.Module):
    """Full decoder."""

    config: ScOTConfig
    grid_size: Tuple[int, int]

    def setup(self):
        num_layers = len(self.config.depths)
        pretrained_window_sizes = self.config.pretrained_window_sizes

        # Compute drop path rates
        total_depth = 2 * sum(self.config.depths)
        drop_rates = jnp.linspace(0, self.config.drop_path_rate, total_depth)
        dpr = [x.astype(float) for x in drop_rates[total_depth // 2 :]]

        self.layers = [
            ScOTDecodeStage(
                config=self.config,
                dim=int(self.config.embed_dim * 2**i_layer),
                input_resolution=(
                    self.grid_size[0] // (2**i_layer),
                    self.grid_size[1] // (2**i_layer),
                ),
                depth=self.config.depths[i_layer],
                num_heads=self.config.num_heads[i_layer],
                drop_path=dpr[
                    sum(self.config.depths[i_layer + 1 :]) : sum(
                        self.config.depths[i_layer:]
                    )
                ],
                upsample=(i_layer > 0),
                upsampled_size=(
                    self.grid_size[0] // (2 ** (i_layer - 1)),
                    self.grid_size[1] // (2 ** (i_layer - 1)),
                ),
                pretrained_window_size=pretrained_window_sizes[i_layer],
                name=f"layer_{i_layer}",
            )
            for i_layer in reversed(range(num_layers))
        ]

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        input_dimensions: Tuple[int, int],
        skip_states: List[jnp.ndarray],
        time: Optional[jnp.ndarray] = None,
        head_mask: Optional[jnp.ndarray] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_hidden_states_before_upsampling: bool = False,
        always_partition: bool = False,
        deterministic: bool = True,
        return_dict: bool = True,
    ) -> Union[Tuple, Swinv2EncoderOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            reshaped_hidden_state = hidden_states.reshape(
                batch_size, *input_dimensions, hidden_size
            )
            reshaped_hidden_state = reshaped_hidden_state.transpose(0, 3, 1, 2)
            all_hidden_states = all_hidden_states + (hidden_states,)
            all_reshaped_hidden_states = all_reshaped_hidden_states + (
                reshaped_hidden_state,
            )

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # Add skip connection
            if i != 0 and len(skip_states) > len(skip_states) - i:
                skip_idx = len(skip_states) - i
                if (
                    skip_idx >= 0
                    and skip_idx < len(skip_states)
                    and skip_states[skip_idx] is not None
                ):
                    hidden_states = hidden_states + skip_states[skip_idx]

            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                time,
                layer_head_mask,
                output_attentions,
                always_partition,
                deterministic,
            )

            hidden_states = layer_outputs[0]

            hidden_states = layer_outputs[0]
            hidden_states_before_upsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            if output_hidden_states and output_hidden_states_before_upsampling:
                batch_size, _, hidden_size = hidden_states_before_upsampling.shape
                reshaped_hidden_state = hidden_states_before_upsampling.reshape(
                    batch_size, output_dimensions[0], output_dimensions[1], hidden_size
                )
                reshaped_hidden_state = reshaped_hidden_state.transpose(0, 3, 1, 2)
                all_hidden_states = all_hidden_states + (
                    hidden_states_before_upsampling,
                )
                all_reshaped_hidden_states = all_reshaped_hidden_states + (
                    reshaped_hidden_state,
                )
            elif output_hidden_states and not output_hidden_states_before_upsampling:
                batch_size, _, hidden_size = hidden_states.shape
                reshaped_hidden_state = hidden_states.reshape(
                    batch_size, *input_dimensions, hidden_size
                )
                reshaped_hidden_state = reshaped_hidden_state.transpose(0, 3, 1, 2)
                all_hidden_states = all_hidden_states + (hidden_states,)
                all_reshaped_hidden_states = all_reshaped_hidden_states + (
                    reshaped_hidden_state,
                )

            if output_attentions and len(layer_outputs) > 3:
                all_self_attentions = all_self_attentions + (layer_outputs[3],)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return Swinv2EncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )


class ResidualBlockWrapper(nn.Module):
    """Wrapper for residual blocks (ConvNeXt or ResNet)."""

    config: ScOTConfig
    dim: int
    depth: int
    drop_path: float = 0.0

    def setup(self):
        if self.depth > 0:
            if self.config.residual_model == "convnext":
                self.blocks = [
                    ConvNeXtBlock(
                        config=self.config,
                        dim=self.dim,
                        drop_path=self.drop_path,
                        name=f"block_{i}",
                    )
                    for i in range(self.depth)
                ]
            elif self.config.residual_model == "resnet":
                self.blocks = [
                    ResNetBlock(config=self.config, dim=self.dim, name=f"block_{i}")
                    for i in range(self.depth)
                ]
            else:
                raise ValueError("residual_model must be 'convnext' or 'resnet'")
        else:
            self.blocks = None

    def __call__(
        self,
        x: jnp.ndarray,
        time: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        if self.blocks is None:
            return x
        for block in self.blocks:
            x = block(x, time, deterministic=deterministic)
        return x


class ScOT(nn.Module):
    """Main ScOT model - Swin Transformer V2 based encoder-decoder."""

    config: ScOTConfig
    use_mask_token: bool = False
    use_conditioning: bool = False

    def setup(self):
        self.num_layers_encoder = len(self.config.depths)
        self.num_layers_decoder = len(self.config.depths)
        self.num_features = int(
            self.config.embed_dim * 2 ** (self.num_layers_encoder - 1)
        )

        self.embeddings = ScOTEmbeddings(
            self.config, use_mask_token=self.use_mask_token
        )

        # Get patch grid from embeddings
        image_size = self.config.image_size
        patch_size = self.config.patch_size
        image_size = (
            (image_size, image_size) if isinstance(image_size, int) else image_size
        )
        patch_size = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )
        self.patch_grid = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )

        self.encoder = ScOTEncoder(self.config, self.patch_grid)
        self.decoder = ScOTDecoder(self.config, self.patch_grid)
        self.patch_recovery = ScOTPatchRecovery(self.config)

        # Residual blocks for skip connections
        skip_connections = self.config.skip_connections
        self.residual_blocks = [
            ResidualBlockWrapper(
                config=self.config,
                dim=self.config.embed_dim * 2**i,
                depth=int(depth) if depth else 0,
                name=f"residual_block_{i}",
            )
            for i, depth in enumerate(skip_connections)
        ]

    def _downsample(self, image: jnp.ndarray, target_size: int) -> jnp.ndarray:
        """Downsample image using FFT."""
        image_size = image.shape[1]  # Assuming (B, H, W, C)
        freqs = jnp.fft.fftfreq(image_size, d=1 / image_size)
        sel = jnp.logical_and(freqs >= -target_size / 2, freqs <= target_size / 2 - 1)

        # FFT on spatial dimensions
        image_hat = jnp.fft.fft2(image, axes=(1, 2), norm="forward")

        # Select frequencies
        image_hat = image_hat[:, sel, :, :][:, :, sel, :]

        # Inverse FFT
        image = jnp.fft.ifft2(image_hat, axes=(1, 2), norm="forward").real
        return image

    def _upsample(self, image: jnp.ndarray, target_size: int) -> jnp.ndarray:
        """Upsample image using FFT."""
        image_size = image.shape[1]  # Assuming (B, H, W, C)

        image_hat = jnp.fft.fft2(image, axes=(1, 2), norm="forward")
        image_hat = jnp.fft.fftshift(image_hat, axes=(1, 2))

        pad_size = (target_size - image_size) // 2

        # Pad in frequency domain
        real = jnp.pad(
            image_hat.real, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0))
        )
        imag = jnp.pad(
            image_hat.imag, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0))
        )

        image_hat = jnp.fft.ifftshift(real + 1j * imag, axes=(1, 2))
        image = jnp.fft.ifft2(image_hat, axes=(1, 2), norm="forward").real
        return image

    def __call__(
        self,
        pixel_values: jnp.ndarray,
        time: Optional[jnp.ndarray] = None,
        bool_masked_pos: Optional[jnp.ndarray] = None,
        head_mask: Optional[jnp.ndarray] = None,
        pixel_mask: Optional[jnp.ndarray] = None,
        labels: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        deterministic: bool = True,
    ) -> Union[Tuple, ScOTOutput]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        # pixel_values expected as (B, H, W, C) in JAX
        image_size = pixel_values.shape[1]
        original_pixel_values = pixel_values

        # Handle image size mismatch
        if image_size != self.config.image_size:
            if image_size < self.config.image_size:
                pixel_values = self._upsample(pixel_values, self.config.image_size)
            else:
                pixel_values = self._downsample(pixel_values, self.config.image_size)

        # Embeddings
        embedding_output, input_dimensions = self.embeddings(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            time=time,
            deterministic=deterministic,
        )

        # Encoder
        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            time,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            output_hidden_states_before_downsampling=True,
            deterministic=deterministic,
            return_dict=True,
        )

        # Get skip states
        skip_states = (
            list(encoder_outputs.hidden_states[1:])
            if encoder_outputs.hidden_states
            else []
        )

        # Apply residual blocks to skip states
        for i in range(len(skip_states)):
            if i < len(self.residual_blocks):
                skip_states[i] = self.residual_blocks[i](
                    skip_states[i], time, deterministic=deterministic
                )

        # Decoder input dimensions
        input_dim = (
            int(math.floor(skip_states[-1].shape[1] ** 0.5))
            if skip_states
            else input_dimensions[0]
        )

        decoder_output = self.decoder(
            skip_states[-1] if skip_states else encoder_outputs.last_hidden_state,
            (input_dim, input_dim),
            skip_states=skip_states[:-1] if skip_states else [],
            time=time,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            deterministic=deterministic,
            return_dict=True,
        )

        # Patch recovery
        sequence_output = decoder_output.last_hidden_state
        prediction = self.patch_recovery(sequence_output)

        # Learn residual if configured
        if self.config.learn_residual:
            if self.config.num_channels > self.config.num_out_channels:
                residual_input = original_pixel_values[
                    :, :, :, : self.config.num_out_channels
                ]
            else:
                residual_input = original_pixel_values

            # Handle size mismatch for residual
            if image_size != self.config.image_size:
                if image_size < self.config.image_size:
                    residual_input = self._upsample(
                        residual_input, self.config.image_size
                    )
                else:
                    residual_input = self._downsample(
                        residual_input, self.config.image_size
                    )

            prediction = prediction + residual_input

        # Handle output size mismatch
        if image_size != self.config.image_size:
            if image_size > self.config.image_size:
                prediction = self._upsample(prediction, image_size)
            else:
                prediction = self._downsample(prediction, image_size)

        # Apply pixel mask if provided
        if pixel_mask is not None and labels is not None:
            prediction = jnp.where(pixel_mask, labels, prediction)

        # Compute loss
        loss = None
        if labels is not None:
            if self.config.p == 1:
                loss_fn = lambda pred, target: jnp.mean(jnp.abs(pred - target))
            elif self.config.p == 2:
                loss_fn = lambda pred, target: jnp.mean((pred - target) ** 2)
            else:
                raise ValueError("p must be 1 or 2")

            if self.config.channel_slice_list_normalized_loss is not None:
                channel_slices = self.config.channel_slice_list_normalized_loss
                losses = []
                for i in range(len(channel_slices) - 1):
                    pred_slice = prediction[
                        :, :, :, channel_slices[i] : channel_slices[i + 1]
                    ]
                    label_slice = labels[
                        :, :, :, channel_slices[i] : channel_slices[i + 1]
                    ]
                    numerator = loss_fn(pred_slice, label_slice)
                    denominator = (
                        loss_fn(label_slice, jnp.zeros_like(label_slice)) + 1e-10
                    )
                    losses.append(numerator / denominator)
                loss = jnp.mean(jnp.stack(losses))
            else:
                loss = loss_fn(prediction, labels)

        if not return_dict:
            output = (prediction,)
            if output_hidden_states:
                output = output + (
                    decoder_output.hidden_states,
                    encoder_outputs.hidden_states,
                )
            if output_attentions:
                output = output + (
                    decoder_output.attentions,
                    encoder_outputs.attentions,
                )
            return ((loss,) + output) if loss is not None else output

        return ScOTOutput(
            loss=loss,
            output=prediction,
            hidden_states=(
                (decoder_output.hidden_states or ())
                + (encoder_outputs.hidden_states or ())
                if output_hidden_states
                else None
            ),
            attentions=(
                (decoder_output.attentions or ()) + (encoder_outputs.attentions or ())
                if output_attentions
                else None
            ),
            reshaped_hidden_states=(
                (decoder_output.reshaped_hidden_states or ())
                + (encoder_outputs.reshaped_hidden_states or ())
                if output_hidden_states
                else None
            ),
        )
