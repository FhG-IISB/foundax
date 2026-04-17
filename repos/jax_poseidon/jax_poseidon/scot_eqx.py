"""
JAX/Equinox implementation of ScOT (Swin Transformer V2 based encoder-decoder).
Converted from the Flax Linen implementation while maintaining exact architecture.

Github repository of original implementation: https://github.com/camlab-ethz/poseidon
Arxiv paper: https://arxiv.org/abs/2405.19101
"""

import jax
import jax.numpy as jnp
import equinox as eqx
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


# ---------------------------------------------------------------------------
# Equinox helper: NHWC Conv2d wrapper
# ---------------------------------------------------------------------------


class Conv2dNHWC(eqx.Module):
    """Conv2d that operates on NHWC tensors (matching Flax convention)."""

    conv: eqx.nn.Conv2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[str, int, Tuple[Tuple[int, int], Tuple[int, int]]] = 0,
        groups: int = 1,
        use_bias: bool = True,
        *,
        key: jax.Array,
    ):
        if isinstance(padding, str) and padding.upper() == "SAME":
            # Compute same padding
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            pad_h = (kernel_size[0] - 1) // 2
            pad_w = (kernel_size[1] - 1) // 2
            padding_val = (
                (pad_h, kernel_size[0] - 1 - pad_h),
                (pad_w, kernel_size[1] - 1 - pad_w),
            )
        elif (
            isinstance(padding, tuple)
            and len(padding) == 2
            and isinstance(padding[0], tuple)
        ):
            padding_val = padding
        elif isinstance(padding, int):
            padding_val = padding
        else:
            padding_val = padding

        self.conv = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding_val,
            groups=groups,
            use_bias=use_bias,
            key=key,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, H, W, C) -> (B, C, H, W) for eqx.nn.Conv2d
        x = x.transpose(0, 3, 1, 2)
        # Conv2d expects unbatched input, so vmap over batch
        x = jax.vmap(self.conv)(x)
        # (B, C_out, H, W) -> (B, H, W, C_out)
        return x.transpose(0, 2, 3, 1)


class ConvTranspose2dNHWC(eqx.Module):
    """ConvTranspose2d that operates on NHWC tensors."""

    conv: eqx.nn.ConvTranspose2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[str, int, Tuple[Tuple[int, int], Tuple[int, int]]] = 0,
        use_bias: bool = True,
        *,
        key: jax.Array,
    ):
        self.conv = eqx.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            use_bias=use_bias,
            key=key,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, H, W, C) -> (B, C, H, W)
        x = x.transpose(0, 3, 1, 2)
        x = jax.vmap(self.conv)(x)
        # (B, C_out, H, W) -> (B, H, W, C_out)
        return x.transpose(0, 2, 3, 1)


# ---------------------------------------------------------------------------
# DropPath
# ---------------------------------------------------------------------------


class DropPath(eqx.Module):
    drop_prob: float = eqx.field(static=True)

    def __init__(self, drop_prob: float = 0.0):
        self.drop_prob = drop_prob

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        if (self.drop_prob == 0.0) or deterministic:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rng = jax.random.PRNGKey(12)
        random_tensor = jax.random.bernoulli(rng, keep_prob, shape).astype(x.dtype)
        return x / keep_prob * random_tensor


# ---------------------------------------------------------------------------
# Layer Norms
# ---------------------------------------------------------------------------


class LayerNormWithTime(eqx.Module):
    """Standard LayerNorm that ignores time parameter."""

    norm: eqx.nn.LayerNorm

    def __init__(self, dim: int, epsilon: float = 1e-5):
        self.norm = eqx.nn.LayerNorm(shape=dim, eps=epsilon)

    def __call__(
        self, x: jnp.ndarray, time: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        # eqx.nn.LayerNorm expects unbatched input on last axis
        # We vmap over all leading dims
        if x.ndim == 3:
            # (B, S, C) -> vmap over B and S
            return jax.vmap(jax.vmap(self.norm))(x)
        elif x.ndim == 4:
            # (B, H, W, C) -> vmap over B, H, W
            return jax.vmap(jax.vmap(jax.vmap(self.norm)))(x)
        elif x.ndim == 2:
            # (B, C)
            return jax.vmap(self.norm)(x)
        else:
            return x


class ConditionalLayerNorm(eqx.Module):
    """Conditional LayerNorm that modulates based on time."""

    dim: int = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    weight_dense: eqx.nn.Linear
    bias_dense: eqx.nn.Linear

    def __init__(self, dim: int, epsilon: float = 1e-5, *, key: jax.Array):
        self.dim = dim
        self.epsilon = epsilon
        key1, key2 = jax.random.split(key)
        self.weight_dense = eqx.nn.Linear(1, dim, use_bias=True, key=key1)
        self.bias_dense = eqx.nn.Linear(1, dim, use_bias=True, key=key2)

    def __call__(
        self, x: jnp.ndarray, time: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(x**2, axis=-1, keepdims=True) - mean**2
        x_norm = (x - mean) / jnp.sqrt(var + self.epsilon)

        time = time.reshape(-1, 1).astype(x.dtype)
        weight = jax.vmap(self.weight_dense)(time)  # (batch, dim)
        bias = jax.vmap(self.bias_dense)(time)  # (batch, dim)

        weight = weight[:, None, :]  # (batch, 1, dim)
        bias = bias[:, None, :]  # (batch, 1, dim)

        if x.ndim == 4:
            weight = weight[:, :, None, :]  # (batch, 1, 1, dim)
            bias = bias[:, :, None, :]  # (batch, 1, 1, dim)

        return weight * x_norm + bias


def make_layer_norm(config: ScOTConfig, dim: int, *, key: jax.Array):
    """Factory function to get appropriate layer norm based on config."""
    if config.use_conditioning:
        return ConditionalLayerNorm(dim=dim, epsilon=config.layer_norm_eps, key=key)
    else:
        return LayerNormWithTime(dim=dim, epsilon=config.layer_norm_eps)


# ---------------------------------------------------------------------------
# ConvNeXt Block
# ---------------------------------------------------------------------------


class ConvNeXtBlock(eqx.Module):
    config: ScOTConfig = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    drop_path_rate: float = eqx.field(static=True)
    layer_scale_init_value: float = eqx.field(static=True)

    dwconv: Conv2dNHWC
    norm: Union[ConditionalLayerNorm, LayerNormWithTime]
    pwconv1: eqx.nn.Linear
    pwconv2: eqx.nn.Linear
    weight: Optional[jnp.ndarray]
    drop_path_layer: DropPath

    def __init__(
        self,
        config: ScOTConfig,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        *,
        key: jax.Array,
    ):
        self.config = config
        self.dim = dim
        self.drop_path_rate = drop_path
        self.layer_scale_init_value = layer_scale_init_value

        keys = jax.random.split(key, 4)

        self.dwconv = Conv2dNHWC(
            in_channels=dim,
            out_channels=dim,
            kernel_size=(7, 7),
            padding=((3, 3), (3, 3)),
            groups=dim,
            key=keys[0],
        )
        self.norm = make_layer_norm(config, dim, key=keys[1])
        self.pwconv1 = eqx.nn.Linear(dim, 4 * dim, key=keys[2])
        self.pwconv2 = eqx.nn.Linear(4 * dim, dim, key=keys[3])

        if layer_scale_init_value > 0:
            self.weight = jnp.ones(dim) * layer_scale_init_value
        else:
            self.weight = None

        self.drop_path_layer = DropPath(drop_prob=drop_path)

    def __call__(
        self,
        x: jnp.ndarray,
        time: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        batch_size, sequence_length, hidden_size = x.shape
        input_dim = int(math.floor(sequence_length**0.5))
        input_x = x

        # Reshape to image format
        x = x.reshape(batch_size, input_dim, input_dim, hidden_size)

        # Depthwise conv
        x = self.dwconv(x)

        # Norm
        x = self.norm(x, time)

        # Pointwise convs as linear layers (applied to last dim)
        x = jax.vmap(jax.vmap(jax.vmap(self.pwconv1)))(x)
        x = jax.nn.gelu(x)
        x = jax.vmap(jax.vmap(jax.vmap(self.pwconv2)))(x)

        # Layer scale
        if self.weight is not None:
            x = self.weight * x

        x = x.reshape(batch_size, sequence_length, hidden_size)
        x = input_x + self.drop_path_layer(x, deterministic=deterministic)
        return x


# ---------------------------------------------------------------------------
# ResNet Block
# ---------------------------------------------------------------------------


class _BatchNorm(eqx.Module):
    """Stateless batchnorm for compatibility."""

    weight: jnp.ndarray
    bias: jnp.ndarray
    eps: float = eqx.field(static=True)

    def __init__(self, num_features: int, eps: float = 1e-5):
        self.weight = jnp.ones(num_features)
        self.bias = jnp.zeros(num_features)
        self.eps = eps

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        # x: (B, H, W, C)
        axes = tuple(range(x.ndim - 1))
        mean = jnp.mean(x, axis=axes, keepdims=True)
        var = jnp.var(x, axis=axes, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        return x * self.weight + self.bias


class ResNetBlock(eqx.Module):
    config: ScOTConfig = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    conv1: Conv2dNHWC
    bn1: _BatchNorm
    conv2: Conv2dNHWC
    bn2: _BatchNorm

    def __init__(self, config: ScOTConfig, dim: int, *, key: jax.Array):
        self.config = config
        self.dim = dim
        key1, key2 = jax.random.split(key)
        self.conv1 = Conv2dNHWC(dim, dim, kernel_size=(3, 3), padding="SAME", key=key1)
        self.bn1 = _BatchNorm(dim)
        self.conv2 = Conv2dNHWC(dim, dim, kernel_size=(3, 3), padding="SAME", key=key2)
        self.bn2 = _BatchNorm(dim)

    def __call__(
        self,
        x: jnp.ndarray,
        time: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        batch_size, sequence_length, hidden_size = x.shape
        input_dim = int(math.floor(sequence_length**0.5))
        input_x = x

        x = x.reshape(batch_size, input_dim, input_dim, hidden_size)
        x = self.conv1(x)
        x = self.bn1(x)
        x = jax.nn.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x.reshape(batch_size, sequence_length, hidden_size)
        return x + input_x


# ---------------------------------------------------------------------------
# Patch Embeddings
# ---------------------------------------------------------------------------


class ScOTPatchEmbeddings(eqx.Module):
    config: ScOTConfig = eqx.field(static=True)
    image_size: Tuple[int, int] = eqx.field(static=True)
    patch_size: Tuple[int, int] = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)
    grid_size: Tuple[int, int] = eqx.field(static=True)
    projection: Conv2dNHWC

    def __init__(self, config: ScOTConfig, *, key: jax.Array):
        self.config = config
        image_size = config.image_size
        patch_size = config.patch_size
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

        self.projection = Conv2dNHWC(
            in_channels=config.num_channels,
            out_channels=config.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            key=key,
        )

    def maybe_pad(
        self, pixel_values: jnp.ndarray, height: int, width: int
    ) -> jnp.ndarray:
        pad_w = (self.patch_size[1] - width % self.patch_size[1]) % self.patch_size[1]
        pad_h = (self.patch_size[0] - height % self.patch_size[0]) % self.patch_size[0]
        if pad_w > 0 or pad_h > 0:
            pixel_values = jnp.pad(
                pixel_values, ((0, 0), (0, pad_h), (0, pad_w), (0, 0))
            )
        return pixel_values

    def __call__(
        self, pixel_values: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Tuple[int, int]]:
        batch_size, height, width, num_channels = pixel_values.shape
        pixel_values = self.maybe_pad(pixel_values, height, width)
        embeddings = self.projection(pixel_values)
        _, new_height, new_width, _ = embeddings.shape
        output_dimensions = (new_height, new_width)
        embeddings = embeddings.reshape(batch_size, -1, self.config.embed_dim)
        return embeddings, output_dimensions


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class ScOTEmbeddings(eqx.Module):
    config: ScOTConfig = eqx.field(static=True)
    use_mask_token: bool = eqx.field(static=True)
    patch_embeddings: ScOTPatchEmbeddings
    mask_token: Optional[jnp.ndarray]
    position_embeddings: Optional[jnp.ndarray]
    norm: Union[ConditionalLayerNorm, LayerNormWithTime]
    dropout: eqx.nn.Dropout

    def __init__(
        self, config: ScOTConfig, use_mask_token: bool = False, *, key: jax.Array
    ):
        self.config = config
        self.use_mask_token = use_mask_token

        key1, key2 = jax.random.split(key)
        self.patch_embeddings = ScOTPatchEmbeddings(config, key=key1)
        num_patches = self.patch_embeddings.num_patches

        if use_mask_token:
            self.mask_token = jnp.zeros((1, 1, config.embed_dim))
        else:
            self.mask_token = None

        if config.use_absolute_embeddings:
            self.position_embeddings = jnp.zeros((1, num_patches, config.embed_dim))
        else:
            self.position_embeddings = None

        self.norm = make_layer_norm(config, config.embed_dim, key=key2)
        self.dropout = eqx.nn.Dropout(p=config.hidden_dropout_prob)

    def __call__(
        self,
        pixel_values: jnp.ndarray,
        bool_masked_pos: Optional[jnp.ndarray] = None,
        time: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        *,
        key: Optional[jax.Array] = None,
    ) -> Tuple[jnp.ndarray, Tuple[int, int]]:
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings, time)
        batch_size, seq_len, _ = embeddings.shape

        if bool_masked_pos is not None and self.mask_token is not None:
            mask_tokens = jnp.broadcast_to(
                self.mask_token, (batch_size, seq_len, self.config.embed_dim)
            )
            mask = bool_masked_pos[:, :, None].astype(embeddings.dtype)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings

        if deterministic:
            pass  # skip dropout
        elif key is not None:
            embeddings = self.dropout(embeddings, key=key)

        return embeddings, output_dimensions


# ---------------------------------------------------------------------------
# Relative Position Bias
# ---------------------------------------------------------------------------


class Swinv2RelativePositionBias(eqx.Module):
    """Relative position bias for Swin Transformer V2."""

    config: ScOTConfig = eqx.field(static=True)
    window_size: Tuple[int, int] = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    pretrained_window_size: Tuple[int, int] = eqx.field(static=True)

    cpb_mlp_0: eqx.nn.Linear
    cpb_mlp_1: eqx.nn.Linear
    relative_coords_table: jnp.ndarray
    relative_position_index: jnp.ndarray

    def __init__(
        self,
        config: ScOTConfig,
        window_size: Tuple[int, int],
        num_heads: int,
        pretrained_window_size: Tuple[int, int] = (0, 0),
        *,
        key: jax.Array,
    ):
        self.config = config
        self.window_size = window_size
        self.num_heads = num_heads
        self.pretrained_window_size = pretrained_window_size

        key1, key2 = jax.random.split(key)
        self.cpb_mlp_0 = eqx.nn.Linear(2, 512, use_bias=True, key=key1)
        self.cpb_mlp_1 = eqx.nn.Linear(512, num_heads, use_bias=False, key=key2)

        # Build relative coords table (static)
        relative_coords_h = np.arange(
            -(window_size[0] - 1), window_size[0], dtype=np.float32
        )
        relative_coords_w = np.arange(
            -(window_size[1] - 1), window_size[1], dtype=np.float32
        )
        relative_coords_table = np.stack(
            np.meshgrid(relative_coords_h, relative_coords_w, indexing="ij")
        )
        relative_coords_table = relative_coords_table.transpose(1, 2, 0).reshape(-1, 2)

        if pretrained_window_size[0] > 0:
            relative_coords_table = relative_coords_table / np.array(
                [pretrained_window_size[0] - 1, pretrained_window_size[1] - 1],
                dtype=np.float32,
            )
        else:
            denom_h = max(window_size[0] - 1, 1)
            denom_w = max(window_size[1] - 1, 1)
            relative_coords_table = relative_coords_table / np.array(
                [denom_h, denom_w], dtype=np.float32
            )

        relative_coords_table = relative_coords_table * 8
        relative_coords_table = (
            np.sign(relative_coords_table)
            * np.log2(np.abs(relative_coords_table) + 1.0)
            / np.log2(8.0)
        )
        self.relative_coords_table = jnp.asarray(relative_coords_table)

        # Relative position index
        coords_h = np.arange(window_size[0])
        coords_w = np.arange(window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose(1, 2, 0)
        relative_coords = relative_coords + np.array(
            [window_size[0] - 1, window_size[1] - 1]
        )
        relative_coords = relative_coords * np.array([2 * window_size[1] - 1, 1])
        self.relative_position_index = jnp.asarray(
            relative_coords.sum(-1).astype(np.int32)
        )

    def __call__(self) -> jnp.ndarray:
        # Apply MLP
        relative_position_bias = jax.vmap(self.cpb_mlp_0)(self.relative_coords_table)
        relative_position_bias = jax.nn.relu(relative_position_bias)
        relative_position_bias = jax.vmap(self.cpb_mlp_1)(relative_position_bias)

        # Index into the bias table
        relative_position_bias = relative_position_bias.reshape(-1, self.num_heads)
        relative_position_bias = relative_position_bias[
            self.relative_position_index.reshape(-1)
        ]
        relative_position_bias = relative_position_bias.reshape(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.transpose(2, 0, 1)
        relative_position_bias = 16 * jax.nn.sigmoid(relative_position_bias)
        return relative_position_bias


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class Swinv2Attention(eqx.Module):
    config: ScOTConfig = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    window_size: Tuple[int, int] = eqx.field(static=True)

    query: eqx.nn.Linear
    key_proj: eqx.nn.Linear
    value: eqx.nn.Linear
    proj: eqx.nn.Linear
    attn_drop: eqx.nn.Dropout
    proj_drop: eqx.nn.Dropout
    relative_position_bias: Swinv2RelativePositionBias
    logit_scale: jnp.ndarray

    def __init__(
        self,
        config: ScOTConfig,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int],
        pretrained_window_size: Tuple[int, int] = (0, 0),
        *,
        key: jax.Array,
    ):
        self.config = config
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        keys = jax.random.split(key, 6)

        self.query = eqx.nn.Linear(dim, dim, use_bias=config.qkv_bias, key=keys[0])
        self.key_proj = eqx.nn.Linear(dim, dim, use_bias=False, key=keys[1])
        self.value = eqx.nn.Linear(dim, dim, use_bias=config.qkv_bias, key=keys[2])
        self.proj = eqx.nn.Linear(dim, dim, key=keys[3])

        self.attn_drop = eqx.nn.Dropout(p=config.attention_probs_dropout_prob)
        self.proj_drop = eqx.nn.Dropout(p=config.hidden_dropout_prob)

        self.relative_position_bias = Swinv2RelativePositionBias(
            config=config,
            window_size=window_size,
            num_heads=num_heads,
            pretrained_window_size=pretrained_window_size,
            key=keys[4],
        )

        self.logit_scale = jnp.log(10 * jnp.ones((num_heads, 1, 1)))

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        head_mask: Optional[jnp.ndarray] = None,
        output_attentions: bool = False,
        deterministic: bool = True,
        *,
        key: Optional[jax.Array] = None,
    ) -> Tuple[jnp.ndarray, ...]:
        batch_size, seq_len, _ = hidden_states.shape

        # Apply linear projections: vmap over (batch, seq)
        def proj_fn(linear, x):
            return jax.vmap(jax.vmap(linear))(x)

        query = proj_fn(self.query, hidden_states)
        k = proj_fn(self.key_proj, hidden_states)
        v = proj_fn(self.value, hidden_states)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        query = query.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        # Cosine attention
        query = query / jnp.maximum(
            jnp.linalg.norm(query, axis=-1, keepdims=True), 1e-6
        )
        k = k / jnp.maximum(jnp.linalg.norm(k, axis=-1, keepdims=True), 1e-6)

        logit_scale = jnp.exp(jnp.clip(self.logit_scale))
        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", query, k) * logit_scale

        # Relative position bias
        relative_position_bias = self.relative_position_bias()
        attn_weights = attn_weights + relative_position_bias[None, :, :, :]

        if attention_mask is not None:
            num_windows = attention_mask.shape[0]
            attn_weights = attn_weights.reshape(
                batch_size // num_windows, num_windows, self.num_heads, seq_len, seq_len
            )
            attn_weights = attn_weights + attention_mask[None, :, None, :, :]
            attn_weights = attn_weights.reshape(-1, self.num_heads, seq_len, seq_len)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # Dropout (only when not deterministic and key is provided)
        if not deterministic and key is not None:
            key1, key2 = jax.random.split(key)
            attn_weights = self.attn_drop(attn_weights, key=key1)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.dim
        )

        attn_output = jax.vmap(jax.vmap(self.proj))(attn_output)

        if not deterministic and key is not None:
            attn_output = self.proj_drop(attn_output, key=key2)

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


# ---------------------------------------------------------------------------
# MLP layers
# ---------------------------------------------------------------------------


class Swinv2Intermediate(eqx.Module):
    dense: eqx.nn.Linear

    def __init__(self, config: ScOTConfig, dim: int, *, key: jax.Array):
        self.dense = eqx.nn.Linear(dim, int(dim * config.mlp_ratio), key=key)

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        hidden_states = jax.vmap(jax.vmap(self.dense))(hidden_states)
        return jax.nn.gelu(hidden_states)


class Swinv2Output(eqx.Module):
    config: ScOTConfig = eqx.field(static=True)
    dense: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, config: ScOTConfig, dim: int, *, key: jax.Array):
        self.config = config
        self.dense = eqx.nn.Linear(int(dim * config.mlp_ratio), dim, key=key)
        self.dropout = eqx.nn.Dropout(p=config.hidden_dropout_prob)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        deterministic: bool = True,
        *,
        key: Optional[jax.Array] = None,
    ) -> jnp.ndarray:
        hidden_states = jax.vmap(jax.vmap(self.dense))(hidden_states)
        if not deterministic and key is not None:
            hidden_states = self.dropout(hidden_states, key=key)
        return hidden_states


# ---------------------------------------------------------------------------
# ScOT Layer (single transformer layer)
# ---------------------------------------------------------------------------


class ScOTLayer(eqx.Module):
    config: ScOTConfig = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    input_resolution: Tuple[int, int] = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    shift_size: Any = eqx.field(static=True)
    drop_path_rate: float = eqx.field(static=True)
    pretrained_window_size: Any = eqx.field(static=True)

    attention: Swinv2Attention
    layernorm_before: Union[ConditionalLayerNorm, LayerNormWithTime]
    layernorm_after: Union[ConditionalLayerNorm, LayerNormWithTime]
    intermediate: Swinv2Intermediate
    output_layer: Swinv2Output
    drop_path_layer: DropPath

    def __init__(
        self,
        config: ScOTConfig,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        shift_size: Any = 0,
        drop_path: float = 0.0,
        pretrained_window_size: Any = 0,
        *,
        key: jax.Array,
    ):
        self.config = config
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.drop_path_rate = drop_path
        self.pretrained_window_size = pretrained_window_size

        keys = jax.random.split(key, 5)

        # Determine effective window size for attention creation
        config_window_size = config.window_size
        if isinstance(config_window_size, int):
            config_window_size = (config_window_size, config_window_size)

        pretrained_ws = pretrained_window_size
        if isinstance(pretrained_ws, int):
            pretrained_ws = (pretrained_ws, pretrained_ws)

        height, width = input_resolution
        window_size = min(config_window_size[0], height, width)

        self.attention = Swinv2Attention(
            config=config,
            dim=dim,
            num_heads=num_heads,
            window_size=(window_size, window_size),
            pretrained_window_size=pretrained_ws,
            key=keys[0],
        )

        self.layernorm_before = make_layer_norm(config, dim, key=keys[1])
        self.layernorm_after = make_layer_norm(config, dim, key=keys[2])
        self.intermediate = Swinv2Intermediate(config, dim, key=keys[3])
        self.output_layer = Swinv2Output(config, dim, key=keys[4])
        self.drop_path_layer = DropPath(drop_prob=drop_path)

    def get_attn_mask(self, height, width, shift_size, window_size):
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

    def maybe_pad(self, hidden_states, height, width, window_size):
        pad_right = (window_size - width % window_size) % window_size
        pad_bottom = (window_size - height % window_size) % window_size
        pad_values = ((0, 0), (0, pad_bottom), (0, pad_right), (0, 0))
        if pad_right > 0 or pad_bottom > 0:
            hidden_states = jnp.pad(hidden_states, pad_values)
        return hidden_states, (0, 0, 0, pad_right, 0, pad_bottom)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        input_dimensions: Tuple[int, int],
        time: Optional[jnp.ndarray] = None,
        head_mask: Optional[jnp.ndarray] = None,
        output_attentions: bool = False,
        always_partition: bool = False,
        deterministic: bool = True,
        *,
        key: Optional[jax.Array] = None,
    ) -> Tuple[jnp.ndarray, ...]:
        height, width = input_dimensions
        batch_size, seq_len, channels = hidden_states.shape
        shortcut = hidden_states

        config_window_size = self.config.window_size
        if isinstance(config_window_size, int):
            config_window_size = (config_window_size, config_window_size)

        shift_size_tuple = (
            self.shift_size
            if isinstance(self.shift_size, (list, tuple))
            else (self.shift_size, self.shift_size)
        )

        window_size = min(config_window_size[0], height, width)
        shift_size = 0 if min(height, width) <= window_size else shift_size_tuple[0]

        hidden_states = hidden_states.reshape(batch_size, height, width, channels)
        hidden_states, pad_values = self.maybe_pad(
            hidden_states, height, width, window_size
        )
        _, height_pad, width_pad, _ = hidden_states.shape

        if shift_size > 0:
            shifted_hidden_states = jnp.roll(
                hidden_states, shift=(-shift_size, -shift_size), axis=(1, 2)
            )
        else:
            shifted_hidden_states = hidden_states

        hidden_states_windows = window_partition(shifted_hidden_states, window_size)
        hidden_states_windows = hidden_states_windows.reshape(
            -1, window_size * window_size, channels
        )

        attn_mask = self.get_attn_mask(height_pad, width_pad, shift_size, window_size)

        attn_key = None
        if key is not None:
            key, attn_key = jax.random.split(key)

        attention_outputs = self.attention(
            hidden_states_windows,
            attention_mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            deterministic=deterministic,
            key=attn_key,
        )
        attention_output = attention_outputs[0]

        attention_windows = attention_output.reshape(
            -1, window_size, window_size, channels
        )
        shifted_windows = window_reverse(
            attention_windows, window_size, height_pad, width_pad
        )

        if shift_size > 0:
            attention_windows = jnp.roll(
                shifted_windows, shift=(shift_size, shift_size), axis=(1, 2)
            )
        else:
            attention_windows = shifted_windows

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :]

        attention_windows = attention_windows.reshape(
            batch_size, height * width, channels
        )

        # First residual
        hidden_states = shortcut + self.drop_path_layer(
            self.layernorm_before(attention_windows, time), deterministic=deterministic
        )

        # MLP
        residual = hidden_states
        mlp_key = None
        if key is not None:
            key, mlp_key = jax.random.split(key)
        layer_output = self.output_layer(
            self.intermediate(hidden_states), deterministic=deterministic, key=mlp_key
        )
        layer_output = residual + self.drop_path_layer(
            self.layernorm_after(layer_output, time), deterministic=deterministic
        )

        outputs = (
            (layer_output, attention_outputs[1])
            if output_attentions
            else (layer_output,)
        )
        return outputs


# ---------------------------------------------------------------------------
# Patch Merging / Unmerging
# ---------------------------------------------------------------------------


class ScOTPatchMerging(eqx.Module):
    config: ScOTConfig = eqx.field(static=True)
    input_resolution: Tuple[int, int] = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    reduction: eqx.nn.Linear
    norm: Union[ConditionalLayerNorm, LayerNormWithTime]

    def __init__(
        self,
        config: ScOTConfig,
        input_resolution: Tuple[int, int],
        dim: int,
        *,
        key: jax.Array,
    ):
        self.config = config
        self.input_resolution = input_resolution
        self.dim = dim
        key1, key2 = jax.random.split(key)
        self.reduction = eqx.nn.Linear(4 * dim, 2 * dim, use_bias=False, key=key1)
        self.norm = make_layer_norm(config, 2 * dim, key=key2)

    def maybe_pad(self, input_feature, height, width):
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            input_feature = jnp.pad(
                input_feature, ((0, 0), (0, height % 2), (0, width % 2), (0, 0))
            )
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

        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]

        input_feature = jnp.concatenate(
            [input_feature_0, input_feature_1, input_feature_2, input_feature_3],
            axis=-1,
        )
        input_feature = input_feature.reshape(batch_size, -1, 4 * num_channels)

        input_feature = jax.vmap(jax.vmap(self.reduction))(input_feature)
        input_feature = self.norm(input_feature, time)
        return input_feature


class ScOTPatchUnmerging(eqx.Module):
    config: ScOTConfig = eqx.field(static=True)
    input_resolution: Tuple[int, int] = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    upsample: eqx.nn.Linear
    mixup: eqx.nn.Linear
    norm: Union[ConditionalLayerNorm, LayerNormWithTime]

    def __init__(
        self,
        config: ScOTConfig,
        input_resolution: Tuple[int, int],
        dim: int,
        *,
        key: jax.Array,
    ):
        self.config = config
        self.input_resolution = input_resolution
        self.dim = dim
        keys = jax.random.split(key, 3)
        self.upsample = eqx.nn.Linear(dim, 2 * dim, use_bias=False, key=keys[0])
        self.mixup = eqx.nn.Linear(dim // 2, dim // 2, use_bias=False, key=keys[1])
        self.norm = make_layer_norm(config, dim // 2, key=keys[2])

    def maybe_crop(self, input_feature, height, width):
        h_in, w_in = input_feature.shape[1], input_feature.shape[2]
        if h_in > height:
            input_feature = input_feature[:, :height, :, :]
        if w_in > width:
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

        input_feature = jax.vmap(jax.vmap(self.upsample))(input_feature)
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
        return jax.vmap(jax.vmap(self.mixup))(input_feature)


# ---------------------------------------------------------------------------
# Patch Recovery
# ---------------------------------------------------------------------------


class ScOTPatchRecovery(eqx.Module):
    config: ScOTConfig = eqx.field(static=True)
    image_size: Tuple[int, int] = eqx.field(static=True)
    patch_size: Tuple[int, int] = eqx.field(static=True)
    grid_size: Tuple[int, int] = eqx.field(static=True)
    projection: ConvTranspose2dNHWC
    mixup: Conv2dNHWC

    def __init__(self, config: ScOTConfig, *, key: jax.Array):
        self.config = config
        image_size = config.image_size
        patch_size = config.patch_size
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

        key1, key2 = jax.random.split(key)
        self.projection = ConvTranspose2dNHWC(
            in_channels=config.embed_dim,
            out_channels=config.num_out_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            key=key1,
        )
        self.mixup = Conv2dNHWC(
            in_channels=config.num_out_channels,
            out_channels=config.num_out_channels,
            kernel_size=(5, 5),
            padding="SAME",
            use_bias=False,
            key=key2,
        )

    def maybe_crop(self, pixel_values, height, width):
        if pixel_values.shape[1] > height:
            pixel_values = pixel_values[:, :height, :, :]
        if pixel_values.shape[2] > width:
            pixel_values = pixel_values[:, :, :width, :]
        return pixel_values

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, *self.grid_size, hidden_size)
        output = self.projection(hidden_states)
        output = self.maybe_crop(output, self.image_size[0], self.image_size[1])
        return self.mixup(output)


# ---------------------------------------------------------------------------
# Encode / Decode Stages
# ---------------------------------------------------------------------------


class ScOTEncodeStage(eqx.Module):
    config: ScOTConfig = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    input_resolution: Tuple[int, int] = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    do_downsample: bool = eqx.field(static=True)

    blocks: List[ScOTLayer]
    downsample_layer: Optional[ScOTPatchMerging]

    def __init__(
        self,
        config: ScOTConfig,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        drop_path: Sequence[float],
        downsample: bool,
        pretrained_window_size: int = 0,
        *,
        key: jax.Array,
    ):
        self.config = config
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.do_downsample = downsample

        window_size = config.window_size
        if isinstance(window_size, int):
            window_size = (window_size, window_size)

        keys = jax.random.split(key, depth + 1)

        self.blocks = [
            ScOTLayer(
                config=config,
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                shift_size=(
                    [0, 0]
                    if (i % 2 == 0)
                    else [window_size[0] // 2, window_size[1] // 2]
                ),
                drop_path=drop_path[i],
                pretrained_window_size=pretrained_window_size,
                key=keys[i],
            )
            for i in range(depth)
        ]

        if downsample:
            self.downsample_layer = ScOTPatchMerging(
                config=config,
                input_resolution=input_resolution,
                dim=dim,
                key=keys[depth],
            )
        else:
            self.downsample_layer = None

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        input_dimensions: Tuple[int, int],
        time: Optional[jnp.ndarray] = None,
        head_mask: Optional[jnp.ndarray] = None,
        output_attentions: bool = False,
        always_partition: bool = False,
        deterministic: bool = True,
        *,
        key: Optional[jax.Array] = None,
    ) -> Tuple[jnp.ndarray, ...]:
        height, width = input_dimensions
        inputs = hidden_states

        for i, block in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            block_key = None
            if key is not None:
                key, block_key = jax.random.split(key)
            layer_outputs = block(
                hidden_states,
                input_dimensions,
                time,
                layer_head_mask,
                output_attentions,
                always_partition,
                deterministic,
                key=block_key,
            )
            hidden_states = layer_outputs[0]

        hidden_states_before_downsampling = hidden_states

        if self.do_downsample and self.downsample_layer is not None:
            height_downsampled = (height + 1) // 2
            width_downsampled = (width + 1) // 2
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


class ScOTDecodeStage(eqx.Module):
    config: ScOTConfig = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    input_resolution: Tuple[int, int] = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    do_upsample: bool = eqx.field(static=True)
    upsampled_size: Tuple[int, int] = eqx.field(static=True)

    blocks: List[ScOTLayer]
    upsample_layer: Optional[ScOTPatchUnmerging]

    def __init__(
        self,
        config: ScOTConfig,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        drop_path: Sequence[float],
        upsample: bool,
        upsampled_size: Tuple[int, int],
        pretrained_window_size: int = 0,
        *,
        key: jax.Array,
    ):
        self.config = config
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.do_upsample = upsample
        self.upsampled_size = upsampled_size

        window_size = config.window_size
        if isinstance(window_size, int):
            window_size = (window_size, window_size)

        keys = jax.random.split(key, depth + 1)

        self.blocks = [
            ScOTLayer(
                config=config,
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                shift_size=(
                    [0, 0]
                    if (i % 2 == 0)
                    else [window_size[0] // 2, window_size[1] // 2]
                ),
                drop_path=drop_path[depth - 1 - i],
                pretrained_window_size=pretrained_window_size,
                key=keys[i],
            )
            for i in reversed(range(depth))
        ]

        if upsample:
            self.upsample_layer = ScOTPatchUnmerging(
                config=config,
                input_resolution=input_resolution,
                dim=dim,
                key=keys[depth],
            )
        else:
            self.upsample_layer = None

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        input_dimensions: Tuple[int, int],
        time: Optional[jnp.ndarray] = None,
        head_mask: Optional[jnp.ndarray] = None,
        output_attentions: bool = False,
        always_partition: bool = False,
        deterministic: bool = True,
        *,
        key: Optional[jax.Array] = None,
    ) -> Tuple[jnp.ndarray, ...]:
        height, width = input_dimensions

        for i, block in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            block_key = None
            if key is not None:
                key, block_key = jax.random.split(key)
            layer_outputs = block(
                hidden_states,
                input_dimensions,
                time,
                layer_head_mask,
                output_attentions,
                always_partition,
                deterministic,
                key=block_key,
            )
            hidden_states = layer_outputs[0]

        hidden_states_before_upsampling = hidden_states

        if self.do_upsample and self.upsample_layer is not None:
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


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------


class ScOTEncoder(eqx.Module):
    config: ScOTConfig = eqx.field(static=True)
    grid_size: Tuple[int, int] = eqx.field(static=True)
    layers: List[ScOTEncodeStage]

    def __init__(
        self, config: ScOTConfig, grid_size: Tuple[int, int], *, key: jax.Array
    ):
        self.config = config
        self.grid_size = grid_size

        num_layers = len(config.depths)
        pretrained_window_sizes = config.pretrained_window_sizes

        total_depth = 2 * sum(config.depths)
        drop_rates = jnp.linspace(0, config.drop_path_rate, total_depth)
        dpr = [float(x) for x in drop_rates[: total_depth // 2]]

        keys = jax.random.split(key, num_layers)

        self.layers = [
            ScOTEncodeStage(
                config=config,
                dim=int(config.embed_dim * 2**i_layer),
                input_resolution=(
                    grid_size[0] // (2**i_layer),
                    grid_size[1] // (2**i_layer),
                ),
                depth=config.depths[i_layer],
                num_heads=config.num_heads[i_layer],
                drop_path=dpr[
                    sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])
                ],
                downsample=(i_layer < num_layers - 1),
                pretrained_window_size=pretrained_window_sizes[i_layer],
                key=keys[i_layer],
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
        *,
        key: Optional[jax.Array] = None,
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
            layer_key = None
            if key is not None:
                key, layer_key = jax.random.split(key)

            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                time,
                layer_head_mask,
                output_attentions,
                always_partition,
                deterministic,
                key=layer_key,
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


class ScOTDecoder(eqx.Module):
    config: ScOTConfig = eqx.field(static=True)
    grid_size: Tuple[int, int] = eqx.field(static=True)
    layers: List[ScOTDecodeStage]

    def __init__(
        self, config: ScOTConfig, grid_size: Tuple[int, int], *, key: jax.Array
    ):
        self.config = config
        self.grid_size = grid_size

        num_layers = len(config.depths)
        pretrained_window_sizes = config.pretrained_window_sizes

        total_depth = 2 * sum(config.depths)
        drop_rates = jnp.linspace(0, config.drop_path_rate, total_depth)
        dpr = [float(x) for x in drop_rates[total_depth // 2 :]]

        keys = jax.random.split(key, num_layers)

        self.layers = [
            ScOTDecodeStage(
                config=config,
                dim=int(config.embed_dim * 2**i_layer),
                input_resolution=(
                    grid_size[0] // (2**i_layer),
                    grid_size[1] // (2**i_layer),
                ),
                depth=config.depths[i_layer],
                num_heads=config.num_heads[i_layer],
                drop_path=dpr[
                    sum(config.depths[i_layer + 1 :]) : sum(config.depths[i_layer:])
                ],
                upsample=(i_layer > 0),
                upsampled_size=(
                    grid_size[0] // (2 ** (i_layer - 1)),
                    grid_size[1] // (2 ** (i_layer - 1)),
                ),
                pretrained_window_size=pretrained_window_sizes[i_layer],
                key=keys[i_layer],
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
        *,
        key: Optional[jax.Array] = None,
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

            if i != 0 and len(skip_states) > len(skip_states) - i:
                skip_idx = len(skip_states) - i
                if (
                    0 <= skip_idx < len(skip_states)
                    and skip_states[skip_idx] is not None
                ):
                    hidden_states = hidden_states + skip_states[skip_idx]

            layer_key = None
            if key is not None:
                key, layer_key = jax.random.split(key)

            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                time,
                layer_head_mask,
                output_attentions,
                always_partition,
                deterministic,
                key=layer_key,
            )

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


# ---------------------------------------------------------------------------
# Residual Block Wrapper
# ---------------------------------------------------------------------------


class ResidualBlockWrapper(eqx.Module):
    config: ScOTConfig = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    blocks: Optional[List[Union[ConvNeXtBlock, ResNetBlock]]]

    def __init__(
        self,
        config: ScOTConfig,
        dim: int,
        depth: int,
        drop_path: float = 0.0,
        *,
        key: jax.Array,
    ):
        self.config = config
        self.dim = dim
        self.depth = depth

        if depth > 0:
            keys = jax.random.split(key, depth)
            if config.residual_model == "convnext":
                self.blocks = [
                    ConvNeXtBlock(
                        config=config, dim=dim, drop_path=drop_path, key=keys[i]
                    )
                    for i in range(depth)
                ]
            elif config.residual_model == "resnet":
                self.blocks = [
                    ResNetBlock(config=config, dim=dim, key=keys[i])
                    for i in range(depth)
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


# ---------------------------------------------------------------------------
# Main ScOT Model
# ---------------------------------------------------------------------------


class ScOT(eqx.Module):
    """Main ScOT model - Swin Transformer V2 based encoder-decoder (Equinox)."""

    config: ScOTConfig = eqx.field(static=True)
    use_mask_token: bool = eqx.field(static=True)
    use_conditioning: bool = eqx.field(static=True)

    num_layers_encoder: int = eqx.field(static=True)
    num_layers_decoder: int = eqx.field(static=True)
    num_features: int = eqx.field(static=True)
    patch_grid: Tuple[int, int] = eqx.field(static=True)

    embeddings: ScOTEmbeddings
    encoder: ScOTEncoder
    decoder: ScOTDecoder
    patch_recovery: ScOTPatchRecovery
    residual_blocks: List[ResidualBlockWrapper]

    def __init__(
        self,
        config: ScOTConfig,
        use_mask_token: bool = False,
        use_conditioning: bool = False,
        *,
        key: jax.Array,
    ):
        self.config = config
        self.use_mask_token = use_mask_token
        self.use_conditioning = use_conditioning

        self.num_layers_encoder = len(config.depths)
        self.num_layers_decoder = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers_encoder - 1))

        image_size = config.image_size
        patch_size = config.patch_size
        image_size_t = (
            (image_size, image_size) if isinstance(image_size, int) else image_size
        )
        patch_size_t = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )
        self.patch_grid = (
            image_size_t[0] // patch_size_t[0],
            image_size_t[1] // patch_size_t[1],
        )

        keys = jax.random.split(key, 4 + len(config.skip_connections))

        self.embeddings = ScOTEmbeddings(
            config, use_mask_token=use_mask_token, key=keys[0]
        )
        self.encoder = ScOTEncoder(config, self.patch_grid, key=keys[1])
        self.decoder = ScOTDecoder(config, self.patch_grid, key=keys[2])
        self.patch_recovery = ScOTPatchRecovery(config, key=keys[3])

        skip_connections = config.skip_connections
        self.residual_blocks = [
            ResidualBlockWrapper(
                config=config,
                dim=config.embed_dim * 2**i,
                depth=int(depth) if depth else 0,
                key=keys[4 + i],
            )
            for i, depth in enumerate(skip_connections)
        ]

    def _downsample(self, image: jnp.ndarray, target_size: int) -> jnp.ndarray:
        image_size = image.shape[1]
        freqs = jnp.fft.fftfreq(image_size, d=1 / image_size)
        sel = jnp.logical_and(freqs >= -target_size / 2, freqs <= target_size / 2 - 1)
        image_hat = jnp.fft.fft2(image, axes=(1, 2), norm="forward")
        image_hat = image_hat[:, sel, :, :][:, :, sel, :]
        image = jnp.fft.ifft2(image_hat, axes=(1, 2), norm="forward").real
        return image

    def _upsample(self, image: jnp.ndarray, target_size: int) -> jnp.ndarray:
        image_size = image.shape[1]
        image_hat = jnp.fft.fft2(image, axes=(1, 2), norm="forward")
        image_hat = jnp.fft.fftshift(image_hat, axes=(1, 2))
        pad_size = (target_size - image_size) // 2
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
        *,
        key: Optional[jax.Array] = None,
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

        image_size = pixel_values.shape[1]
        original_pixel_values = pixel_values

        if image_size != self.config.image_size:
            if image_size < self.config.image_size:
                pixel_values = self._upsample(pixel_values, self.config.image_size)
            else:
                pixel_values = self._downsample(pixel_values, self.config.image_size)

        emb_key = None
        if key is not None:
            key, emb_key = jax.random.split(key)

        embedding_output, input_dimensions = self.embeddings(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            time=time,
            deterministic=deterministic,
            key=emb_key,
        )

        enc_key = None
        if key is not None:
            key, enc_key = jax.random.split(key)

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
            key=enc_key,
        )

        skip_states = (
            list(encoder_outputs.hidden_states[1:])
            if encoder_outputs.hidden_states
            else []
        )

        for i in range(len(skip_states)):
            if i < len(self.residual_blocks):
                skip_states[i] = self.residual_blocks[i](
                    skip_states[i], time, deterministic=deterministic
                )

        input_dim = (
            int(math.floor(skip_states[-1].shape[1] ** 0.5))
            if skip_states
            else input_dimensions[0]
        )

        dec_key = None
        if key is not None:
            key, dec_key = jax.random.split(key)

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
            key=dec_key,
        )

        sequence_output = decoder_output.last_hidden_state
        prediction = self.patch_recovery(sequence_output)

        if self.config.learn_residual:
            if self.config.num_channels > self.config.num_out_channels:
                residual_input = original_pixel_values[
                    :, :, :, : self.config.num_out_channels
                ]
            else:
                residual_input = original_pixel_values
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

        if image_size != self.config.image_size:
            if image_size > self.config.image_size:
                prediction = self._upsample(prediction, image_size)
            else:
                prediction = self._downsample(prediction, image_size)

        if pixel_mask is not None and labels is not None:
            prediction = jnp.where(pixel_mask, labels, prediction)

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


# ---------------------------------------------------------------------------
# Weight transfer: Flax -> Equinox
# ---------------------------------------------------------------------------


def transfer_flax_to_eqx(flax_model, flax_params, eqx_model):
    """Transfer weights from a Flax ScOT model to an Equinox ScOT model.

    Both models must have the same config. Returns a new Equinox model
    with weights copied from the Flax params tree.
    """
    # This is a structured approach: we walk the Flax param tree and
    # map each leaf to the corresponding Equinox module field.
    # For now, we provide a simpler approach: flatten both and transfer by shape.

    flax_flat = _flatten_flax_params(flax_params)
    return _set_eqx_params(eqx_model, flax_flat)


def _flatten_flax_params(params, prefix=""):
    """Flatten Flax params dict to {path: array}."""
    result = {}
    if isinstance(params, dict):
        for k, v in params.items():
            new_prefix = f"{prefix}/{k}" if prefix else k
            result.update(_flatten_flax_params(v, new_prefix))
    else:
        result[prefix] = params
    return result


def _set_eqx_params(model, flax_flat):
    """Set Equinox model params from flattened Flax params.

    This requires a mapping between Flax param paths and Equinox module paths.
    Implementation depends on the specific model structure.
    """
    # TODO: Implement detailed path mapping
    # For now, this is a placeholder that will be filled in during testing
    raise NotImplementedError(
        "Automatic weight transfer not yet implemented. "
        "Use the test script's manual approach instead."
    )
