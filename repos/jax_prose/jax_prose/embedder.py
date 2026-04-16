import math
from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp

from .config import EmbedderConfig


def patchify(data: jnp.ndarray, patch_num: int) -> jnp.ndarray:
    """
    Input:  (bs, nt, px, py, d)
    Output: (bs, nt, p*p, x*y*d)
    """
    bs, nt, px, py, d = data.shape
    p = patch_num
    x = px // p
    y = py // p
    data = data.reshape(bs, nt, p, x, p, y, d)
    data = jnp.transpose(data, (0, 1, 2, 4, 3, 5, 6))
    data = data.reshape(bs, nt, p * p, x * y * d)
    return data


def depatchify(
    data: jnp.ndarray, patch_num: int, x: int, y: int, d: int
) -> jnp.ndarray:
    """
    Input:  (bs, nt, p*p, x*y*d)
    Output: (bs, nt, px, py, d)
    """
    bs, nt, _, _ = data.shape
    p = patch_num
    data = data.reshape(bs, nt, p, p, x, y, d)
    data = jnp.transpose(data, (0, 1, 2, 4, 3, 5, 6))
    data = data.reshape(bs, nt, p * x, p * y, d)
    return data


class LinearEmbedder(nn.Module):
    config: EmbedderConfig
    x_num: int
    data_dim: int

    def setup(self):
        assert self.x_num % self.config.patch_num == 0
        assert self.x_num % self.config.patch_num_output == 0

        self.dim = self.config.dim
        self.patch_resolution = self.x_num // self.config.patch_num
        self.patch_dim = self.data_dim * self.patch_resolution * self.patch_resolution

        self.patch_resolution_output = self.x_num // self.config.patch_num_output
        self.patch_dim_output = (
            self.data_dim * self.patch_resolution_output * self.patch_resolution_output
        )

        self.patch_position_embeddings = self.param(
            "patch_position_embeddings",
            nn.initializers.normal(),
            (1, 1, self.config.patch_num * self.config.patch_num, self.dim),
        )

        self.time_proj = nn.Sequential(
            [nn.Dense(self.dim), nn.gelu, nn.Dense(self.dim)]
        )
        self.pre_proj = nn.Sequential([nn.Dense(self.dim), nn.gelu, nn.Dense(self.dim)])
        self.post_proj = nn.Sequential(
            [
                nn.Dense(self.dim * 2),
                nn.gelu,
                nn.Dense(self.dim * 2),
                nn.gelu,
                nn.Dense(self.patch_dim_output),
            ]
        )

    def encode(self, data: jnp.ndarray, times: jnp.ndarray) -> jnp.ndarray:
        bs = data.shape[0]
        data = patchify(data, self.config.patch_num)
        data = self.pre_proj(data)
        time_embeddings = self.time_proj(times)[:, :, None, :]
        data = (data + time_embeddings + self.patch_position_embeddings).reshape(
            bs, -1, self.dim
        )
        return data

    def decode(self, data_output: jnp.ndarray) -> jnp.ndarray:
        bs = data_output.shape[0]
        data_output = self.post_proj(data_output)
        p2 = self.config.patch_num_output * self.config.patch_num_output
        data_output = data_output.reshape(bs, -1, p2, self.patch_dim_output)
        data_output = depatchify(
            data_output,
            self.config.patch_num_output,
            self.patch_resolution_output,
            self.patch_resolution_output,
            self.data_dim,
        )
        return data_output


class ConvEmbedder(nn.Module):
    config: EmbedderConfig
    x_num: int
    data_dim: int

    def setup(self):
        assert self.x_num % self.config.patch_num == 0
        assert self.x_num % self.config.patch_num_output == 0

        self.dim = self.config.dim
        self.patch_resolution = self.x_num // self.config.patch_num
        self.patch_resolution_output = self.x_num // self.config.patch_num_output

        self.patch_position_embeddings = self.param(
            "patch_position_embeddings",
            nn.initializers.normal(),
            (1, 1, self.config.patch_num * self.config.patch_num, self.dim),
        )

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

        self.conv_proj_0 = nn.Conv(
            features=self.dim,
            kernel_size=(self.patch_resolution, self.patch_resolution),
            strides=(self.patch_resolution, self.patch_resolution),
            padding="VALID",
            use_bias=True,
        )
        self.conv_proj_1 = nn.Conv(
            features=self.dim, kernel_size=(1, 1), padding="VALID", use_bias=True
        )

        self.conv_dim = self.config.conv_dim
        self.deconv = nn.ConvTranspose(
            features=self.conv_dim,
            kernel_size=(self.patch_resolution_output, self.patch_resolution_output),
            strides=(self.patch_resolution_output, self.patch_resolution_output),
            padding="VALID",
            use_bias=True,
        )
        self.post_conv_0 = nn.Conv(
            features=self.conv_dim, kernel_size=(1, 1), padding="VALID", use_bias=True
        )
        self.post_conv_1 = nn.Conv(
            features=self.data_dim, kernel_size=(1, 1), padding="VALID", use_bias=True
        )

    def encode(self, data: jnp.ndarray, times: jnp.ndarray) -> jnp.ndarray:
        bs, t, h, w, c = data.shape
        x = data.reshape(bs * t, h, w, c)
        x = self.conv_proj_0(x)
        x = nn.gelu(x)
        x = self.conv_proj_1(x)

        p = self.config.patch_num
        x = x.reshape(bs, t, p * p, self.dim)

        if self.config.time_embed == "continuous":
            time_embeddings = self.time_proj(times)[:, :, None, :]
        else:
            time_embeddings = self.time_embed[:, :t, :, :]

        return (x + time_embeddings + self.patch_position_embeddings).reshape(
            bs, -1, self.dim
        )

    def decode(self, data_output: jnp.ndarray) -> jnp.ndarray:
        bs = data_output.shape[0]
        p_out = self.config.patch_num_output
        query_len = data_output.shape[1]
        out_t = query_len // (p_out * p_out)

        x = data_output.reshape(bs, out_t, p_out, p_out, self.dim)
        x = x.reshape(bs * out_t, p_out, p_out, self.dim)

        x = self.deconv(x)
        x = nn.gelu(x)
        x = self.post_conv_0(x)
        x = nn.gelu(x)
        x = self.post_conv_1(x)

        h = x.shape[1]
        w = x.shape[2]
        return x.reshape(bs, out_t, h, w, self.data_dim)


def get_embedder(config: EmbedderConfig, x_num: int, max_output_dim: int):
    if config.type == "linear":
        return LinearEmbedder(config=config, x_num=x_num, data_dim=max_output_dim)
    if config.type == "conv":
        return ConvEmbedder(config=config, x_num=x_num, data_dim=max_output_dim)
    raise ValueError(f"Unknown embedder type: {config.type}")
