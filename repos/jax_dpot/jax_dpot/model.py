from __future__ import annotations

from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp


def _activation(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    table = {
        "gelu": lambda x: nn.gelu(x, approximate=False),
        "tanh": jnp.tanh,
        "sigmoid": jax.nn.sigmoid,
        "relu": jax.nn.relu,
        "leaky_relu": lambda x: jax.nn.leaky_relu(x, negative_slope=0.1),
        "softplus": jax.nn.softplus,
        "ELU": jax.nn.elu,
        "silu": jax.nn.silu,
    }
    if name not in table:
        raise ValueError(f"Unsupported activation: {name}")
    return table[name]


class AFNO2D(nn.Module):
    width: int
    num_blocks: int = 8
    channel_first: bool = False
    sparsity_threshold: float = 0.01
    modes: int = 32
    hidden_size_factor: int = 1
    act: str = "gelu"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.channel_first:
            x = jnp.transpose(x, (0, 2, 3, 1))

        b, h, w, c = x.shape
        x_orig = x

        assert (
            c % self.num_blocks == 0
        ), f"hidden_size {c} should be divisible by num_blocks {self.num_blocks}"
        block_size = c // self.num_blocks
        scale = 1.0 / (block_size * block_size * self.hidden_size_factor)

        w1 = self.param(
            "w1",
            lambda key, shape: scale
            * jax.random.uniform(key, shape, dtype=jnp.float32),
            (2, self.num_blocks, block_size, block_size * self.hidden_size_factor),
        )
        b1 = self.param(
            "b1",
            lambda key, shape: scale
            * jax.random.uniform(key, shape, dtype=jnp.float32),
            (2, self.num_blocks, block_size * self.hidden_size_factor),
        )
        w2 = self.param(
            "w2",
            lambda key, shape: scale
            * jax.random.uniform(key, shape, dtype=jnp.float32),
            (2, self.num_blocks, block_size * self.hidden_size_factor, block_size),
        )
        b2 = self.param(
            "b2",
            lambda key, shape: scale
            * jax.random.uniform(key, shape, dtype=jnp.float32),
            (2, self.num_blocks, block_size),
        )

        x_fft = jnp.fft.rfft2(x, axes=(1, 2), norm="ortho")
        x_fft = x_fft.reshape(
            b, x_fft.shape[1], x_fft.shape[2], self.num_blocks, block_size
        )

        o1_real = jnp.zeros(
            (
                b,
                x_fft.shape[1],
                x_fft.shape[2],
                self.num_blocks,
                block_size * self.hidden_size_factor,
            ),
            dtype=jnp.float32,
        )
        o1_imag = jnp.zeros_like(o1_real)
        o2_real = jnp.zeros(
            (b, x_fft.shape[1], x_fft.shape[2], self.num_blocks, block_size),
            dtype=jnp.float32,
        )
        o2_imag = jnp.zeros_like(o2_real)

        kept_modes_h = min(self.modes, x_fft.shape[1])
        kept_modes_w = min(self.modes, x_fft.shape[2])
        xr = jnp.real(x_fft[:, :kept_modes_h, :kept_modes_w])
        xi = jnp.imag(x_fft[:, :kept_modes_h, :kept_modes_w])

        act = _activation(self.act)
        o1r = act(
            jnp.einsum("...bi,bio->...bo", xr, w1[0])
            - jnp.einsum("...bi,bio->...bo", xi, w1[1])
            + b1[0]
        )
        o1i = act(
            jnp.einsum("...bi,bio->...bo", xi, w1[0])
            + jnp.einsum("...bi,bio->...bo", xr, w1[1])
            + b1[1]
        )

        o2r = (
            jnp.einsum("...bi,bio->...bo", o1r, w2[0])
            - jnp.einsum("...bi,bio->...bo", o1i, w2[1])
            + b2[0]
        )
        o2i = (
            jnp.einsum("...bi,bio->...bo", o1i, w2[0])
            + jnp.einsum("...bi,bio->...bo", o1r, w2[1])
            + b2[1]
        )

        o1_real = o1_real.at[:, :kept_modes_h, :kept_modes_w].set(o1r)
        o1_imag = o1_imag.at[:, :kept_modes_h, :kept_modes_w].set(o1i)
        o2_real = o2_real.at[:, :kept_modes_h, :kept_modes_w].set(o2r)
        o2_imag = o2_imag.at[:, :kept_modes_h, :kept_modes_w].set(o2i)

        x_fft2 = (o2_real + 1j * o2_imag).reshape(b, x_fft.shape[1], x_fft.shape[2], c)
        x = jnp.fft.irfft2(x_fft2, s=(h, w), axes=(1, 2), norm="ortho")
        x = x + x_orig

        if self.channel_first:
            x = jnp.transpose(x, (0, 3, 1, 2))
        return x


class Block(nn.Module):
    mixing_type: str = "afno"
    double_skip: bool = True
    width: int = 32
    n_blocks: int = 4
    mlp_ratio: float = 1.0
    channel_first: bool = False
    modes: int = 32
    act: str = "gelu"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = nn.GroupNorm(num_groups=8, epsilon=1e-5)(x)
        if self.mixing_type == "afno":
            x = AFNO2D(
                width=self.width,
                num_blocks=self.n_blocks,
                channel_first=self.channel_first,
                modes=self.modes,
                act=self.act,
            )(x)
        else:
            raise ValueError(f"Unsupported mixing_type: {self.mixing_type}")

        if self.double_skip:
            x = x + residual
            residual = x

        x = nn.GroupNorm(num_groups=8, epsilon=1e-5)(x)
        hidden = int(self.width * self.mlp_ratio)
        x = nn.Dense(hidden, use_bias=True, name="mlp_dense_1")(x)
        x = _activation(self.act)(x)
        x = nn.Dense(self.width, use_bias=True, name="mlp_dense_2")(x)
        x = x + residual
        return x


class PatchEmbed(nn.Module):
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768
    out_dim: int = 128
    act: str = "gelu"

    @property
    def out_size(self) -> tuple[int, int]:
        return (self.img_size // self.patch_size, self.img_size // self.patch_size)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b, h, w, c = x.shape
        assert (
            h == self.img_size and w == self.img_size
        ), f"Input image size ({h}*{w}) does not match model ({self.img_size}*{self.img_size})."
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            use_bias=True,
            name="conv_patch",
        )(x)
        x = _activation(self.act)(x)
        x = nn.Conv(
            features=self.out_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=True,
            name="conv_1x1",
        )(x)
        return x


class TimeAggregator(nn.Module):
    n_channels: int
    n_timesteps: int
    out_channels: int
    kind: str = "mlp"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        w = self.param(
            "w",
            lambda key, shape: jax.random.normal(key, shape, dtype=jnp.float32)
            / (self.n_timesteps * (self.out_channels**0.5)),
            (self.n_timesteps, self.out_channels, self.out_channels),
        )

        if self.kind == "mlp":
            return jnp.einsum("tij,...ti->...j", w, x)

        if self.kind == "exp_mlp":
            gamma = self.param(
                "gamma",
                lambda _k, shape: (
                    2.0 ** jnp.linspace(-10.0, 10.0, shape[1], dtype=jnp.float32)
                )[None, :],
                (1, self.out_channels),
            )
            t = jnp.linspace(0.0, 1.0, x.shape[-2], dtype=jnp.float32)[:, None]
            t_embed = jnp.cos(t @ gamma)
            return jnp.einsum("tij,...ti->...j", w, x * t_embed)

        raise ValueError(f"Unsupported TimeAggregator type: {self.kind}")


class DPOTNet(nn.Module):
    img_size: int = 224
    patch_size: int = 16
    mixing_type: str = "afno"
    in_channels: int = 1
    out_channels: int = 4
    in_timesteps: int = 1
    out_timesteps: int = 1
    n_blocks: int = 4
    embed_dim: int = 768
    out_layer_dim: int = 32
    depth: int = 12
    modes: int = 32
    mlp_ratio: float = 1.0
    n_cls: int = 12
    normalize: bool = False
    act: str = "gelu"
    time_agg: str = "exp_mlp"

    def _get_grid_3d(self, x: jnp.ndarray) -> jnp.ndarray:
        b, sx, sy, sz, _ = x.shape
        gridx = jnp.linspace(0.0, 1.0, sx, dtype=jnp.float32).reshape(1, sx, 1, 1, 1)
        gridy = jnp.linspace(0.0, 1.0, sy, dtype=jnp.float32).reshape(1, 1, sy, 1, 1)
        gridz = jnp.linspace(0.0, 1.0, sz, dtype=jnp.float32).reshape(1, 1, 1, sz, 1)
        gridx = jnp.tile(gridx, (b, 1, sy, sz, 1))
        gridy = jnp.tile(gridy, (b, sx, 1, sz, 1))
        gridz = jnp.tile(gridz, (b, sx, sy, 1, 1))
        return jnp.concatenate((gridx, gridy, gridz), axis=-1)

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        b, _, _, t, _ = x.shape

        if self.normalize:
            mu = jnp.mean(x, axis=(1, 2, 3), keepdims=True)
            sigma = jnp.std(x, axis=(1, 2, 3), keepdims=True) + 1e-6
            x = (x - mu) / sigma
            feats = jnp.concatenate([mu, sigma], axis=-1)
            scale_mu = nn.Dense(self.embed_dim, use_bias=True, name="scale_feats_mu")(
                feats
            ).squeeze(axis=3)
            scale_sigma = nn.Dense(
                self.embed_dim, use_bias=True, name="scale_feats_sigma"
            )(feats).squeeze(axis=3)

        grid = self._get_grid_3d(x)
        x = jnp.concatenate((x, grid), axis=-1)
        x = jnp.transpose(x, (0, 3, 1, 2, 4)).reshape(
            b * t, x.shape[1], x.shape[2], x.shape[4]
        )

        patch = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_channels + 3,
            embed_dim=self.out_channels * self.patch_size + 3,
            out_dim=self.embed_dim,
            act=self.act,
            name="patch_embed",
        )
        x = patch(x)

        ph, pw = patch.out_size
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.truncated_normal(stddev=0.02),
            (1, ph, pw, self.embed_dim),
        )
        x = x + pos_embed

        x = x.reshape(b, t, ph, pw, self.embed_dim)
        x = jnp.transpose(x, (0, 2, 3, 1, 4))

        x = TimeAggregator(
            n_channels=self.in_channels,
            n_timesteps=self.in_timesteps,
            out_channels=self.embed_dim,
            kind=self.time_agg,
            name="time_agg_layer",
        )(x)

        if self.normalize:
            x = scale_sigma * x + scale_mu

        for i in range(self.depth):
            x = Block(
                mixing_type=self.mixing_type,
                width=self.embed_dim,
                mlp_ratio=self.mlp_ratio,
                channel_first=False,
                n_blocks=self.n_blocks,
                modes=self.modes,
                double_skip=False,
                act=self.act,
                name=f"blocks_{i}",
            )(x)

        cls_token = jnp.mean(x, axis=(1, 2))
        cls_pred = nn.Dense(self.embed_dim, use_bias=True, name="cls_dense_1")(
            cls_token
        )
        cls_pred = _activation(self.act)(cls_pred)
        cls_pred = nn.Dense(self.embed_dim, use_bias=True, name="cls_dense_2")(cls_pred)
        cls_pred = _activation(self.act)(cls_pred)
        cls_pred = nn.Dense(self.n_cls, use_bias=True, name="cls_dense_3")(cls_pred)

        x = nn.ConvTranspose(
            features=self.out_layer_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            use_bias=True,
            transpose_kernel=True,
            name="out_deconv",
        )(x)
        x = _activation(self.act)(x)
        x = nn.Conv(
            features=self.out_layer_dim,
            kernel_size=(1, 1),
            padding="VALID",
            use_bias=True,
            name="out_conv_1",
        )(x)
        x = _activation(self.act)(x)
        x = nn.Conv(
            features=self.out_channels * self.out_timesteps,
            kernel_size=(1, 1),
            padding="VALID",
            use_bias=True,
            name="out_conv_2",
        )(x)

        h, w = x.shape[1], x.shape[2]
        x = x.reshape(b, h, w, self.out_timesteps, self.out_channels)

        if self.normalize:
            x = x * sigma + mu

        return x, cls_pred
