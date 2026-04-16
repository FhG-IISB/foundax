from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from .filter_networks import LReLuRegular, LReLuTorch
from .model import AFNO2D, Block, TimeAggregator, _activation


class CNOBlock(nn.Module):
    in_channels: int
    out_channels: int
    in_size: int
    out_size: int
    cutoff_den: float = 2.0001
    conv_kernel: int = 3
    activation: str = "cno_lrelu_torch"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Low-pass frequency filter
        b, h, w, c = x.shape
        fft_x = jnp.fft.fft2(x, axes=(1, 2))
        cutoff = h // self.conv_kernel
        mask = jnp.zeros((h, w), dtype=jnp.bool_)
        mask = mask.at[:cutoff, :cutoff].set(True)
        x = jnp.fft.ifft2(fft_x * mask[None, :, :, None], axes=(1, 2)).real

        # Convolution
        pad = (self.conv_kernel - 1) // 2
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.conv_kernel, self.conv_kernel),
            padding=((pad, pad), (pad, pad)),
            use_bias=True,
            name="conv",
        )(x)

        # Activation with up/downsampling
        if self.activation == "cno_lrelu_torch":
            x = LReLuTorch(
                channels=self.out_channels,
                in_size=self.in_size,
                out_size=self.out_size,
                name="lrelu_torch",
            )(x)
        elif self.activation == "lrelu":
            x = LReLuRegular(
                in_size=self.in_size,
                out_size=self.out_size,
                name="lrelu_regular",
            )(x)
        else:
            raise ValueError(f"Unsupported CNOBlock activation: {self.activation}")
        return x


class CNOPatchEmbed(nn.Module):
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768
    out_dim: int = 128
    act: str = "cno_lrelu_torch"

    @property
    def out_size(self) -> tuple[int, int]:
        return (self.img_size // self.patch_size, self.img_size // self.patch_size)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        ps = self.out_size[0]
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            use_bias=True,
            name="conv_patch",
        )(x)
        x = LReLuTorch(
            channels=self.embed_dim,
            in_size=ps,
            out_size=ps,
            name="act_patching",
        )(x)
        x = nn.Conv(
            features=self.out_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=True,
            name="conv_1x1",
        )(x)
        return x


class CDPOTNet(nn.Module):
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
        latent_size = self.img_size // self.patch_size

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

        patch = CNOPatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_channels + 3,
            embed_dim=self.out_channels * self.patch_size + 3,
            out_dim=self.embed_dim,
            act="cno_lrelu_torch",
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

        # cls head
        cls_token = jnp.mean(x, axis=(1, 2))
        cls_pred = nn.Dense(self.embed_dim, use_bias=True, name="cls_dense_1")(
            cls_token
        )
        cls_pred = _activation(self.act)(cls_pred)
        cls_pred = nn.Dense(self.embed_dim, use_bias=True, name="cls_dense_2")(cls_pred)
        cls_pred = _activation(self.act)(cls_pred)
        cls_pred = nn.Dense(self.n_cls, use_bias=True, name="cls_dense_3")(cls_pred)

        # Output head: CNOBlock (upsamples from latent to full resolution) + 1x1 convs
        x = CNOBlock(
            in_channels=self.embed_dim,
            out_channels=self.out_layer_dim,
            in_size=latent_size,
            out_size=self.img_size,
            cutoff_den=2.0001,
            conv_kernel=1,
            activation="cno_lrelu_torch",
            name="out_cno_block",
        )(x)
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
