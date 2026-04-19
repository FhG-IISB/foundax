"""Equinox reimplementation of DPOTNet (2-D).

Mirrors ``model.py`` but uses ``equinox.Module`` instead of ``flax.linen``.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# AFNO2D – Adaptive Fourier Neural Operator (2D)
# ---------------------------------------------------------------------------


class AFNO2D(eqx.Module):
    width: int = eqx.field(static=True)
    num_blocks: int = eqx.field(static=True)
    channel_first: bool = eqx.field(static=True)
    modes: int = eqx.field(static=True)
    hidden_size_factor: int = eqx.field(static=True)
    act: Callable = eqx.field(static=True)

    w1: jnp.ndarray
    b1: jnp.ndarray
    w2: jnp.ndarray
    b2: jnp.ndarray

    def __init__(
        self,
        width: int,
        num_blocks: int = 8,
        channel_first: bool = False,
        modes: int = 32,
        hidden_size_factor: int = 1,
        act: Callable = jax.nn.gelu,
        *,
        key: jax.Array,
    ):
        self.width = width
        self.num_blocks = num_blocks
        self.channel_first = channel_first
        self.modes = modes
        self.hidden_size_factor = hidden_size_factor
        self.act = act

        block_size = width // num_blocks
        scale = 1.0 / (block_size * block_size * hidden_size_factor)

        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.w1 = scale * jax.random.uniform(
            k1, (2, num_blocks, block_size, block_size * hidden_size_factor)
        )
        self.b1 = scale * jax.random.uniform(
            k2, (2, num_blocks, block_size * hidden_size_factor)
        )
        self.w2 = scale * jax.random.uniform(
            k3, (2, num_blocks, block_size * hidden_size_factor, block_size)
        )
        self.b2 = scale * jax.random.uniform(k4, (2, num_blocks, block_size))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.channel_first:
            x = jnp.transpose(x, (0, 2, 3, 1))

        b, h, w, c = x.shape
        x_orig = x
        block_size = c // self.num_blocks

        x_fft = jnp.fft.rfft2(x, axes=(1, 2), norm="ortho")
        x_fft = x_fft.reshape(
            b, x_fft.shape[1], x_fft.shape[2], self.num_blocks, block_size
        )

        o2_real = jnp.zeros(
            (b, x_fft.shape[1], x_fft.shape[2], self.num_blocks, block_size),
            dtype=jnp.float32,
        )
        o2_imag = jnp.zeros_like(o2_real)

        kept_modes_h = min(self.modes, x_fft.shape[1])
        kept_modes_w = min(self.modes, x_fft.shape[2])
        xr = jnp.real(x_fft[:, :kept_modes_h, :kept_modes_w])
        xi = jnp.imag(x_fft[:, :kept_modes_h, :kept_modes_w])

        o1r = self.act(
            jnp.einsum("...bi,bio->...bo", xr, self.w1[0])
            - jnp.einsum("...bi,bio->...bo", xi, self.w1[1])
            + self.b1[0]
        )
        o1i = self.act(
            jnp.einsum("...bi,bio->...bo", xi, self.w1[0])
            + jnp.einsum("...bi,bio->...bo", xr, self.w1[1])
            + self.b1[1]
        )

        o2r = (
            jnp.einsum("...bi,bio->...bo", o1r, self.w2[0])
            - jnp.einsum("...bi,bio->...bo", o1i, self.w2[1])
            + self.b2[0]
        )
        o2i = (
            jnp.einsum("...bi,bio->...bo", o1i, self.w2[0])
            + jnp.einsum("...bi,bio->...bo", o1r, self.w2[1])
            + self.b2[1]
        )

        o2_real = o2_real.at[:, :kept_modes_h, :kept_modes_w].set(o2r)
        o2_imag = o2_imag.at[:, :kept_modes_h, :kept_modes_w].set(o2i)

        x_fft2 = (o2_real + 1j * o2_imag).reshape(b, x_fft.shape[1], x_fft.shape[2], c)
        x = jnp.fft.irfft2(x_fft2, s=(h, w), axes=(1, 2), norm="ortho")
        x = x + x_orig

        if self.channel_first:
            x = jnp.transpose(x, (0, 3, 1, 2))
        return x


# ---------------------------------------------------------------------------
# Block – GroupNorm + AFNO + MLP with residual
# ---------------------------------------------------------------------------


class Block(eqx.Module):
    mixing_type: str = eqx.field(static=True)
    double_skip: bool = eqx.field(static=True)
    width: int = eqx.field(static=True)
    act: Callable = eqx.field(static=True)

    norm1: eqx.nn.GroupNorm
    afno: AFNO2D
    norm2: eqx.nn.GroupNorm
    mlp_dense_1: eqx.nn.Linear
    mlp_dense_2: eqx.nn.Linear

    def __init__(
        self,
        mixing_type: str = "afno",
        double_skip: bool = True,
        width: int = 32,
        n_blocks: int = 4,
        mlp_ratio: float = 1.0,
        channel_first: bool = False,
        modes: int = 32,
        act: Callable = jax.nn.gelu,
        *,
        key: jax.Array,
    ):
        self.mixing_type = mixing_type
        self.double_skip = double_skip
        self.width = width
        self.act = act

        k1, k2, k3 = jax.random.split(key, 3)

        # GroupNorm with 8 groups, channels-last
        self.norm1 = eqx.nn.GroupNorm(
            groups=8, channels=width, eps=1e-5, channelwise_affine=True
        )
        self.norm2 = eqx.nn.GroupNorm(
            groups=8, channels=width, eps=1e-5, channelwise_affine=True
        )

        self.afno = AFNO2D(
            width=width,
            num_blocks=n_blocks,
            channel_first=channel_first,
            modes=modes,
            act=act,
            key=k1,
        )

        hidden = int(width * mlp_ratio)
        self.mlp_dense_1 = eqx.nn.Linear(width, hidden, key=k2)
        self.mlp_dense_2 = eqx.nn.Linear(hidden, width, key=k3)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x

        # GroupNorm: Flax operates on last dim channel-last; Equinox expects (C, ...).
        # Input is (B, H, W, C) → vmap over batch, apply GN on (C, H, W).
        x = jax.vmap(lambda v: self.norm1(v.transpose(2, 0, 1)).transpose(1, 2, 0))(x)

        if self.mixing_type == "afno":
            x = self.afno(x)
        else:
            raise ValueError(f"Unsupported mixing_type: {self.mixing_type}")

        if self.double_skip:
            x = x + residual
            residual = x

        x = jax.vmap(lambda v: self.norm2(v.transpose(2, 0, 1)).transpose(1, 2, 0))(x)

        # Dense applied pointwise (B, H, W, C) → vmap over B, H, W
        x = jax.vmap(jax.vmap(jax.vmap(self.mlp_dense_1)))(x)
        x = self.act(x)
        x = jax.vmap(jax.vmap(jax.vmap(self.mlp_dense_2)))(x)
        x = x + residual
        return x


# ---------------------------------------------------------------------------
# PatchEmbed – spatial patching via convolutions
# ---------------------------------------------------------------------------


class PatchEmbed(eqx.Module):
    img_size: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    act: Callable = eqx.field(static=True)

    conv_patch: eqx.nn.Conv2d
    conv_1x1: eqx.nn.Conv2d

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        out_dim: int,
        act: Callable = jax.nn.gelu,
        *,
        key: jax.Array,
    ):
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_dim = out_dim
        self.act = act

        k1, k2 = jax.random.split(key)
        self.conv_patch = eqx.nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            use_bias=True,
            key=k1,
        )
        self.conv_1x1 = eqx.nn.Conv2d(
            embed_dim,
            out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            use_bias=True,
            key=k2,
        )

    @property
    def out_size(self) -> Tuple[int, int]:
        s = self.img_size // self.patch_size
        return (s, s)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, H, W, C) → NCHW for eqx.nn.Conv2d
        x = x.transpose(0, 3, 1, 2)  # (B, C, H, W)
        x = jax.vmap(self.conv_patch)(x)  # (B, embed_dim, pH, pW)
        x = self.act(x)
        x = jax.vmap(self.conv_1x1)(x)  # (B, out_dim, pH, pW)
        x = x.transpose(0, 2, 3, 1)  # (B, pH, pW, out_dim)
        return x


# ---------------------------------------------------------------------------
# TimeAggregator – temporal reduction
# ---------------------------------------------------------------------------


class TimeAggregator(eqx.Module):
    n_channels: int = eqx.field(static=True)
    n_timesteps: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    kind: str = eqx.field(static=True)

    w: jnp.ndarray
    gamma: Optional[jnp.ndarray]

    def __init__(
        self,
        n_channels: int,
        n_timesteps: int,
        out_channels: int,
        kind: str = "mlp",
        *,
        key: jax.Array,
    ):
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.out_channels = out_channels
        self.kind = kind

        k1, k2 = jax.random.split(key)
        self.w = jax.random.normal(k1, (n_timesteps, out_channels, out_channels)) / (
            n_timesteps * (out_channels**0.5)
        )

        if kind == "exp_mlp":
            self.gamma = (
                2.0 ** jnp.linspace(-10.0, 10.0, out_channels, dtype=jnp.float32)
            )[None, :]
        else:
            self.gamma = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.kind == "mlp":
            return jnp.einsum("tij,...ti->...j", self.w, x)

        if self.kind == "exp_mlp":
            t = jnp.linspace(0.0, 1.0, x.shape[-2], dtype=jnp.float32)[:, None]
            t_embed = jnp.cos(t @ self.gamma)
            return jnp.einsum("tij,...ti->...j", self.w, x * t_embed)

        raise ValueError(f"Unsupported TimeAggregator type: {self.kind}")


# ---------------------------------------------------------------------------
# DPOTNet – main 2D model
# ---------------------------------------------------------------------------


class DPOTNet(eqx.Module):
    img_size: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    mixing_type: str = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    in_timesteps: int = eqx.field(static=True)
    out_timesteps: int = eqx.field(static=True)
    n_blocks: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    out_layer_dim: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    modes: int = eqx.field(static=True)
    mlp_ratio: float = eqx.field(static=True)
    n_cls: int = eqx.field(static=True)
    normalize: bool = eqx.field(static=True)
    act: Callable = eqx.field(static=True)
    time_agg: str = eqx.field(static=True)

    # Sub-modules
    patch_embed: PatchEmbed
    time_agg_layer: TimeAggregator
    pos_embed: jnp.ndarray
    blocks: list
    cls_dense_1: eqx.nn.Linear
    cls_dense_2: eqx.nn.Linear
    cls_dense_3: eqx.nn.Linear
    out_deconv: eqx.nn.ConvTranspose2d
    out_conv_1: eqx.nn.Conv2d
    out_conv_2: eqx.nn.Conv2d

    # Optional normalization layers (only when normalize=True)
    scale_feats_mu: Optional[eqx.nn.Linear]
    scale_feats_sigma: Optional[eqx.nn.Linear]

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        mixing_type: str = "afno",
        in_channels: int = 1,
        out_channels: int = 4,
        in_timesteps: int = 1,
        out_timesteps: int = 1,
        n_blocks: int = 4,
        embed_dim: int = 768,
        out_layer_dim: int = 32,
        depth: int = 12,
        modes: int = 32,
        mlp_ratio: float = 1.0,
        n_cls: int = 12,
        normalize: bool = False,
        act: Callable = jax.nn.gelu,
        time_agg: str = "exp_mlp",
        *,
        key: jax.Array,
    ):
        self.img_size = img_size
        self.patch_size = patch_size
        self.mixing_type = mixing_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.n_blocks = n_blocks
        self.embed_dim = embed_dim
        self.out_layer_dim = out_layer_dim
        self.depth = depth
        self.modes = modes
        self.mlp_ratio = mlp_ratio
        self.n_cls = n_cls
        self.normalize = normalize
        self.act = act
        self.time_agg = time_agg

        keys = jax.random.split(key, 8 + depth)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels + 3,
            embed_dim=out_channels * patch_size + 3,
            out_dim=embed_dim,
            act=act,
            key=keys[0],
        )

        ph = img_size // patch_size
        self.pos_embed = (
            jax.random.truncated_normal(keys[1], -2.0, 2.0, (1, ph, ph, embed_dim))
            * 0.02
        )

        self.time_agg_layer = TimeAggregator(
            n_channels=in_channels,
            n_timesteps=in_timesteps,
            out_channels=embed_dim,
            kind=time_agg,
            key=keys[2],
        )

        self.blocks = [
            Block(
                mixing_type=mixing_type,
                width=embed_dim,
                mlp_ratio=mlp_ratio,
                channel_first=False,
                n_blocks=n_blocks,
                modes=modes,
                double_skip=False,
                act=act,
                key=keys[3 + i],
            )
            for i in range(depth)
        ]

        bk = keys[3 + depth]
        k1, k2, k3 = jax.random.split(bk, 3)
        self.cls_dense_1 = eqx.nn.Linear(embed_dim, embed_dim, key=k1)
        self.cls_dense_2 = eqx.nn.Linear(embed_dim, embed_dim, key=k2)
        self.cls_dense_3 = eqx.nn.Linear(embed_dim, n_cls, key=k3)

        ok = keys[4 + depth]
        k1, k2, k3 = jax.random.split(ok, 3)
        self.out_deconv = eqx.nn.ConvTranspose2d(
            embed_dim,
            out_layer_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            use_bias=True,
            key=k1,
        )
        self.out_conv_1 = eqx.nn.Conv2d(
            out_layer_dim,
            out_layer_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            use_bias=True,
            key=k2,
        )
        self.out_conv_2 = eqx.nn.Conv2d(
            out_layer_dim,
            out_channels * out_timesteps,
            kernel_size=1,
            stride=1,
            padding=0,
            use_bias=True,
            key=k3,
        )

        if normalize:
            nk = keys[5 + depth]
            k1, k2 = jax.random.split(nk)
            self.scale_feats_mu = eqx.nn.Linear(in_channels * 2, embed_dim, key=k1)
            self.scale_feats_sigma = eqx.nn.Linear(in_channels * 2, embed_dim, key=k2)
        else:
            self.scale_feats_mu = None
            self.scale_feats_sigma = None

    def _get_grid_3d(self, x: jnp.ndarray) -> jnp.ndarray:
        b, sx, sy, sz, _ = x.shape
        gridx = jnp.linspace(0.0, 1.0, sx, dtype=jnp.float32).reshape(1, sx, 1, 1, 1)
        gridy = jnp.linspace(0.0, 1.0, sy, dtype=jnp.float32).reshape(1, 1, sy, 1, 1)
        gridz = jnp.linspace(0.0, 1.0, sz, dtype=jnp.float32).reshape(1, 1, 1, sz, 1)
        gridx = jnp.tile(gridx, (b, 1, sy, sz, 1))
        gridy = jnp.tile(gridy, (b, sx, 1, sz, 1))
        gridz = jnp.tile(gridz, (b, sx, sy, 1, 1))
        return jnp.concatenate((gridx, gridy, gridz), axis=-1)

    def __call__(self, x: jnp.ndarray):
        b, _, _, t, _ = x.shape

        if self.normalize:
            mu = jnp.mean(x, axis=(1, 2, 3), keepdims=True)
            sigma = jnp.std(x, axis=(1, 2, 3), keepdims=True) + 1e-6
            x = (x - mu) / sigma
            feats = jnp.concatenate([mu, sigma], axis=-1)
            # feats: (B, 1, 1, 1, 2*C) → squeeze and apply Linear
            feats_flat = feats.reshape(b, -1)
            scale_mu = jax.vmap(self.scale_feats_mu)(feats_flat)[:, None, None, :]
            scale_sigma = jax.vmap(self.scale_feats_sigma)(feats_flat)[:, None, None, :]

        grid = self._get_grid_3d(x)
        x = jnp.concatenate((x, grid), axis=-1)
        x = jnp.transpose(x, (0, 3, 1, 2, 4)).reshape(
            b * t, x.shape[1], x.shape[2], x.shape[4]
        )

        x = self.patch_embed(x)  # (B*T, pH, pW, embed_dim)

        ph, pw = self.patch_embed.out_size
        x = x + self.pos_embed

        x = x.reshape(b, t, ph, pw, self.embed_dim)
        x = jnp.transpose(x, (0, 2, 3, 1, 4))

        x = self.time_agg_layer(x)

        if self.normalize:
            x = scale_sigma * x + scale_mu

        for block in self.blocks:
            x = block(x)

        # Classification head
        cls_token = jnp.mean(x, axis=(1, 2))
        cls_pred = jax.vmap(self.cls_dense_1)(cls_token)
        cls_pred = self.act(cls_pred)
        cls_pred = jax.vmap(self.cls_dense_2)(cls_pred)
        cls_pred = self.act(cls_pred)
        cls_pred = jax.vmap(self.cls_dense_3)(cls_pred)

        # Output head: NHWC → NCHW for conv, then back
        x = x.transpose(0, 3, 1, 2)  # (B, C, H, W)
        x = jax.vmap(self.out_deconv)(x)  # (B, out_layer_dim, H', W')
        x = self.act(x)
        x = jax.vmap(self.out_conv_1)(x)
        x = self.act(x)
        x = jax.vmap(self.out_conv_2)(x)
        x = x.transpose(0, 2, 3, 1)  # (B, H, W, C_out*T_out)

        h, w = x.shape[1], x.shape[2]
        x = x.reshape(b, h, w, self.out_timesteps, self.out_channels)

        if self.normalize:
            x = x * sigma + mu

        return x, cls_pred
