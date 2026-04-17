"""
Walrus IsotropicModel — Equinox implementation.

1-to-1 Equinox translation of the Flax Linen ``IsotropicModel`` for
pretrained weight transfer and forward-pass equivalence.
"""

from __future__ import annotations

import math
from math import pi
from typing import List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

BC_PERIODIC = 2

_PATCH_DICT = {
    0: (1, 1),
    1: (1, 1),
    4: (2, 2),
    8: (4, 2),
    12: (6, 2),
    16: (4, 4),
    24: (6, 4),
    32: (8, 4),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions (pure, no learnable params)
# ═══════════════════════════════════════════════════════════════════════════════


def choose_kernel_size_deterministic(x_shape):
    dims = len(x_shape)
    non_singleton = sum(int(s != 1) for s in x_shape)
    if dims == 1:
        per_axis = 512 // 16
        return (_PATCH_DICT[x_shape[0] // per_axis],)
    elif dims == 2:
        per_axis = 512 // 16
        H, W = x_shape
        return (_PATCH_DICT[H // per_axis], _PATCH_DICT[W // per_axis])
    elif dims == 3:
        per_axis = 256 // 16 if non_singleton >= 3 else 512 // 16
        H, W, D = x_shape
        h_p = H // per_axis if H != 1 else 0
        w_p = W // per_axis if W != 1 else 0
        d_p = D // per_axis if D != 1 else 0
        return (_PATCH_DICT[h_p], _PATCH_DICT[w_p], _PATCH_DICT[d_p])
    else:
        raise ValueError(f"Spatial dims must be 1-3, got {dims}")


def _conv3d(x, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0)):
    x_t = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_t = jnp.transpose(weight, (2, 3, 4, 1, 0))
    out = jax.lax.conv_general_dilated(
        x_t,
        w_t,
        window_strides=stride,
        padding=[(p, p) for p in padding],
        dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
    )
    if bias is not None:
        out = out + bias[None, None, None, None, :]
    return jnp.transpose(out, (0, 4, 1, 2, 3))


def _conv_transpose3d(x, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0)):
    N, C_in, D, H, W = x.shape
    sD, sH, sW = stride
    kD, kH, kW = weight.shape[2:]
    if any(s > 1 for s in stride):
        D_up = D + (D - 1) * (sD - 1)
        H_up = H + (H - 1) * (sH - 1)
        W_up = W + (W - 1) * (sW - 1)
        x_up = jnp.zeros((N, C_in, D_up, H_up, W_up), dtype=x.dtype)
        x_up = x_up.at[:, :, ::sD, ::sH, ::sW].set(x)
        x = x_up
    weight = weight[:, :, ::-1, ::-1, ::-1]
    eff_padding = [(k - 1 - p, k - 1 - p) for k, p in zip([kD, kH, kW], padding)]
    x_t = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_t = jnp.transpose(weight, (2, 3, 4, 1, 0))
    out = jax.lax.conv_general_dilated(
        x_t,
        w_t,
        window_strides=(1, 1, 1),
        padding=eff_padding,
        dimension_numbers=("NDHWC", "DHWOI", "NDHWC"),
    )
    if bias is not None:
        out = out + bias[None, None, None, None, :]
    return jnp.transpose(out, (0, 4, 1, 2, 3))


def _conv3d_1x1(x, weight, bias=None):
    w = weight[:, :, 0, 0, 0]
    x_t = jnp.transpose(x, (0, 2, 3, 4, 1))
    out = x_t @ w.T
    if bias is not None:
        out = out + bias
    return jnp.transpose(out, (0, 4, 1, 2, 3))


def _compute_padding(
    shape, bcs, n_dims, max_d, base_kernel, random_kernel, jitter_patches
):
    constant_paddings = []
    periodic_paddings = []
    effective_ps = 0
    effective_stride = 0
    for i in range(max_d):
        if i >= n_dims or shape[i] == 1:
            periodic_paddings = [0, 0] + periodic_paddings
            constant_paddings = [0, 0] + constant_paddings
            continue
        base_k1 = base_kernel[i][0]
        base_k2 = base_kernel[i][1]
        s1 = random_kernel[i][0]
        s2 = random_kernel[i][1]
        effective_ps = base_k1 + s1 * (base_k2 - 1)
        effective_stride = s1 * s2
        extra_padding = (effective_ps - effective_stride) // 2
        is_periodic = bcs[i][0] == BC_PERIODIC
        if is_periodic:
            jitter_pad = [0, 0]
        else:
            jitter_pad = (
                [effective_stride // 2, effective_stride // 2]
                if jitter_patches
                else [0, 0]
            )
        axis_pad = [p + extra_padding for p in jitter_pad]
        if is_periodic:
            periodic_paddings = axis_pad + periodic_paddings
            constant_paddings = [0, 0] + constant_paddings
        else:
            constant_paddings = axis_pad + constant_paddings
            periodic_paddings = [0, 0] + periodic_paddings
    return constant_paddings, periodic_paddings, effective_ps, effective_stride


def _pad_nd(x, paddings, mode="constant"):
    if sum(abs(p) for p in paddings) == 0:
        return x
    n_spatial = len(paddings) // 2
    pad_per_axis = []
    for i in range(n_spatial):
        idx = len(paddings) - 2 * (i + 1)
        pad_per_axis.append((paddings[idx], paddings[idx + 1]))
    full_pad = [(0, 0)] * (x.ndim - n_spatial) + pad_per_axis
    if mode == "constant":
        return jnp.pad(x, full_pad, mode="constant", constant_values=0)
    elif mode == "circular":
        return jnp.pad(x, full_pad, mode="wrap")
    else:
        raise ValueError(f"Unsupported pad mode: {mode}")


def _slice_padding(x, paddings, n_leading_dims):
    if sum(abs(p) for p in paddings) == 0:
        return x
    n_spatial = len(paddings) // 2
    slices = [slice(None)] * n_leading_dims
    for i in range(n_spatial):
        idx = len(paddings) - 2 * (i + 1)
        ps, pe = paddings[idx], paddings[idx + 1]
        dim_size = x.shape[n_leading_dims + i]
        if ps + pe > 0:
            slices.append(slice(ps, dim_size - pe if pe > 0 else None))
        else:
            slices.append(slice(None))
    return x[tuple(slices)]


def _jitter_forward(
    x, bcs, n_dims, max_d, base_kernel, random_kernel, jitter_patches, rng_key
):
    T = x.shape[0]
    shape = x.shape[3:]
    const_pad, periodic_pad, _, _ = _compute_padding(
        shape, bcs, n_dims, max_d, base_kernel, random_kernel, jitter_patches
    )
    x_flat = rearrange(x, "t b c h w d -> (t b) c h w d")
    x_flat = _pad_nd(x_flat, const_pad, mode="constant")
    x = rearrange(x_flat, "(t b) c h w d -> t b c h w d", t=T)
    bc_flag_shape = list(x.shape)
    bc_flag_shape[2] = 3
    bc_flags = jnp.zeros(bc_flag_shape, dtype=x.dtype)
    bc_flags = bc_flags.at[:, :, 0].set(1.0)
    dim_offset = 3
    roll_quantities = []
    roll_dims = []
    for i in range(max_d):
        if i >= n_dims or shape[i] == 1:
            continue
        is_periodic = bcs[i][0] == BC_PERIODIC
        if not is_periodic:
            pad_idx = len(const_pad) - 2 * (i + 1)
            pad_start = const_pad[pad_idx]
            pad_end = const_pad[pad_idx + 1]
            if pad_start > 0:
                sl = [slice(None)] * len(bc_flags.shape)
                sl[i + dim_offset] = slice(None, pad_start)
                sl[2] = 1 + int(bcs[i][0])
                bc_flags = bc_flags.at[tuple(sl)].add(1.0)
                sl[2] = 0
                bc_flags = bc_flags.at[tuple(sl)].set(0.0)
            if pad_end > 0:
                sl = [slice(None)] * len(bc_flags.shape)
                sl[i + dim_offset] = slice(-pad_end, None)
                sl[2] = 1 + int(bcs[i][1])
                bc_flags = bc_flags.at[tuple(sl)].add(1.0)
                sl[2] = 0
                bc_flags = bc_flags.at[tuple(sl)].set(0.0)
        if jitter_patches and rng_key is not None:
            total_pad_idx = len(const_pad) - 2 * (i + 1)
            total_pad = const_pad[total_pad_idx + 1] + periodic_pad[total_pad_idx + 1]
            half_patch = total_pad if not is_periodic else x.shape[i + dim_offset] // 2
            rng_key, subkey = jax.random.split(rng_key)
            if half_patch > 1:
                roll_rate = int(
                    jax.random.randint(subkey, (), -(half_patch - 1), half_patch)
                )
            else:
                roll_rate = 0
            roll_quantities.append(roll_rate)
            roll_dims.append(i + dim_offset)
    x = jnp.concatenate([x, bc_flags], axis=2)
    if jitter_patches and len(roll_quantities) > 0:
        for rq, rd in zip(roll_quantities, roll_dims):
            x = jnp.roll(x, rq, axis=rd)
    if sum(periodic_pad) > 0:
        x_flat = rearrange(x, "t b c h w d -> (t b) c h w d")
        x_flat = _pad_nd(x_flat, periodic_pad, mode="circular")
        x = rearrange(x_flat, "(t b) c h w d -> t b c h w d", t=T)
    jitter_info = {
        "constant_paddings": const_pad,
        "periodic_paddings": periodic_pad,
        "rolls": (roll_quantities, roll_dims),
    }
    return x, jitter_info


def _unjitter(x, jitter_info, jitter_patches):
    const_pad = jitter_info["constant_paddings"]
    periodic_pad = jitter_info["periodic_paddings"]
    roll_quantities, roll_dims = jitter_info["rolls"]
    x = _slice_padding(x, periodic_pad, n_leading_dims=x.ndim - len(periodic_pad) // 2)
    if jitter_patches and len(roll_quantities) > 0:
        for rq, rd in zip(roll_quantities, roll_dims):
            x = jnp.roll(x, -rq, axis=rd)
    x = _slice_padding(x, const_pad, n_leading_dims=x.ndim - len(const_pad) // 2)
    return x


# ═══════════════════════════════════════════════════════════════════════════════
# RoPE helpers
# ═══════════════════════════════════════════════════════════════════════════════


def rotate_half_lr(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = jnp.split(x, 2, axis=-1)
    x1 = x1[..., 0]
    x2 = x2[..., 0]
    x = jnp.stack((-x2, x1), axis=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_emb(
    freqs, t, start_index=0, scale=1.0, seq_dim=-2, freqs_seq_dim=None
):
    if t.ndim == 3 or freqs_seq_dim is not None:
        if freqs_seq_dim is None:
            freqs_seq_dim = 0
        seq_len = t.shape[seq_dim]
        freqs = jax.lax.dynamic_slice_in_dim(
            freqs, freqs.shape[freqs_seq_dim] - seq_len, seq_len, axis=freqs_seq_dim
        )
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]
    t_transformed = (t_middle * jnp.cos(freqs) * scale) + (
        rotate_half_lr(t_middle) * jnp.sin(freqs) * scale
    )
    return jnp.concatenate((t_left, t_transformed, t_right), axis=-1)


def rotate_half_simple(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x[..., 0, :], x[..., 1, :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb_simple(pos, t):
    return (t * jnp.cos(pos)) + (rotate_half_simple(t) * jnp.sin(pos))


# ═══════════════════════════════════════════════════════════════════════════════
# T5-style relative position bias
# ═══════════════════════════════════════════════════════════════════════════════


def _relative_position_bucket(
    relative_position, bidirectional=True, num_buckets=32, max_distance=32
):
    ret = jnp.zeros_like(relative_position, dtype=jnp.int32)
    n = -relative_position
    if bidirectional:
        num_buckets //= 2
        ret = ret + (n < 0).astype(jnp.int32) * num_buckets
        n = jnp.abs(n)
    else:
        n = jnp.maximum(n, 0)
    max_exact = num_buckets // 2
    is_small = n < max_exact
    val_if_large = max_exact + (
        jnp.log(n.astype(jnp.float32) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(jnp.int32)
    val_if_large = jnp.minimum(val_if_large, num_buckets - 1)
    ret = ret + jnp.where(is_small, n, val_if_large)
    return ret


# ═══════════════════════════════════════════════════════════════════════════════
# Equinox modules
# ═══════════════════════════════════════════════════════════════════════════════


class RMSGroupNorm(eqx.Module):
    num_groups: int
    num_channels: int
    eps: float
    weight: jnp.ndarray

    def __init__(self, num_groups, num_channels, eps=1e-6, *, key=None):
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = jnp.ones((num_channels,))

    def __call__(self, x):
        spatial_dims = x.shape[2:]
        B = x.shape[0]
        grouped = x.reshape(B, self.num_groups, -1, *spatial_dims)
        norm_axes = tuple(range(3, grouped.ndim))
        rms = jnp.sqrt(jnp.mean(grouped**2, axis=norm_axes, keepdims=True) + self.eps)
        grouped = grouped / rms
        out = grouped.reshape(B, self.num_channels, *spatial_dims)
        indexing = (slice(None),) + (None,) * len(spatial_dims)
        return out * self.weight[indexing]


class LRRotaryEmbedding(eqx.Module):
    dim: int
    freqs_for: str
    max_freq: float
    freqs: jnp.ndarray

    def __init__(
        self,
        dim,
        freqs_for="pixel",
        max_freq=256.0,
        theta=10000.0,
        theta_rescale_factor=1.0,
        *,
        key=None,
    ):
        self.dim = dim
        self.freqs_for = freqs_for
        self.max_freq = max_freq
        theta_adj = (
            theta * theta_rescale_factor ** (dim / (dim - 2)) if dim > 2 else theta
        )
        if freqs_for == "pixel":
            self.freqs = jnp.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "lang":
            self.freqs = 1.0 / (
                theta_adj
                ** (jnp.arange(0, dim, 2)[: dim // 2].astype(jnp.float32) / dim)
            )
        elif freqs_for == "constant":
            self.freqs = jnp.ones(1)
        else:
            raise ValueError(f"Unknown freqs_for: {freqs_for}")

    def __call__(self, t):
        freqs = jnp.einsum("..., f -> ... f", t.astype(self.freqs.dtype), self.freqs)
        return jnp.repeat(freqs, 2, axis=-1)

    def get_axial_freqs(self, *dims):
        all_freqs = []
        for ind, dim_size in enumerate(dims):
            if self.freqs_for == "pixel":
                pos = jnp.linspace(-1.0, 1.0, dim_size)
            else:
                pos = jnp.arange(dim_size, dtype=jnp.float32)
            freqs = self(pos)
            shape = [1] * len(dims)
            shape[ind] = dim_size
            freqs = freqs.reshape(*shape, -1)
            all_freqs.append(freqs)
        all_freqs = jnp.broadcast_arrays(*all_freqs)
        return jnp.concatenate(all_freqs, axis=-1)


class SimpleRotaryEmbedding(eqx.Module):
    inv_freq: jnp.ndarray

    def __init__(self, dim, *, key=None):
        self.inv_freq = 1.0 / (
            10000 ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim)
        )

    def __call__(self, max_seq_len):
        seq = jnp.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = jnp.einsum("i , j -> i j", seq, self.inv_freq)
        return jnp.concatenate((freqs, freqs), axis=-1)


class RelativePositionBias(eqx.Module):
    bidirectional: bool
    num_buckets: int
    max_distance: int
    n_heads: int
    embedding: jnp.ndarray  # (num_buckets, n_heads)

    def __init__(
        self,
        bidirectional=True,
        num_buckets=32,
        max_distance=128,
        n_heads=2,
        *,
        key=None,
    ):
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        if key is None:
            key = jax.random.PRNGKey(0)
        self.embedding = jax.random.normal(key, (num_buckets, n_heads)) * 0.02

    def __call__(self, qlen, klen):
        context_position = jnp.arange(qlen)[:, None]
        memory_position = jnp.arange(klen)[None, :]
        relative_position = memory_position - context_position
        rp_bucket = _relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
        )
        values = self.embedding[rp_bucket]  # (qlen, klen, n_heads)
        values = jnp.transpose(values, (2, 0, 1))[None, ...]
        if not self.bidirectional:
            mask = relative_position > 0
            values = jnp.where(mask[None, None, :, :], float("-inf"), values)
        return values


# ── Encoder / Decoder ──


class AdaptiveDVstrideEncoder(eqx.Module):
    input_dim: int
    inner_dim: int
    output_dim: int
    base_kernel_size: tuple
    groups: int
    spatial_dims: int
    use_silu: bool
    proj1_weight: jnp.ndarray
    norm1: RMSGroupNorm
    proj2_weight: jnp.ndarray
    norm2: RMSGroupNorm

    def __init__(
        self,
        input_dim,
        inner_dim,
        output_dim,
        base_kernel_size,
        groups,
        spatial_dims=3,
        use_silu=True,
        *,
        key=None,
    ):
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.output_dim = output_dim
        self.base_kernel_size = base_kernel_size
        self.groups = groups
        self.spatial_dims = spatial_dims
        self.use_silu = use_silu
        bk1 = tuple(base_kernel_size[i][0] for i in range(spatial_dims))
        bk2 = tuple(base_kernel_size[i][1] for i in range(spatial_dims))
        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        self.proj1_weight = jax.random.normal(k1, (inner_dim, input_dim, *bk1)) * 0.01
        self.norm1 = RMSGroupNorm(groups, inner_dim)
        self.proj2_weight = jax.random.normal(k2, (output_dim, inner_dim, *bk2)) * 0.01
        self.norm2 = RMSGroupNorm(groups, output_dim)

    def __call__(self, x, stride1, stride2):
        bk1 = tuple(self.base_kernel_size[i][0] for i in range(self.spatial_dims))
        w1 = self.proj1_weight
        s1 = list(stride1)
        spatial_shape = x.shape[2:]
        for i, dim_size in enumerate(reversed(spatial_shape), start=1):
            if dim_size == 1:
                w1 = jnp.sum(w1, axis=-i, keepdims=True)
                s1[-i] = 1
        act = jax.nn.silu if self.use_silu else jax.nn.gelu
        x = _conv3d(x, w1, bias=None, stride=tuple(s1), padding=(0, 0, 0))
        x = self.norm1(x)
        x = act(x)
        w2 = self.proj2_weight
        s2 = list(stride2)
        spatial_shape2 = x.shape[2:]
        for i, dim_size in enumerate(reversed(spatial_shape2), start=1):
            if dim_size == 1:
                w2 = jnp.sum(w2, axis=-i, keepdims=True)
                s2[-i] = 1
        x = _conv3d(x, w2, bias=None, stride=tuple(s2), padding=(0, 0, 0))
        x = self.norm2(x)
        x = act(x)
        return x


class SpaceBagAdaptiveDVstrideEncoder(eqx.Module):
    input_dim: int
    inner_dim: int
    output_dim: int
    base_kernel_size: tuple
    groups: int
    spatial_dims: int
    extra_dims: int
    use_silu: bool
    proj1_weight: jnp.ndarray
    norm1: RMSGroupNorm
    proj2_weight: jnp.ndarray
    norm2: RMSGroupNorm

    def __init__(
        self,
        input_dim,
        inner_dim,
        output_dim,
        base_kernel_size,
        groups,
        spatial_dims=3,
        extra_dims=3,
        use_silu=True,
        *,
        key=None,
    ):
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.output_dim = output_dim
        self.base_kernel_size = base_kernel_size
        self.groups = groups
        self.spatial_dims = spatial_dims
        self.extra_dims = extra_dims
        self.use_silu = use_silu
        bk1 = tuple(base_kernel_size[i][0] for i in range(spatial_dims))
        bk2 = tuple(base_kernel_size[i][1] for i in range(spatial_dims))
        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        self.proj1_weight = jax.random.normal(k1, (inner_dim, input_dim, *bk1)) * 0.01
        self.norm1 = RMSGroupNorm(groups, inner_dim)
        self.proj2_weight = jax.random.normal(k2, (output_dim, inner_dim, *bk2)) * 0.01
        self.norm2 = RMSGroupNorm(groups, output_dim)

    def __call__(self, x, field_indices, stride1, stride2):
        w1 = self.proj1_weight[:, field_indices]
        scale_factor = (
            (self.proj1_weight.shape[1] - self.extra_dims)
            / (w1.shape[1] - self.extra_dims)
        ) ** 0.5
        w1 = jnp.concatenate([w1[:, :-2] * scale_factor, w1[:, -2:]], axis=1)
        s1 = list(stride1)
        spatial_shape = x.shape[2:]
        for i, dim_size in enumerate(reversed(spatial_shape), start=1):
            if dim_size == 1:
                w1 = jnp.sum(w1, axis=-i, keepdims=True)
                s1[-i] = 1
        act = jax.nn.silu if self.use_silu else jax.nn.gelu
        x = _conv3d(x, w1, bias=None, stride=tuple(s1), padding=(0, 0, 0))
        x = self.norm1(x)
        x = act(x)
        w2 = self.proj2_weight
        s2 = list(stride2)
        spatial_shape2 = x.shape[2:]
        for i, dim_size in enumerate(reversed(spatial_shape2), start=1):
            if dim_size == 1:
                w2 = jnp.sum(w2, axis=-i, keepdims=True)
                s2[-i] = 1
        x = _conv3d(x, w2, bias=None, stride=tuple(s2), padding=(0, 0, 0))
        x = self.norm2(x)
        x = act(x)
        return x


class AdaptiveDVstrideDecoder(eqx.Module):
    input_dim: int
    inner_dim: int
    output_dim: int
    base_kernel_size: tuple
    groups: int
    spatial_dims: int
    use_silu: bool
    proj1_weight: jnp.ndarray
    norm1: RMSGroupNorm
    proj2_weight: jnp.ndarray
    proj2_bias: jnp.ndarray

    def __init__(
        self,
        input_dim,
        inner_dim,
        output_dim,
        base_kernel_size,
        groups,
        spatial_dims=3,
        use_silu=True,
        *,
        key=None,
    ):
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.output_dim = output_dim
        self.base_kernel_size = base_kernel_size
        self.groups = groups
        self.spatial_dims = spatial_dims
        self.use_silu = use_silu
        # Decoder reverses kernel order
        bk1 = tuple(base_kernel_size[i][1] for i in range(spatial_dims))
        bk2 = tuple(base_kernel_size[i][0] for i in range(spatial_dims))
        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        self.proj1_weight = jax.random.normal(k1, (input_dim, inner_dim, *bk1)) * 0.01
        self.norm1 = RMSGroupNorm(groups, inner_dim)
        self.proj2_weight = jax.random.normal(k2, (inner_dim, output_dim, *bk2)) * 0.01
        self.proj2_bias = jnp.zeros((output_dim,))

    def __call__(self, x, state_labels, bcs, stride1, stride2):
        x = self._adaptive_conv_transpose(
            x,
            bcs,
            self.proj1_weight,
            bias=None,
            stride=stride1,
            padding=(0,) * self.spatial_dims,
        )
        act = jax.nn.silu if self.use_silu else jax.nn.gelu
        x = self.norm1(x)
        x = act(x)
        w2 = self.proj2_weight[:, state_labels]
        b2 = self.proj2_bias[state_labels]
        x = self._adaptive_conv_transpose(
            x, bcs, w2, bias=b2, stride=stride2, padding=(0,) * self.spatial_dims
        )
        return x

    def _adaptive_conv_transpose(self, x, bcs, weight, bias, stride, padding):
        spatial_shape = x.shape[2:]
        stride = list(stride)
        padding = list(padding)
        periodic_padding_list = []
        padding_out_list = []
        bcs_padded = (
            list(bcs) if len(bcs) >= self.spatial_dims else list(bcs) + [[2, 2]]
        )
        for i in range(1, self.spatial_dims + 1):
            dim_idx = self.spatial_dims - i
            dim_size = spatial_shape[dim_idx]
            weight_dim_idx = -(i)
            if dim_size == 1:
                weight = jnp.mean(weight, axis=weight_dim_idx, keepdims=True)
                stride[-i] = 1
                padding[-i] = 0
                periodic_padding_list.extend([0, 0])
                padding_out_list.extend([0, 0])
            else:
                k = weight.shape[weight_dim_idx]
                s = stride[-i]
                pad_in = (k - s) // s
                pad_out = k - s
                bc_val = int(bcs_padded[-i][0])
                if bc_val == BC_PERIODIC:
                    periodic_padding_list.extend([pad_in, pad_in])
                    padding_out_list.extend([-pad_out, -pad_out])
                else:
                    periodic_padding_list.extend([0, 0])
                    padding_out_list.extend([0, 0])
        periodic_padding_list = periodic_padding_list[::-1]
        padding_out_list = padding_out_list[::-1]
        if any(p > 0 for p in periodic_padding_list):
            pad_pairs = [(0, 0), (0, 0)]
            for ii in range(0, len(periodic_padding_list), 2):
                pad_pairs.append(
                    (periodic_padding_list[ii], periodic_padding_list[ii + 1])
                )
            x = jnp.pad(x, pad_pairs, mode="wrap")
        x = _conv_transpose3d(
            x, weight, bias=bias, stride=tuple(stride), padding=tuple(padding)
        )
        if any(p < 0 for p in padding_out_list):
            slices = [slice(None), slice(None)]
            for ii in range(0, len(padding_out_list), 2):
                left_crop = -padding_out_list[ii]
                right_crop = -padding_out_list[ii + 1]
                if left_crop > 0 or right_crop > 0:
                    dim_size = x.shape[2 + ii // 2]
                    slices.append(
                        slice(
                            left_crop if left_crop > 0 else None,
                            (dim_size - right_crop) if right_crop > 0 else None,
                        )
                    )
                else:
                    slices.append(slice(None))
            x = x[tuple(slices)]
        return x


# ── Attention modules ──


class SwiGLU(eqx.Module):
    def __call__(self, x):
        x, gate = jnp.split(x, 2, axis=-1)
        return jax.nn.silu(gate) * x


def _drop_path_fn(x, rate, deterministic, rng_key):
    if rate == 0.0 or deterministic:
        return x
    keep = 1.0 - rate
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = jax.random.bernoulli(rng_key, keep, shape=shape).astype(x.dtype)
    return x * mask / keep


class FullAttention(eqx.Module):
    hidden_dim: int
    mlp_dim: int
    num_heads: int
    drop_path: float
    norm1: RMSGroupNorm
    fused_ff_qkv: eqx.nn.Linear
    q_norm: eqx.nn.LayerNorm
    k_norm: eqx.nn.LayerNorm
    rotary_emb: LRRotaryEmbedding
    attn_out: eqx.nn.Linear
    ff_out: eqx.nn.Linear

    def __init__(self, hidden_dim=768, mlp_dim=0, num_heads=12, drop_path=0.0, *, key):
        self.hidden_dim = hidden_dim
        mlp_dim = mlp_dim if mlp_dim > 0 else hidden_dim * 4
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.drop_path = drop_path
        head_dim = hidden_dim // num_heads
        k1, k2, k3 = jax.random.split(key, 3)
        self.norm1 = RMSGroupNorm(num_heads, hidden_dim)
        total_dim = mlp_dim + hidden_dim * 3
        self.fused_ff_qkv = eqx.nn.Linear(hidden_dim, total_dim, key=k1)
        self.q_norm = eqx.nn.LayerNorm(head_dim)
        self.k_norm = eqx.nn.LayerNorm(head_dim)
        self.rotary_emb = LRRotaryEmbedding(
            dim=head_dim // 4, freqs_for="pixel", max_freq=256.0
        )
        self.attn_out = eqx.nn.Linear(hidden_dim, hidden_dim, use_bias=False, key=k2)
        self.ff_out = eqx.nn.Linear(mlp_dim // 2, hidden_dim, key=k3)

    def __call__(self, x, deterministic=True, *, key=None):
        B, C, H, W, D = x.shape
        residual = x
        x = self.norm1(x)
        x = rearrange(x, "b c h w d -> b h w d c")
        # Fused projection
        fused = jax.vmap(jax.vmap(jax.vmap(jax.vmap(self.fused_ff_qkv))))(x)
        ff, q, k, v = jnp.split(
            fused,
            [
                self.mlp_dim,
                self.mlp_dim + self.hidden_dim,
                self.mlp_dim + 2 * self.hidden_dim,
            ],
            axis=-1,
        )
        head_dim = self.hidden_dim // self.num_heads
        q = rearrange(q, "b h w d (he c) -> b he h w d c", he=self.num_heads)
        k = rearrange(k, "b h w d (he c) -> b he h w d c", he=self.num_heads)
        v = rearrange(v, "b h w d (he c) -> b he h w d c", he=self.num_heads)
        # QK norm
        q = jax.vmap(jax.vmap(jax.vmap(jax.vmap(jax.vmap(self.q_norm)))))(q)
        k = jax.vmap(jax.vmap(jax.vmap(jax.vmap(jax.vmap(self.k_norm)))))(k)
        # RoPE
        pos_emb = self.rotary_emb.get_axial_freqs(H, W, D)
        q = apply_rotary_emb(pos_emb, q)
        k = apply_rotary_emb(pos_emb, k)
        # Attention
        q = rearrange(q, "b he h w d c -> b he (h w d) c")
        k = rearrange(k, "b he h w d c -> b he (h w d) c")
        v = rearrange(v, "b he h w d c -> b he (h w d) c")
        scale = 1.0 / math.sqrt(head_dim)
        attn_weights = jnp.einsum("bhsc, bhtc -> bhst", q, k) * scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        att = jnp.einsum("bhst, bhtc -> bhsc", attn_weights, v)
        att = rearrange(att, "b he (h w d) c -> b h w d (he c)", h=H, w=W)
        # Output projections
        attn_out = jax.vmap(jax.vmap(jax.vmap(jax.vmap(self.attn_out))))(att)
        swiglu = SwiGLU()
        ff_activated = swiglu(ff)
        ff_out = jax.vmap(jax.vmap(jax.vmap(jax.vmap(self.ff_out))))(ff_activated)
        x = attn_out + ff_out
        if self.drop_path > 0.0 and not deterministic and key is not None:
            x = _drop_path_fn(x, self.drop_path, deterministic, key)
        x = rearrange(x, "b h w d c -> b c h w d") + residual
        return x, []


class AxialTimeAttention(eqx.Module):
    hidden_dim: int
    num_heads: int
    drop_path: float
    bias_type: str
    causal_in_time: bool
    norm1: RMSGroupNorm
    input_head_weight: jnp.ndarray
    input_head_bias: jnp.ndarray
    output_head_weight: jnp.ndarray
    output_head_bias: jnp.ndarray
    qnorm: eqx.nn.LayerNorm
    knorm: eqx.nn.LayerNorm
    rel_pos_bias: Optional[RelativePositionBias]
    rotary_emb: Optional[SimpleRotaryEmbedding]

    def __init__(
        self,
        hidden_dim=768,
        num_heads=12,
        drop_path=0.0,
        bias_type="rel",
        causal_in_time=False,
        *,
        key,
    ):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.drop_path = drop_path
        self.bias_type = bias_type
        self.causal_in_time = causal_in_time
        head_dim = hidden_dim // num_heads
        k1, k2 = jax.random.split(key)
        self.norm1 = RMSGroupNorm(num_heads, hidden_dim)
        self.input_head_weight = (
            jax.random.normal(k1, (3 * hidden_dim, hidden_dim, 1, 1, 1)) * 0.01
        )
        self.input_head_bias = jnp.zeros((3 * hidden_dim,))
        self.output_head_weight = (
            jax.random.normal(k2, (hidden_dim, hidden_dim, 1, 1, 1)) * 0.01
        )
        self.output_head_bias = jnp.zeros((hidden_dim,))
        self.qnorm = eqx.nn.LayerNorm(head_dim)
        self.knorm = eqx.nn.LayerNorm(head_dim)
        if bias_type == "rel":
            self.rel_pos_bias = RelativePositionBias(
                bidirectional=not causal_in_time, n_heads=num_heads
            )
            self.rotary_emb = None
        elif bias_type == "rotary":
            self.rel_pos_bias = None
            self.rotary_emb = SimpleRotaryEmbedding(dim=head_dim)
        else:
            self.rel_pos_bias = None
            self.rotary_emb = None

    def __call__(self, x, deterministic=True, *, key=None):
        T, B, C, H, W, D = x.shape
        residual = x
        x = rearrange(x, "t b c h w d -> (t b) c h w d")
        x = self.norm1(x)
        x = _conv3d_1x1(x, self.input_head_weight, self.input_head_bias)
        head_dim = self.hidden_dim // self.num_heads
        x = rearrange(
            x, "(t b) (he c) h w d -> (b h w d) he t c", t=T, he=self.num_heads
        )
        q, k, v = jnp.split(x, 3, axis=-1)
        # QK norm via vmap
        q = jax.vmap(jax.vmap(jax.vmap(self.qnorm)))(q)
        k = jax.vmap(jax.vmap(jax.vmap(self.knorm)))(k)
        # Position encoding
        rel_pos_bias_val = None
        if self.bias_type == "rotary" and self.rotary_emb is not None:
            positions = self.rotary_emb(T)
            q = apply_rotary_pos_emb_simple(positions, q)
            k = apply_rotary_pos_emb_simple(positions, k)
        elif self.bias_type == "rel" and self.rel_pos_bias is not None:
            rel_pos_bias_val = self.rel_pos_bias(T, T)
        # Attention
        scale = 1.0 / math.sqrt(head_dim)
        attn_weights = jnp.einsum("bhsc, bhtc -> bhst", q, k) * scale
        if rel_pos_bias_val is not None:
            attn_weights = attn_weights + rel_pos_bias_val
        if self.causal_in_time and self.bias_type != "rel":
            mask = jnp.triu(jnp.ones((T, T), dtype=jnp.bool_), k=1)
            attn_weights = jnp.where(
                mask[None, None, :, :], float("-inf"), attn_weights
            )
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        att_out = jnp.einsum("bhst, bhtc -> bhsc", attn_weights, v)
        att_out = rearrange(
            att_out, "(b h w d) he t c -> (t b) (he c) h w d", h=H, w=W, d=D
        )
        x = _conv3d_1x1(att_out, self.output_head_weight, self.output_head_bias)
        x = rearrange(x, "(t b) c h w d -> t b c h w d", t=T)
        if self.drop_path > 0.0 and not deterministic and key is not None:
            x = _drop_path_fn(x, self.drop_path, deterministic, key)
        return x + residual, []


class SpaceTimeSplitBlock(eqx.Module):
    hidden_dim: int
    num_heads: int
    drop_path: float
    time_mixing: AxialTimeAttention
    space_mixing: FullAttention

    def __init__(
        self,
        hidden_dim=768,
        num_heads=12,
        mlp_dim=0,
        drop_path=0.0,
        causal_in_time=False,
        bias_type="rel",
        *,
        key,
    ):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.drop_path = drop_path
        k1, k2 = jax.random.split(key)
        self.time_mixing = AxialTimeAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            drop_path=drop_path,
            bias_type=bias_type,
            causal_in_time=causal_in_time,
            key=k1,
        )
        self.space_mixing = FullAttention(
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            drop_path=drop_path,
            key=k2,
        )

    def __call__(self, x, bcs=None, deterministic=True, *, key=None):
        T, B, C, H, W, D = x.shape
        x, t_att = self.time_mixing(x, deterministic=deterministic, key=key)
        x = rearrange(x, "t b c h w d -> (t b) c h w d")
        x, s_att = self.space_mixing(x, deterministic=deterministic, key=key)
        x = rearrange(x, "(t b) c h w d -> t b c h w d", t=T)
        return x, t_att + s_att


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level IsotropicModel
# ═══════════════════════════════════════════════════════════════════════════════


class IsotropicModel(eqx.Module):
    hidden_dim: int
    intermediate_dim: int
    n_states: int
    processor_blocks_count: int
    groups: int
    num_heads: int
    mlp_dim: int
    max_d: int
    causal_in_time: bool
    drop_path: float
    bias_type: str
    base_kernel_size: tuple
    use_spacebag: bool
    use_silu: bool
    include_d: tuple
    encoder_groups: int
    learned_pad: bool

    encoder_dummy: jnp.ndarray
    encoders: dict  # dim_key -> encoder
    decoders: dict  # dim_key -> decoder
    blocks: list  # list of SpaceTimeSplitBlock

    def __init__(
        self,
        hidden_dim=768,
        intermediate_dim=192,
        n_states=4,
        processor_blocks=12,
        groups=16,
        num_heads=12,
        mlp_dim=0,
        max_d=3,
        causal_in_time=False,
        drop_path=0.05,
        bias_type="rel",
        base_kernel_size=((8, 4), (8, 4), (8, 4)),
        use_spacebag=True,
        use_silu=True,
        include_d=(2, 3),
        encoder_groups=16,
        learned_pad=True,
        *,
        key,
    ):
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.n_states = n_states
        self.processor_blocks_count = processor_blocks
        self.groups = groups
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_d = max_d
        self.causal_in_time = causal_in_time
        self.drop_path = drop_path
        self.bias_type = bias_type
        self.base_kernel_size = tuple(tuple(x) for x in base_kernel_size)
        self.use_spacebag = use_spacebag
        self.use_silu = use_silu
        self.include_d = include_d
        self.encoder_groups = encoder_groups
        self.learned_pad = learned_pad

        self.encoder_dummy = jnp.ones((1,))

        keys = jax.random.split(key, processor_blocks + 10)
        ki = 0

        # Encoders / decoders per dim variant
        encoders = {}
        decoders = {}
        for d in include_d:
            if use_spacebag:
                encoders[d] = SpaceBagAdaptiveDVstrideEncoder(
                    input_dim=n_states,
                    inner_dim=intermediate_dim,
                    output_dim=hidden_dim,
                    base_kernel_size=base_kernel_size,
                    groups=encoder_groups,
                    spatial_dims=max_d,
                    extra_dims=3,
                    use_silu=use_silu,
                    key=keys[ki],
                )
            else:
                encoders[d] = AdaptiveDVstrideEncoder(
                    input_dim=n_states,
                    inner_dim=intermediate_dim,
                    output_dim=hidden_dim,
                    base_kernel_size=base_kernel_size,
                    groups=encoder_groups,
                    spatial_dims=max_d,
                    use_silu=use_silu,
                    key=keys[ki],
                )
            ki += 1
            decoders[d] = AdaptiveDVstrideDecoder(
                input_dim=hidden_dim,
                inner_dim=intermediate_dim,
                output_dim=n_states,
                base_kernel_size=base_kernel_size,
                groups=encoder_groups,
                spatial_dims=max_d,
                use_silu=use_silu,
                key=keys[ki],
            )
            ki += 1
        self.encoders = encoders
        self.decoders = decoders

        # Processor blocks
        dp_rates = [
            i * drop_path / max(processor_blocks - 1, 1)
            for i in range(processor_blocks)
        ]
        self.blocks = [
            SpaceTimeSplitBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                drop_path=dp_rates[i],
                causal_in_time=causal_in_time,
                bias_type=bias_type,
                key=keys[ki + i],
            )
            for i in range(processor_blocks)
        ]

    def __call__(
        self,
        x: jnp.ndarray,
        state_labels: jnp.ndarray,
        bcs: list,
        stride1=None,
        stride2=None,
        field_indices=None,
        dim_key=None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        state_labels = jnp.asarray(state_labels, dtype=jnp.int32)
        if field_indices is not None:
            field_indices = jnp.asarray(field_indices, dtype=jnp.int32)

        # Convert (B, T, *spatial, C) -> (T, B, C, *spatial)
        n_spatial = x.ndim - 3
        x = jnp.moveaxis(x, -1, 2)
        x = jnp.swapaxes(x, 0, 1)

        squeeze_out = 0
        while x.ndim - 3 < self.max_d:
            x = jnp.expand_dims(x, axis=-1)
            squeeze_out += 1

        T, B, C = x.shape[:3]
        x_shape = x.shape[3:]

        if dim_key is None:
            dim_key = sum(int(s != 1) for s in x_shape)

        if stride1 is None or stride2 is None:
            dynamic_ks = choose_kernel_size_deterministic(x_shape)
            stride1 = stride1 or tuple(k[0] for k in dynamic_ks)
            stride2 = stride2 or tuple(k[1] for k in dynamic_ks)
            random_kernel = dynamic_ks
        else:
            random_kernel = tuple(zip(stride1, stride2))

        if self.use_spacebag and field_indices is None:
            field_indices = jnp.concatenate(
                [
                    state_labels,
                    jnp.array([2, 0, 1], dtype=state_labels.dtype),
                ]
            )

        x = x * self.encoder_dummy

        # Jitter / learned padding
        jitter_active = False  # deterministic mode only for testing
        should_jitter = self.learned_pad
        if should_jitter:
            bcs_flat = bcs[0] if isinstance(bcs, tuple) else bcs
            x, jitter_info = _jitter_forward(
                x,
                bcs=bcs_flat,
                n_dims=dim_key,
                max_d=self.max_d,
                base_kernel=self.base_kernel_size,
                random_kernel=random_kernel,
                jitter_patches=False,
                rng_key=None,
            )
        else:
            jitter_info = None

        # Encode
        x_flat = rearrange(x, "T B ... -> (T B) ...")
        encoder = self.encoders[dim_key]
        if self.use_spacebag and field_indices is not None:
            x_enc = encoder(x_flat, field_indices, stride1, stride2)
        else:
            x_enc = encoder(x_flat, stride1, stride2)
        x_enc = rearrange(x_enc, "(T B) ... -> T B ...", T=T)

        # Process
        x_proc = x_enc
        for i, block in enumerate(self.blocks):
            x_proc, _ = block(x_proc, bcs=bcs, deterministic=deterministic)

        # Non-causal: decode only last time step
        if not self.causal_in_time:
            x_proc = x_proc[-1:]

        # Decode
        T_out = x_proc.shape[0]
        bcs_flat = bcs[0] if isinstance(bcs, tuple) else bcs
        x_dec = rearrange(x_proc, "T B ... -> (T B) ...")
        decoder = self.decoders[dim_key]
        x_dec = decoder(x_dec, state_labels, bcs_flat, stride2, stride1)
        x_dec = rearrange(x_dec, "(T B) ... -> T B ...", T=T_out)

        # Unjitter
        if should_jitter and jitter_info is not None:
            x_dec = _unjitter(x_dec, jitter_info, False)

        for _ in range(squeeze_out):
            x_dec = x_dec[..., 0]

        # Convert (T, B, C, *spatial) -> (B, T, *spatial, C)
        x_dec = jnp.swapaxes(x_dec, 0, 1)
        x_dec = jnp.moveaxis(x_dec, 2, -1)
        return x_dec


# ═══════════════════════════════════════════════════════════════════════════════
# Weight transfer: Flax -> Equinox
# ═══════════════════════════════════════════════════════════════════════════════


def transfer_weights(flax_params: dict, eqx_model: IsotropicModel) -> IsotropicModel:
    """Transfer Flax parameters to an Equinox IsotropicModel."""
    p = flax_params["params"] if "params" in flax_params else flax_params

    def _get(path):
        keys = path.split(".")
        d = p
        for k in keys:
            d = d[k]
        return jnp.asarray(d)

    # encoder_dummy
    model = eqx.tree_at(lambda m: m.encoder_dummy, eqx_model, _get("encoder_dummy"))

    # Encoders
    for d in model.include_d:
        prefix = f"embed_{d}"
        if prefix not in p:
            continue
        enc = model.encoders[d]
        enc = eqx.tree_at(lambda e: e.proj1_weight, enc, _get(f"{prefix}.proj1_weight"))
        enc = eqx.tree_at(lambda e: e.norm1.weight, enc, _get(f"{prefix}.norm1.weight"))
        enc = eqx.tree_at(lambda e: e.proj2_weight, enc, _get(f"{prefix}.proj2_weight"))
        enc = eqx.tree_at(lambda e: e.norm2.weight, enc, _get(f"{prefix}.norm2.weight"))
        model = eqx.tree_at(lambda m: m.encoders[d], model, enc)

    # Decoders
    for d in model.include_d:
        prefix = f"debed_{d}"
        if prefix not in p:
            continue
        dec = model.decoders[d]
        dec = eqx.tree_at(lambda e: e.proj1_weight, dec, _get(f"{prefix}.proj1_weight"))
        dec = eqx.tree_at(lambda e: e.norm1.weight, dec, _get(f"{prefix}.norm1.weight"))
        dec = eqx.tree_at(lambda e: e.proj2_weight, dec, _get(f"{prefix}.proj2_weight"))
        dec = eqx.tree_at(lambda e: e.proj2_bias, dec, _get(f"{prefix}.proj2_bias"))
        model = eqx.tree_at(lambda m: m.decoders[d], model, dec)

    # Processor blocks
    for i in range(model.processor_blocks_count):
        blk_prefix = f"blocks_{i}"
        block = model.blocks[i]

        # Space mixing
        sm = f"{blk_prefix}.space_mixing"
        sa = block.space_mixing
        sa = eqx.tree_at(lambda s: s.norm1.weight, sa, _get(f"{sm}.norm1.weight"))
        # fused_ff_qkv: Flax kernel is (in, out), eqx Linear weight is (out, in)
        sa = eqx.tree_at(
            lambda s: s.fused_ff_qkv.weight, sa, _get(f"{sm}.fused_ff_qkv.kernel").T
        )
        sa = eqx.tree_at(
            lambda s: s.fused_ff_qkv.bias, sa, _get(f"{sm}.fused_ff_qkv.bias")
        )
        sa = eqx.tree_at(lambda s: s.q_norm.weight, sa, _get(f"{sm}.q_norm.scale"))
        sa = eqx.tree_at(lambda s: s.q_norm.bias, sa, _get(f"{sm}.q_norm.bias"))
        sa = eqx.tree_at(lambda s: s.k_norm.weight, sa, _get(f"{sm}.k_norm.scale"))
        sa = eqx.tree_at(lambda s: s.k_norm.bias, sa, _get(f"{sm}.k_norm.bias"))
        try:
            sa = eqx.tree_at(
                lambda s: s.rotary_emb.freqs, sa, _get(f"{sm}.rotary_emb.freqs")
            )
        except KeyError:
            pass
        sa = eqx.tree_at(
            lambda s: s.attn_out.weight, sa, _get(f"{sm}.attn_out.kernel").T
        )
        # ff_out: Flax kernel (in, out) -> eqx weight (out, in)
        sa = eqx.tree_at(lambda s: s.ff_out.weight, sa, _get(f"{sm}.ff_out.kernel").T)
        sa = eqx.tree_at(lambda s: s.ff_out.bias, sa, _get(f"{sm}.ff_out.bias"))
        block = eqx.tree_at(lambda b: b.space_mixing, block, sa)

        # Time mixing
        tm = f"{blk_prefix}.time_mixing"
        ta = block.time_mixing
        ta = eqx.tree_at(lambda t: t.norm1.weight, ta, _get(f"{tm}.norm1.weight"))
        ta = eqx.tree_at(
            lambda t: t.input_head_weight, ta, _get(f"{tm}.input_head_weight")
        )
        ta = eqx.tree_at(lambda t: t.input_head_bias, ta, _get(f"{tm}.input_head_bias"))
        ta = eqx.tree_at(
            lambda t: t.output_head_weight, ta, _get(f"{tm}.output_head_weight")
        )
        ta = eqx.tree_at(
            lambda t: t.output_head_bias, ta, _get(f"{tm}.output_head_bias")
        )
        ta = eqx.tree_at(lambda t: t.qnorm.weight, ta, _get(f"{tm}.qnorm.scale"))
        ta = eqx.tree_at(lambda t: t.qnorm.bias, ta, _get(f"{tm}.qnorm.bias"))
        ta = eqx.tree_at(lambda t: t.knorm.weight, ta, _get(f"{tm}.knorm.scale"))
        ta = eqx.tree_at(lambda t: t.knorm.bias, ta, _get(f"{tm}.knorm.bias"))
        if ta.rel_pos_bias is not None:
            try:
                ta = eqx.tree_at(
                    lambda t: t.rel_pos_bias.embedding,
                    ta,
                    _get(f"{tm}.rel_pos_bias.relative_attention_bias.embedding"),
                )
            except KeyError:
                pass
        if ta.rotary_emb is not None:
            try:
                ta = eqx.tree_at(
                    lambda t: t.rotary_emb.inv_freq,
                    ta,
                    _get(f"{tm}.rotary_emb.inv_freq"),
                )
            except KeyError:
                pass
        block = eqx.tree_at(lambda b: b.time_mixing, block, ta)

        model = eqx.tree_at(lambda m: m.blocks[i], model, block)

    return model
