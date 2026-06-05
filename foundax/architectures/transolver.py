"""
Transolver — Physics-Attention transformer for PDEs (Wu et al. 2024, ICML)
==========================================================================

Paper: https://arxiv.org/abs/2402.02366
Code:  https://github.com/thuml/Transolver  (PyTorch reference)

Physics-Attention replaces standard self-attention's O(N²) point-to-point
cost with a three-stage operation:

  (1) **Slice**: softmax-assign each of N tokens to one of M learnable
      "physical slices" per head, then form slice tokens as the weighted
      average of token features.
  (2) **Attention**: standard scaled dot-product attention across the M
      slice tokens (cheap, since M ≪ N).
  (3) **Deslice**: pour the attended slice tokens back to the original
      N tokens using the same assignment weights.

Total cost is O(N·M·D + M²·D), making it scalable to large meshes.

Three Physics-Attention variants are provided:
- ``PhysicsAttentionIrregular``     : unstructured point clouds, ``Linear`` projections.
- ``PhysicsAttentionStructured2D``  : structured 2-D grids, ``Conv2d`` projections.
- ``PhysicsAttentionStructured3D``  : structured 3-D grids, ``Conv3d`` projections.

The Conv-based projections in the structured variants act as a local
context aggregator before slicing.

All modules follow the channel-last, unbatched foundax convention:
  ``(N, C)``       — point cloud
  ``(H, W, C)``    — structured 2-D grid
  ``(D, H, W, C)`` — structured 3-D grid

Batch with ``jax.vmap`` externally.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from .common import Conv2d, get_activation
from .linear import Linear
from .time_embed import SinusoidalTimeEmbedding
from .unet import Conv3dNHWC


# ── helpers ─────────────────────────────────────────────────────────────────


def _orthogonal_matrix(key: jax.Array, shape: Tuple[int, int]) -> jnp.ndarray:
    """Orthogonal initialisation matching ``torch.nn.init.orthogonal_``."""
    a = jax.random.normal(key, shape)
    q, r = jnp.linalg.qr(a if shape[0] >= shape[1] else a.T)
    d = jnp.sign(jnp.diag(r))
    q = q * d
    return q if shape[0] >= shape[1] else q.T


def _vmap_layer_norm(ln: eqx.nn.LayerNorm, x: jnp.ndarray) -> jnp.ndarray:
    """Apply LayerNorm to the last axis, vmap'd over all leading axes."""
    leading = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    out = jax.vmap(ln)(flat)
    return out.reshape(*leading, x.shape[-1])


# ── feed-forward ────────────────────────────────────────────────────────────


class TransolverFFN(eqx.Module):
    """Two-layer FFN with activation: ``Linear → act → Linear``.

    Matches the paper's MLP block (``n_layers=0``, ``res=False``).
    """

    pre: Linear
    post: Linear
    act: Callable = eqx.field(static=True)

    def __init__(self, dim: int, mlp_ratio: int = 4, act: str = "gelu", *, key):
        k1, k2 = jax.random.split(key)
        self.pre = Linear(dim, dim * mlp_ratio, key=k1)
        self.post = Linear(dim * mlp_ratio, dim, key=k2)
        self.act = get_activation(act)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.post(self.act(self.pre(x)))


# ── physics attention variants ──────────────────────────────────────────────


def _physics_attention_core(
    fx_mid: jnp.ndarray,
    x_mid: jnp.ndarray,
    in_project_slice: Linear,
    temperature: jnp.ndarray,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    dropout_rate: float,
    key,
) -> jnp.ndarray:
    """Stages (1)–(3) shared by all three Physics-Attention variants.

    Inputs ``fx_mid`` and ``x_mid`` are both ``(H, N, D)`` per-head views
    after the variant-specific projection. Returns ``(N, H*D)`` which the
    caller passes through ``to_out``.
    """
    slice_logits = in_project_slice(x_mid) / temperature  # (H, N, G)
    slice_weights = jax.nn.softmax(slice_logits, axis=-1)
    slice_norm = jnp.sum(slice_weights, axis=1)  # (H, G)
    slice_token = jnp.einsum("hnd,hng->hgd", fx_mid, slice_weights)
    slice_token = slice_token / (slice_norm[..., None] + 1e-5)

    q = to_q(slice_token)
    k_ = to_k(slice_token)
    v = to_v(slice_token)
    D = q.shape[-1]
    scale = 1.0 / jnp.sqrt(jnp.array(D, dtype=q.dtype))
    attn = jax.nn.softmax(jnp.einsum("hgd,hjd->hgj", q, k_) * scale, axis=-1)
    if dropout_rate > 0 and key is not None:
        k_attn, key = jax.random.split(key)
        attn = eqx.nn.Dropout(p=dropout_rate)(attn, key=k_attn)
    out_slice = jnp.einsum("hgj,hjd->hgd", attn, v)

    out = jnp.einsum("hgd,hng->hnd", out_slice, slice_weights)
    N = out.shape[1]
    out = out.transpose(1, 0, 2).reshape(N, -1)  # (N, H*D)
    out = to_out(out)
    if dropout_rate > 0 and key is not None:
        out = eqx.nn.Dropout(p=dropout_rate)(out, key=key)
    return out


class PhysicsAttentionIrregular(eqx.Module):
    """Physics-Attention for unstructured point clouds.

    Input / output shape ``(N, dim)``.

    The temperature parameter is initialised to ``0.5`` per head; gradients
    can drive it to very small values, which is why the structured
    variants clamp it (matching the reference).
    """

    in_project_x: Linear
    in_project_fx: Linear
    in_project_slice: Linear
    to_q: Linear
    to_k: Linear
    to_v: Linear
    to_out: Linear
    temperature: jnp.ndarray
    num_heads: int = eqx.field(static=True)
    dim_head: int = eqx.field(static=True)
    slice_num: int = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        slice_num: int = 64,
        dropout: float = 0.0,
        *,
        key,
    ):
        keys = jax.random.split(key, 8)
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.slice_num = slice_num
        self.dropout_rate = dropout
        self.temperature = jnp.full((num_heads, 1, 1), 0.5)

        self.in_project_x = Linear(dim, inner_dim, key=keys[0])
        self.in_project_fx = Linear(dim, inner_dim, key=keys[1])

        slice_proj = Linear(dim_head, slice_num, key=keys[2])
        slice_proj = eqx.tree_at(
            lambda m: m.weight,
            slice_proj,
            _orthogonal_matrix(keys[3], (slice_num, dim_head)),
        )
        self.in_project_slice = slice_proj

        self.to_q = Linear(dim_head, dim_head, use_bias=False, key=keys[4])
        self.to_k = Linear(dim_head, dim_head, use_bias=False, key=keys[5])
        self.to_v = Linear(dim_head, dim_head, use_bias=False, key=keys[6])
        self.to_out = Linear(inner_dim, dim, key=keys[7])

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        N, _ = x.shape
        H, D = self.num_heads, self.dim_head
        fx_mid = self.in_project_fx(x).reshape(N, H, D).transpose(1, 0, 2)
        x_mid = self.in_project_x(x).reshape(N, H, D).transpose(1, 0, 2)
        return _physics_attention_core(
            fx_mid,
            x_mid,
            self.in_project_slice,
            self.temperature,
            self.to_q,
            self.to_k,
            self.to_v,
            self.to_out,
            self.dropout_rate,
            key,
        )


class PhysicsAttentionStructured2D(eqx.Module):
    """Physics-Attention for structured 2-D grids.

    Input / output shape ``(H, W, dim)``. Uses 3×3 ``Conv2d`` projections
    so the slicing has local spatial context.
    """

    in_project_x: Conv2d
    in_project_fx: Conv2d
    in_project_slice: Linear
    to_q: Linear
    to_k: Linear
    to_v: Linear
    to_out: Linear
    temperature: jnp.ndarray
    num_heads: int = eqx.field(static=True)
    dim_head: int = eqx.field(static=True)
    slice_num: int = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        slice_num: int = 64,
        dropout: float = 0.0,
        kernel: int = 3,
        *,
        key,
    ):
        keys = jax.random.split(key, 8)
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.slice_num = slice_num
        self.dropout_rate = dropout
        self.temperature = jnp.full((num_heads, 1, 1), 0.5)

        self.in_project_x = Conv2d(dim, inner_dim, kernel, padding="SAME", key=keys[0])
        self.in_project_fx = Conv2d(dim, inner_dim, kernel, padding="SAME", key=keys[1])

        slice_proj = Linear(dim_head, slice_num, key=keys[2])
        slice_proj = eqx.tree_at(
            lambda m: m.weight,
            slice_proj,
            _orthogonal_matrix(keys[3], (slice_num, dim_head)),
        )
        self.in_project_slice = slice_proj

        self.to_q = Linear(dim_head, dim_head, use_bias=False, key=keys[4])
        self.to_k = Linear(dim_head, dim_head, use_bias=False, key=keys[5])
        self.to_v = Linear(dim_head, dim_head, use_bias=False, key=keys[6])
        self.to_out = Linear(inner_dim, dim, key=keys[7])

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        H_grid, W_grid, _ = x.shape
        N = H_grid * W_grid
        H, D = self.num_heads, self.dim_head
        temperature = jnp.clip(self.temperature, 0.1, 5.0)

        fx_mid = self.in_project_fx(x).reshape(N, H, D).transpose(1, 0, 2)
        x_mid = self.in_project_x(x).reshape(N, H, D).transpose(1, 0, 2)
        out_flat = _physics_attention_core(
            fx_mid,
            x_mid,
            self.in_project_slice,
            temperature,
            self.to_q,
            self.to_k,
            self.to_v,
            self.to_out,
            self.dropout_rate,
            key,
        )
        return out_flat.reshape(H_grid, W_grid, -1)


class PhysicsAttentionStructured3D(eqx.Module):
    """Physics-Attention for structured 3-D grids.

    Input / output shape ``(D, H, W, dim)``. Uses 3×3×3 ``Conv3d``
    projections so the slicing has local spatial context.
    """

    in_project_x: Conv3dNHWC
    in_project_fx: Conv3dNHWC
    in_project_slice: Linear
    to_q: Linear
    to_k: Linear
    to_v: Linear
    to_out: Linear
    temperature: jnp.ndarray
    num_heads: int = eqx.field(static=True)
    dim_head: int = eqx.field(static=True)
    slice_num: int = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        slice_num: int = 32,
        dropout: float = 0.0,
        kernel: int = 3,
        *,
        key,
    ):
        keys = jax.random.split(key, 8)
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.slice_num = slice_num
        self.dropout_rate = dropout
        self.temperature = jnp.full((num_heads, 1, 1), 0.5)

        self.in_project_x = Conv3dNHWC(dim, inner_dim, kernel, padding="SAME", key=keys[0])
        self.in_project_fx = Conv3dNHWC(dim, inner_dim, kernel, padding="SAME", key=keys[1])

        slice_proj = Linear(dim_head, slice_num, key=keys[2])
        slice_proj = eqx.tree_at(
            lambda m: m.weight,
            slice_proj,
            _orthogonal_matrix(keys[3], (slice_num, dim_head)),
        )
        self.in_project_slice = slice_proj

        self.to_q = Linear(dim_head, dim_head, use_bias=False, key=keys[4])
        self.to_k = Linear(dim_head, dim_head, use_bias=False, key=keys[5])
        self.to_v = Linear(dim_head, dim_head, use_bias=False, key=keys[6])
        self.to_out = Linear(inner_dim, dim, key=keys[7])

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        D_grid, H_grid, W_grid, _ = x.shape
        N = D_grid * H_grid * W_grid
        H, D = self.num_heads, self.dim_head
        temperature = jnp.clip(self.temperature, 0.1, 5.0)

        fx_mid = self.in_project_fx(x).reshape(N, H, D).transpose(1, 0, 2)
        x_mid = self.in_project_x(x).reshape(N, H, D).transpose(1, 0, 2)
        out_flat = _physics_attention_core(
            fx_mid,
            x_mid,
            self.in_project_slice,
            temperature,
            self.to_q,
            self.to_k,
            self.to_v,
            self.to_out,
            self.dropout_rate,
            key,
        )
        return out_flat.reshape(D_grid, H_grid, W_grid, -1)


# ── transolver block ────────────────────────────────────────────────────────


class TransolverBlock(eqx.Module):
    """Pre-norm Transolver encoder block:

        x ← x + PhysicsAttention(LN(x))
        x ← x + FFN(LN(x))

    Pass any ``PhysicsAttention*`` instance as ``physics_attn``. The block
    is shape-agnostic — the underlying physics-attention determines whether
    inputs are ``(N, C)``, ``(H, W, C)`` or ``(D, H, W, C)``.
    """

    ln_1: eqx.nn.LayerNorm
    ln_2: eqx.nn.LayerNorm
    physics_attn: eqx.Module
    ffn: TransolverFFN

    def __init__(
        self,
        dim: int,
        physics_attn: eqx.Module,
        mlp_ratio: int = 4,
        act: str = "gelu",
        *,
        key,
    ):
        self.ln_1 = eqx.nn.LayerNorm(dim)
        self.ln_2 = eqx.nn.LayerNorm(dim)
        self.physics_attn = physics_attn
        self.ffn = TransolverFFN(dim, mlp_ratio=mlp_ratio, act=act, key=key)

    def __call__(self, x: jnp.ndarray, *, key=None) -> jnp.ndarray:
        x = x + self.physics_attn(_vmap_layer_norm(self.ln_1, x), key=key)
        x = x + self.ffn(_vmap_layer_norm(self.ln_2, x))
        return x


# ── top-level Transolver models ─────────────────────────────────────────────


def _make_lift(in_features: int, hidden_dim: int, act: str, *, key) -> "_Lift":
    k1, k2 = jax.random.split(key)
    return _Lift(
        Linear(in_features, hidden_dim * 2, key=k1),
        Linear(hidden_dim * 2, hidden_dim, key=k2),
        get_activation(act),
    )


class _Lift(eqx.Module):
    """Two-layer GELU MLP used as the input lifter (matches reference)."""

    pre: Linear
    post: Linear
    act: Callable = eqx.field(static=True)

    def __init__(self, pre, post, act):
        self.pre = pre
        self.post = post
        self.act = act

    def __call__(self, x):
        return self.post(self.act(self.pre(x)))


class TransolverIrregular(eqx.Module):
    """Transolver for unstructured 1-D / 2-D / 3-D point clouds.

    Inputs:
        x_coords : ``(N, space_dim)`` query-point coordinates.
        x_func   : ``(N, fun_dim)`` per-point input function values (or ``None``).
        t        : optional scalar timestep for time-conditioning.

    Output:
        ``(N, out_features)``.
    """

    lift: _Lift
    blocks: list
    head_ln: eqx.nn.LayerNorm
    head: Linear
    placeholder: jnp.ndarray
    time_embed: Optional[SinusoidalTimeEmbedding]
    fun_dim: int = eqx.field(static=True)
    space_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)

    def __init__(
        self,
        space_dim: int,
        fun_dim: int,
        out_features: int,
        hidden_dim: int = 128,
        n_layers: int = 8,
        n_heads: int = 8,
        n_slices: int = 64,
        mlp_ratio: int = 4,
        act: str = "gelu",
        dropout: float = 0.0,
        time_input: bool = False,
        *,
        key,
    ):
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
            )
        keys = jax.random.split(key, 5)
        self.fun_dim = fun_dim
        self.space_dim = space_dim
        self.hidden_dim = hidden_dim

        self.lift = _make_lift(space_dim + fun_dim, hidden_dim, act, key=keys[0])

        dim_head = hidden_dim // n_heads
        attn_keys = jax.random.split(keys[1], n_layers)
        block_keys = jax.random.split(keys[2], n_layers)
        self.blocks = [
            TransolverBlock(
                hidden_dim,
                PhysicsAttentionIrregular(
                    hidden_dim,
                    num_heads=n_heads,
                    dim_head=dim_head,
                    slice_num=n_slices,
                    dropout=dropout,
                    key=attn_keys[i],
                ),
                mlp_ratio=mlp_ratio,
                act=act,
                key=block_keys[i],
            )
            for i in range(n_layers)
        ]
        self.head_ln = eqx.nn.LayerNorm(hidden_dim)
        self.head = Linear(hidden_dim, out_features, key=keys[3])
        # Per-channel placeholder bias, added when no input function is given.
        self.placeholder = jax.random.uniform(
            keys[-1], (hidden_dim,), minval=0.0, maxval=1.0 / hidden_dim
        )
        self.time_embed = (
            SinusoidalTimeEmbedding(hidden_dim, key=keys[0])
            if time_input
            else None
        )

    def __call__(
        self,
        x_coords: jnp.ndarray,
        x_func: Optional[jnp.ndarray] = None,
        t: Optional[jnp.ndarray] = None,
        *,
        key=None,
    ) -> jnp.ndarray:
        if x_func is not None:
            fx = self.lift(jnp.concatenate([x_coords, x_func], axis=-1))
        else:
            if self.fun_dim > 0:
                raise ValueError(
                    f"x_func is None but model was built with fun_dim={self.fun_dim}; "
                    "either pass x_func of shape (..., fun_dim) or build the model with fun_dim=0."
                )
            fx = self.lift(x_coords)
        # The Irregular variant in the reference always adds the placeholder
        # bias, regardless of whether x_func was supplied; the structured
        # variants only add it when x_func is None.
        fx = fx + self.placeholder

        if t is not None:
            if self.time_embed is None:
                raise ValueError("Model was built with time_input=False; got t.")
            fx = fx + self.time_embed(t)

        if key is not None:
            block_keys = jax.random.split(key, len(self.blocks))
        else:
            block_keys = [None] * len(self.blocks)
        for block, bk in zip(self.blocks, block_keys):
            fx = block(fx, key=bk)
        return self.head(_vmap_layer_norm(self.head_ln, fx))


class TransolverStructured2D(eqx.Module):
    """Transolver for structured 2-D grids.

    Inputs:
        x_coords : ``(H, W, space_dim)`` per-pixel coordinates.
        x_func   : ``(H, W, fun_dim)`` per-pixel input function (or ``None``).
        t        : optional scalar timestep.

    Output:
        ``(H, W, out_features)``.
    """

    lift: _Lift
    blocks: list
    head_ln: eqx.nn.LayerNorm
    head: Linear
    placeholder: jnp.ndarray
    time_embed: Optional[SinusoidalTimeEmbedding]
    fun_dim: int = eqx.field(static=True)
    space_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)

    def __init__(
        self,
        space_dim: int,
        fun_dim: int,
        out_features: int,
        hidden_dim: int = 128,
        n_layers: int = 8,
        n_heads: int = 8,
        n_slices: int = 64,
        mlp_ratio: int = 4,
        act: str = "gelu",
        dropout: float = 0.0,
        kernel: int = 3,
        time_input: bool = False,
        *,
        key,
    ):
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
            )
        keys = jax.random.split(key, 5)
        self.fun_dim = fun_dim
        self.space_dim = space_dim
        self.hidden_dim = hidden_dim

        self.lift = _make_lift(space_dim + fun_dim, hidden_dim, act, key=keys[0])

        dim_head = hidden_dim // n_heads
        attn_keys = jax.random.split(keys[1], n_layers)
        block_keys = jax.random.split(keys[2], n_layers)
        self.blocks = [
            TransolverBlock(
                hidden_dim,
                PhysicsAttentionStructured2D(
                    hidden_dim,
                    num_heads=n_heads,
                    dim_head=dim_head,
                    slice_num=n_slices,
                    dropout=dropout,
                    kernel=kernel,
                    key=attn_keys[i],
                ),
                mlp_ratio=mlp_ratio,
                act=act,
                key=block_keys[i],
            )
            for i in range(n_layers)
        ]
        self.head_ln = eqx.nn.LayerNorm(hidden_dim)
        self.head = Linear(hidden_dim, out_features, key=keys[3])
        ph_key, t_key = jax.random.split(keys[4])
        self.placeholder = jax.random.uniform(
            ph_key, (hidden_dim,), minval=0.0, maxval=1.0 / hidden_dim
        )
        self.time_embed = (
            SinusoidalTimeEmbedding(hidden_dim, key=t_key) if time_input else None
        )

    def __call__(
        self,
        x_coords: jnp.ndarray,
        x_func: Optional[jnp.ndarray] = None,
        t: Optional[jnp.ndarray] = None,
        *,
        key=None,
    ) -> jnp.ndarray:
        if x_func is not None:
            fx = self.lift(jnp.concatenate([x_coords, x_func], axis=-1))
        else:
            if self.fun_dim > 0:
                raise ValueError(
                    f"x_func is None but model was built with fun_dim={self.fun_dim}; "
                    "either pass x_func of shape (..., fun_dim) or build the model with fun_dim=0."
                )
            fx = self.lift(x_coords) + self.placeholder

        if t is not None:
            if self.time_embed is None:
                raise ValueError("Model was built with time_input=False; got t.")
            fx = fx + self.time_embed(t)

        if key is not None:
            block_keys = jax.random.split(key, len(self.blocks))
        else:
            block_keys = [None] * len(self.blocks)
        for block, bk in zip(self.blocks, block_keys):
            fx = block(fx, key=bk)
        return self.head(_vmap_layer_norm(self.head_ln, fx))


class TransolverStructured3D(eqx.Module):
    """Transolver for structured 3-D grids.

    Inputs:
        x_coords : ``(D, H, W, space_dim)`` per-voxel coordinates.
        x_func   : ``(D, H, W, fun_dim)`` per-voxel input function (or ``None``).
        t        : optional scalar timestep.

    Output:
        ``(D, H, W, out_features)``.
    """

    lift: _Lift
    blocks: list
    head_ln: eqx.nn.LayerNorm
    head: Linear
    placeholder: jnp.ndarray
    time_embed: Optional[SinusoidalTimeEmbedding]
    fun_dim: int = eqx.field(static=True)
    space_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)

    def __init__(
        self,
        space_dim: int,
        fun_dim: int,
        out_features: int,
        hidden_dim: int = 128,
        n_layers: int = 8,
        n_heads: int = 8,
        n_slices: int = 32,
        mlp_ratio: int = 4,
        act: str = "gelu",
        dropout: float = 0.0,
        kernel: int = 3,
        time_input: bool = False,
        *,
        key,
    ):
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
            )
        keys = jax.random.split(key, 5)
        self.fun_dim = fun_dim
        self.space_dim = space_dim
        self.hidden_dim = hidden_dim

        self.lift = _make_lift(space_dim + fun_dim, hidden_dim, act, key=keys[0])

        dim_head = hidden_dim // n_heads
        attn_keys = jax.random.split(keys[1], n_layers)
        block_keys = jax.random.split(keys[2], n_layers)
        self.blocks = [
            TransolverBlock(
                hidden_dim,
                PhysicsAttentionStructured3D(
                    hidden_dim,
                    num_heads=n_heads,
                    dim_head=dim_head,
                    slice_num=n_slices,
                    dropout=dropout,
                    kernel=kernel,
                    key=attn_keys[i],
                ),
                mlp_ratio=mlp_ratio,
                act=act,
                key=block_keys[i],
            )
            for i in range(n_layers)
        ]
        self.head_ln = eqx.nn.LayerNorm(hidden_dim)
        self.head = Linear(hidden_dim, out_features, key=keys[3])
        ph_key, t_key = jax.random.split(keys[4])
        self.placeholder = jax.random.uniform(
            ph_key, (hidden_dim,), minval=0.0, maxval=1.0 / hidden_dim
        )
        self.time_embed = (
            SinusoidalTimeEmbedding(hidden_dim, key=t_key) if time_input else None
        )

    def __call__(
        self,
        x_coords: jnp.ndarray,
        x_func: Optional[jnp.ndarray] = None,
        t: Optional[jnp.ndarray] = None,
        *,
        key=None,
    ) -> jnp.ndarray:
        if x_func is not None:
            fx = self.lift(jnp.concatenate([x_coords, x_func], axis=-1))
        else:
            if self.fun_dim > 0:
                raise ValueError(
                    f"x_func is None but model was built with fun_dim={self.fun_dim}; "
                    "either pass x_func of shape (..., fun_dim) or build the model with fun_dim=0."
                )
            fx = self.lift(x_coords) + self.placeholder

        if t is not None:
            if self.time_embed is None:
                raise ValueError("Model was built with time_input=False; got t.")
            fx = fx + self.time_embed(t)

        if key is not None:
            block_keys = jax.random.split(key, len(self.blocks))
        else:
            block_keys = [None] * len(self.blocks)
        for block, bk in zip(self.blocks, block_keys):
            fx = block(fx, key=bk)
        return self.head(_vmap_layer_norm(self.head_ln, fx))


__all__ = [
    "PhysicsAttentionIrregular",
    "PhysicsAttentionStructured2D",
    "PhysicsAttentionStructured3D",
    "TransolverBlock",
    "TransolverFFN",
    "TransolverIrregular",
    "TransolverStructured2D",
    "TransolverStructured3D",
]
