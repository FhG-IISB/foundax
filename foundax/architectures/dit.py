# Diffusion Transformer backbone for flow matching on PDE fields.
#
# Paper: "Scalable Diffusion Models with Transformers"
#        Peebles & Xie (2022) — https://arxiv.org/abs/2212.09748
# Code:  https://github.com/facebookresearch/DiT  (PyTorch reference)
#
# Differences from the original:
#   • channel-last, unbatched tensors  (H, W, C) and (D, H, W, C)
#   • accepts scalar t and embeds it internally via SinusoidalTimeEmbedding
#   • **kwargs forwarded through __call__ for pipe compatibility

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from .linear import Linear
from .time_embed import SinusoidalTimeEmbedding
from .transformer import SelfAttention


# ── helpers ──────────────────────────────────────────────────────────────────


def _sincos_pos_embed_2d(H_p: int, W_p: int, d: int) -> jnp.ndarray:
    """Fixed 2-D sin/cos positional embedding, shape (H_p*W_p, d)."""
    assert d % 4 == 0, "hidden_size must be divisible by 4 for 2-D pos embed"
    half = d // 4
    freqs = 1.0 / (10000.0 ** (jnp.arange(half) / half))
    h_idx = jnp.arange(H_p)
    w_idx = jnp.arange(W_p)
    h_enc = jnp.concatenate(
        [
            jnp.sin(h_idx[:, None] * freqs[None, :]),
            jnp.cos(h_idx[:, None] * freqs[None, :]),
        ],
        axis=1,
    )  # (H_p, d//2)
    w_enc = jnp.concatenate(
        [
            jnp.sin(w_idx[:, None] * freqs[None, :]),
            jnp.cos(w_idx[:, None] * freqs[None, :]),
        ],
        axis=1,
    )  # (W_p, d//2)
    # Broadcast to 2-D grid
    h_grid = jnp.tile(h_enc[:, None, :], (1, W_p, 1))  # (H_p, W_p, d//2)
    w_grid = jnp.tile(w_enc[None, :, :], (H_p, 1, 1))  # (H_p, W_p, d//2)
    return jnp.concatenate([h_grid, w_grid], axis=-1).reshape(H_p * W_p, d)


def _sincos_pos_embed_3d(D_p: int, H_p: int, W_p: int, d: int) -> jnp.ndarray:
    """Fixed 3-D sin/cos positional embedding, shape (D_p*H_p*W_p, d)."""
    assert d % 6 == 0, "hidden_size must be divisible by 6 for 3-D pos embed"
    third = d // 6
    freqs = 1.0 / (10000.0 ** (jnp.arange(third) / third))

    def _enc(idx):
        return jnp.concatenate(
            [
                jnp.sin(idx[:, None] * freqs[None, :]),
                jnp.cos(idx[:, None] * freqs[None, :]),
            ],
            axis=1,
        )  # (N, d//3)

    d_enc = _enc(jnp.arange(D_p))  # (D_p, d//3)
    h_enc = _enc(jnp.arange(H_p))  # (H_p, d//3)
    w_enc = _enc(jnp.arange(W_p))  # (W_p, d//3)
    d_grid = jnp.tile(d_enc[:, None, None, :], (1, H_p, W_p, 1))
    h_grid = jnp.tile(h_enc[None, :, None, :], (D_p, 1, W_p, 1))
    w_grid = jnp.tile(w_enc[None, None, :, :], (D_p, H_p, 1, 1))
    return jnp.concatenate([d_grid, h_grid, w_grid], axis=-1).reshape(
        D_p * H_p * W_p, d
    )


# ── patch embedding ───────────────────────────────────────────────────────────


class PatchEmbed2d(eqx.Module):
    """Flatten 2-D spatial patches and project to hidden_size.

    (H, W, C) → (H//p * W//p, hidden_size)
    """

    patch_size: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    proj: Linear

    def __init__(self, patch_size: int, in_channels: int, hidden_size: int, *, key):
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.proj = Linear(patch_size * patch_size * in_channels, hidden_size, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        H, W, C = x.shape
        p = self.patch_size
        # (H, W, C) → (H//p, p, W//p, p, C) → (N, p*p*C)
        x = x.reshape(H // p, p, W // p, p, C).transpose(0, 2, 1, 3, 4)
        x = x.reshape(H // p * W // p, p * p * C)
        return self.proj(x)  # (N, hidden_size)


class PatchEmbed3d(eqx.Module):
    """Flatten 3-D cubic patches and project to hidden_size.

    (D, H, W, C) → (D//p * H//p * W//p, hidden_size)
    """

    patch_size: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    proj: Linear

    def __init__(self, patch_size: int, in_channels: int, hidden_size: int, *, key):
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.proj = Linear(patch_size**3 * in_channels, hidden_size, key=key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        D, H, W, C = x.shape
        p = self.patch_size
        x = x.reshape(D // p, p, H // p, p, W // p, p, C).transpose(0, 2, 4, 1, 3, 5, 6)
        x = x.reshape(D // p * H // p * W // p, p**3 * C)
        return self.proj(x)


# ── transformer block ─────────────────────────────────────────────────────────


class DiTBlock(eqx.Module):
    """Single DiT transformer block with adaLN-Zero conditioning.

    Applies:  x = x + gate_attn * Attn(adaLN(x))
              x = x + gate_mlp  * MLP(adaLN(x))
    """

    hidden_size: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)

    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    attn: SelfAttention
    mlp1: Linear
    mlp2: Linear
    adaLN_proj: Linear  # zero-initialised, emb_dim → 6*hidden_size

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        emb_dim: int,
        *,
        key,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        mlp_hidden = int(hidden_size * mlp_ratio)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.norm1 = eqx.nn.LayerNorm(hidden_size)
        self.norm2 = eqx.nn.LayerNorm(hidden_size)
        self.attn = SelfAttention(
            hidden_size, hidden_size, hidden_size, num_heads, key=k1
        )
        self.mlp1 = Linear(hidden_size, mlp_hidden, key=k2)
        self.mlp2 = Linear(mlp_hidden, hidden_size, key=k3)
        # adaLN-Zero: zero-initialise only the gate outputs (indices 2d:3d and 5d:)
        # so that the residual contributions are zero at init (identity block), while
        # shift/scale still produce non-zero conditioning from the first forward pass.
        ada = Linear(emb_dim, 6 * hidden_size, key=k4)
        d = hidden_size
        w = ada.weight.at[2 * d : 3 * d, :].set(0.0).at[5 * d :, :].set(0.0)
        ada = eqx.tree_at(lambda m: m.weight, ada, w)
        self.adaLN_proj = ada

    def __call__(self, x: jnp.ndarray, emb: jnp.ndarray) -> jnp.ndarray:
        """x: (N, hidden_size), emb: (emb_dim,) → (N, hidden_size)."""
        d = self.hidden_size
        cond = self.adaLN_proj(emb)
        shift_msa, scale_msa, gate_msa = cond[:d], cond[d : 2 * d], cond[2 * d : 3 * d]
        shift_mlp, scale_mlp, gate_mlp = (
            cond[3 * d : 4 * d],
            cond[4 * d : 5 * d],
            cond[5 * d :],
        )

        # Attention path with adaLN modulation
        x_a = jax.vmap(self.norm1)(x) * (1.0 + scale_msa) + shift_msa
        x = x + gate_msa * self.attn(x_a, x_a)

        # MLP path with adaLN modulation
        x_m = jax.vmap(self.norm2)(x) * (1.0 + scale_mlp) + shift_mlp
        x_m = self.mlp2(jax.nn.gelu(self.mlp1(x_m)))
        x = x + gate_mlp * x_m
        return x


# ── final layer ───────────────────────────────────────────────────────────────


class FinalLayer(eqx.Module):
    """Final adaLN + linear projection before unpatching."""

    hidden_size: int = eqx.field(static=True)

    norm: eqx.nn.LayerNorm
    proj: Linear
    adaLN_proj: Linear  # zero-initialised, emb_dim → 2*hidden_size

    def __init__(self, hidden_size: int, out_features: int, emb_dim: int, *, key):
        self.hidden_size = hidden_size
        k1, k2 = jax.random.split(key)
        self.norm = eqx.nn.LayerNorm(hidden_size)
        self.proj = Linear(hidden_size, out_features, key=k1)
        self.adaLN_proj = Linear(emb_dim, 2 * hidden_size, key=k2)

    def __call__(self, x: jnp.ndarray, emb: jnp.ndarray) -> jnp.ndarray:
        """x: (N, hidden_size), emb: (emb_dim,) → (N, out_features)."""
        d = self.hidden_size
        cond = self.adaLN_proj(emb)
        shift, scale = cond[:d], cond[d:]
        x = jax.vmap(self.norm)(x) * (1.0 + scale) + shift
        return self.proj(x)


# ── DiT2d ─────────────────────────────────────────────────────────────────────


class DiT2d(eqx.Module):
    """Diffusion Transformer for 2-D PDE fields.

    Input:  (H, W, in_channels)   — H and W must be divisible by patch_size
    Output: (H, W, in_channels)   — velocity field (same shape as input)
    """

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    patch_embed: PatchEmbed2d
    time_embed: SinusoidalTimeEmbedding
    blocks: list
    final_layer: FinalLayer

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        *,
        key,
    ):
        assert hidden_size % 4 == 0, "hidden_size must be divisible by 4"
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        keys = jax.random.split(key, depth + 3)
        self.patch_embed = PatchEmbed2d(
            patch_size, in_channels, hidden_size, key=keys[0]
        )
        self.time_embed = SinusoidalTimeEmbedding(hidden_size, key=keys[1])
        self.blocks = [
            DiTBlock(hidden_size, num_heads, mlp_ratio, hidden_size, key=keys[2 + i])
            for i in range(depth)
        ]
        self.final_layer = FinalLayer(
            hidden_size,
            patch_size * patch_size * in_channels,
            hidden_size,
            key=keys[-1],
        )

    def __call__(self, x: jnp.ndarray, t, **kwargs) -> jnp.ndarray:
        """x: (H, W, C), t: scalar → (H, W, C)."""
        H, W, C = x.shape
        p = self.patch_size
        H_p, W_p = H // p, W // p

        # Patch embed + positional encoding
        tokens = self.patch_embed(x)  # (N, hidden_size)
        pe = _sincos_pos_embed_2d(H_p, W_p, self.hidden_size)
        tokens = tokens + pe

        # Time embedding
        emb = self.time_embed(t)

        # Transformer blocks
        for blk in self.blocks:
            tokens = blk(tokens, emb)

        # Final layer → (N, p*p*C)
        out = self.final_layer(tokens, emb)

        # Unpatch: (N, p^2*C) → (H, W, C)
        out = out.reshape(H_p, W_p, p, p, C).transpose(0, 2, 1, 3, 4)
        return out.reshape(H, W, C)


# ── DiT3d ─────────────────────────────────────────────────────────────────────


class DiT3d(eqx.Module):
    """Diffusion Transformer for 3-D volumetric PDE fields.

    Input:  (D, H, W, in_channels)  — each spatial dim divisible by patch_size
    Output: (D, H, W, in_channels)
    """

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    patch_embed: PatchEmbed3d
    time_embed: SinusoidalTimeEmbedding
    blocks: list
    final_layer: FinalLayer

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        *,
        key,
    ):
        assert hidden_size % 6 == 0, (
            "hidden_size must be divisible by 6 for 3-D pos embed"
        )
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        keys = jax.random.split(key, depth + 3)
        self.patch_embed = PatchEmbed3d(
            patch_size, in_channels, hidden_size, key=keys[0]
        )
        self.time_embed = SinusoidalTimeEmbedding(hidden_size, key=keys[1])
        self.blocks = [
            DiTBlock(hidden_size, num_heads, mlp_ratio, hidden_size, key=keys[2 + i])
            for i in range(depth)
        ]
        self.final_layer = FinalLayer(
            hidden_size, patch_size**3 * in_channels, hidden_size, key=keys[-1]
        )

    def __call__(self, x: jnp.ndarray, t, **kwargs) -> jnp.ndarray:
        """x: (D, H, W, C), t: scalar → (D, H, W, C)."""
        D, H, W, C = x.shape
        p = self.patch_size
        D_p, H_p, W_p = D // p, H // p, W // p

        tokens = self.patch_embed(x)  # (N, hidden_size)
        pe = _sincos_pos_embed_3d(D_p, H_p, W_p, self.hidden_size)
        tokens = tokens + pe

        emb = self.time_embed(t)

        for blk in self.blocks:
            tokens = blk(tokens, emb)

        out = self.final_layer(tokens, emb)  # (N, p^3*C)
        out = out.reshape(D_p, H_p, W_p, p, p, p, C).transpose(0, 3, 1, 4, 2, 5, 6)
        return out.reshape(D, H, W, C)
