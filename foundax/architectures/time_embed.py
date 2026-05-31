# Time-conditioning primitives for diffusion / flow-matching models.
#
# Paper: "Denoising Diffusion Probabilistic Models" (sinusoidal embedding)
#        Ho et al. (2020) — https://arxiv.org/abs/2006.11239
# Paper: "Diffusion Models Beat GANs on Image Synthesis" (FiLM / classifier-free)
#        Dhariwal & Nichol (2021) — https://arxiv.org/abs/2105.05233
# Paper: "Scalable Diffusion Models with Transformers" (AdaLayerNorm-Zero)
#        Peebles & Xie (2022) — https://arxiv.org/abs/2212.09748

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from .linear import Linear


class SinusoidalTimeEmbedding(eqx.Module):
    """Sinusoidal timestep embedding followed by a 2-layer GELU MLP.

    Maps scalar t → R^dim.  Consistent with the DDPM convention
    (Ho et al., arXiv 2006.11239) but adds a learned MLP head.

    ``dim`` must be even.
    """

    linear1: Linear
    linear2: Linear
    dim: int = eqx.field(static=True)

    def __init__(self, dim: int, *, key):
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim
        k1, k2 = jax.random.split(key)
        self.linear1 = Linear(dim, dim, key=k1)
        self.linear2 = Linear(dim, dim, key=k2)

    def __call__(self, t):
        """t: scalar → (dim,)."""
        half = self.dim // 2
        freqs = jnp.exp(
            -jnp.log(10000.0)
            * jnp.arange(half, dtype=jnp.float32)
            / (half - 1)
        )
        emb = jnp.concatenate([jnp.sin(t * freqs), jnp.cos(t * freqs)])
        emb = jax.nn.gelu(self.linear1(emb))
        return self.linear2(emb)


class FiLMLayer(eqx.Module):
    """Feature-wise Linear Modulation: (1 + γ) * x + β from an embedding.

    Reference: Dhariwal & Nichol, arXiv 2105.05233.
    """

    proj: Linear
    feature_dim: int = eqx.field(static=True)

    def __init__(self, emb_dim: int, feature_dim: int, *, key):
        self.proj = Linear(emb_dim, 2 * feature_dim, key=key)
        self.feature_dim = feature_dim

    def __call__(self, x, emb):
        """x: (..., feature_dim), emb: (emb_dim,) → (..., feature_dim)."""
        params = self.proj(emb)
        gamma = params[: self.feature_dim]
        beta = params[self.feature_dim :]
        return (1.0 + gamma) * x + beta


class AdaLayerNorm(eqx.Module):
    """LayerNorm with scale and shift derived from an embedding.

    Reference: Peebles & Xie, arXiv 2212.09748.
    """

    norm: eqx.nn.LayerNorm
    proj: Linear
    dim: int = eqx.field(static=True)

    def __init__(self, dim: int, emb_dim: int, *, key):
        self.dim = dim
        self.norm = eqx.nn.LayerNorm(dim)
        self.proj = Linear(emb_dim, 2 * dim, key=key)

    def __call__(self, x, emb):
        """x: (dim,), emb: (emb_dim,) → (dim,).  Vmap for sequences."""
        params = self.proj(emb)
        scale, shift = params[: self.dim], params[self.dim :]
        return self.norm(x) * (1.0 + scale) + shift


class AdaLayerNormZero(eqx.Module):
    """adaLN-Zero: LayerNorm + conditioning with zero-initialised projection.

    Returns six conditioning signals for a DiT block:
    ``(x_normed_for_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)``.
    The zero-initialised weight makes each block an identity at init,
    which stabilises early training (Peebles & Xie, arXiv 2212.09748).
    """

    norm: eqx.nn.LayerNorm
    proj: Linear
    dim: int = eqx.field(static=True)

    def __init__(self, dim: int, emb_dim: int, *, key):
        self.dim = dim
        self.norm = eqx.nn.LayerNorm(dim)
        proj = Linear(emb_dim, 6 * dim, key=key)
        # Zero-init weight + bias so the block is identity at init
        proj = eqx.tree_at(lambda m: m.weight, proj, jnp.zeros_like(proj.weight))
        if proj.bias is not None:
            proj = eqx.tree_at(lambda m: m.bias, proj, jnp.zeros_like(proj.bias))
        self.proj = proj

    def __call__(self, x, emb):
        """x: (N, dim) token sequence, emb: (emb_dim,).

        Returns (x_normed, gate_attn, shift_mlp, scale_mlp, gate_mlp),
        each (N, dim) or (dim,) as appropriate.
        """
        d = self.dim
        cond = self.proj(emb)  # (6*d,)
        shift_msa = cond[:d]
        scale_msa = cond[d : 2 * d]
        gate_msa = cond[2 * d : 3 * d]
        shift_mlp = cond[3 * d : 4 * d]
        scale_mlp = cond[4 * d : 5 * d]
        gate_mlp = cond[5 * d :]
        x_normed = jax.vmap(self.norm)(x) * (1.0 + scale_msa) + shift_msa
        return x_normed, gate_msa, shift_mlp, scale_mlp, gate_mlp
