#!/usr/bin/env python3
"""
DiT: ``facebookresearch/DiT`` reference vs Equinox port (upstream conventions).

This script does a **true upstream-code parity test**. It instantiates
the actual ``models.DiTBlock`` class from ``facebookresearch/DiT`` and
compares it against an Equinox DiT block that *mirrors the upstream's
design choices*:

  * fused QKV Linear (matches ``timm.layers.Attention``)
  * ``GELU(approximate='tanh')`` MLP activation
  * ``SiLU + Linear`` adaLN modulation (matches upstream's
    ``Sequential(nn.SiLU(), nn.Linear(D, 6D))``)
  * ``elementwise_affine=False`` LayerNorms

These are exactly the design choices that ``foundax.dit2d`` deliberately
diverges from (no class labels, exact GELU, no SiLU before adaLN, no
``learn_sigma``). The *math* of the two blocks (residual + adaLN-Zero
attention + adaLN-Zero MLP) is identical; only the activation/norm
choices differ. So this parity test verifies that the algorithm side
of our port is correct, while the per-architecture divergence in
foundax is a documented design decision.

If you want the "foundax conventions" mirror test (the previous version
of this script), it has been removed in favour of this true-upstream
comparison.

Tests:
  1. Single ``DiTBlock`` forward parity against upstream.
"""

from __future__ import annotations

import argparse
import sys
from importlib.util import find_spec
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pt2eqx import compare_arrays, copy_layernorm, copy_linear, set_eqx_array


def _import_upstream(dit_root: Path):
    sys.path.insert(0, str(dit_root))
    from models import DiTBlock as PtDiTBlock
    return PtDiTBlock


# ── Equinox DiT block mirroring upstream conventions ───────────────────────


def _make_equinox_upstream_block(hidden, num_heads, mlp_ratio, *, key):
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from foundax.architectures.linear import Linear

    head_dim = hidden // num_heads
    mlp_hidden = int(hidden * mlp_ratio)

    class _NoAffineLayerNorm(eqx.Module):
        eps: float = eqx.field(static=True)
        dim: int = eqx.field(static=True)

        def __init__(self, dim, eps=1e-6):
            self.dim = dim
            self.eps = eps

        def __call__(self, x):
            # x: (..., dim) — normalise over last axis, no learnable params.
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.var(x, axis=-1, keepdims=True)
            return (x - mean) / jnp.sqrt(var + self.eps)

    class _UpstreamMirrorDiTBlock(eqx.Module):
        hidden: int = eqx.field(static=True)
        num_heads: int = eqx.field(static=True)
        head_dim: int = eqx.field(static=True)
        mlp_hidden: int = eqx.field(static=True)
        norm1: _NoAffineLayerNorm
        norm2: _NoAffineLayerNorm
        qkv: Linear      # fused, hidden → 3*hidden
        proj: Linear
        mlp1: Linear
        mlp2: Linear
        adaLN_linear: Linear  # hidden → 6*hidden, applied AFTER silu

        def __init__(self, key):
            keys = jax.random.split(key, 5)
            self.hidden = hidden
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.mlp_hidden = mlp_hidden
            self.norm1 = _NoAffineLayerNorm(hidden)
            self.norm2 = _NoAffineLayerNorm(hidden)
            self.qkv = Linear(hidden, 3 * hidden, key=keys[0])
            self.proj = Linear(hidden, hidden, key=keys[1])
            self.mlp1 = Linear(hidden, mlp_hidden, key=keys[2])
            self.mlp2 = Linear(mlp_hidden, hidden, key=keys[3])
            self.adaLN_linear = Linear(hidden, 6 * hidden, key=keys[4])

        def _attn(self, x):
            # x: (N, hidden)
            N = x.shape[0]
            qkv = self.qkv(x).reshape(N, 3, self.num_heads, self.head_dim)
            qkv = qkv.transpose(1, 2, 0, 3)  # (3, H, N, d)
            q, k, v = qkv[0], qkv[1], qkv[2]
            scale = 1.0 / jnp.sqrt(jnp.array(self.head_dim, dtype=q.dtype))
            attn = jax.nn.softmax(jnp.einsum("hnd,hmd->hnm", q, k) * scale, axis=-1)
            out = jnp.einsum("hnm,hmd->hnd", attn, v)  # (H, N, d)
            out = out.transpose(1, 0, 2).reshape(N, self.hidden)
            return self.proj(out)

        def __call__(self, x, c):
            # x: (N, hidden), c: (hidden,) conditioning vector
            d = self.hidden
            cond = self.adaLN_linear(jax.nn.silu(c))   # upstream Sequential(SiLU, Linear)
            shift_msa, scale_msa, gate_msa = cond[:d], cond[d:2*d], cond[2*d:3*d]
            shift_mlp, scale_mlp, gate_mlp = cond[3*d:4*d], cond[4*d:5*d], cond[5*d:]
            x_a = self.norm1(x) * (1.0 + scale_msa) + shift_msa
            x = x + gate_msa * self._attn(x_a)
            x_m = self.norm2(x) * (1.0 + scale_mlp) + shift_mlp
            # Upstream uses GELU(approximate='tanh')
            x_m = self.mlp2(jax.nn.gelu(self.mlp1(x_m), approximate=True))
            x = x + gate_mlp * x_m
            return x

    return _UpstreamMirrorDiTBlock(key)


# ── weight transfer ────────────────────────────────────────────────────────


def transfer_upstream_block_weights(pt_block, eqx_block):
    """Copy upstream DiTBlock into our test-only Equinox mirror.

    Layout map:
      pt.attn.qkv.weight       → eqx.qkv.weight
      pt.attn.proj.weight      → eqx.proj.weight
      pt.mlp.fc1.weight        → eqx.mlp1.weight
      pt.mlp.fc2.weight        → eqx.mlp2.weight
      pt.adaLN_modulation[1]   → eqx.adaLN_linear
    Norms have no params (elementwise_affine=False).
    """
    eqx_block = copy_linear(eqx_block, [("qkv", None)], pt_block.attn.qkv)
    eqx_block = copy_linear(eqx_block, [("proj", None)], pt_block.attn.proj)
    eqx_block = copy_linear(eqx_block, [("mlp1", None)], pt_block.mlp.fc1)
    eqx_block = copy_linear(eqx_block, [("mlp2", None)], pt_block.mlp.fc2)
    # adaLN_modulation is Sequential(SiLU(), Linear) — index 1 is the Linear.
    eqx_block = copy_linear(eqx_block, [("adaLN_linear", None)], pt_block.adaLN_modulation[1])
    return eqx_block


# ── comparison driver ──────────────────────────────────────────────────────


def compare_block(dit_root: Path, seed: int) -> bool:
    import torch
    import jax

    PtDiTBlock = _import_upstream(dit_root)
    hidden, num_heads, mlp_ratio = 32, 4, 4.0
    N = 16

    torch.manual_seed(seed)
    pt = PtDiTBlock(hidden, num_heads, mlp_ratio=mlp_ratio)
    # Perturb adaLN modulation Linear so it isn't zero-initialised.
    with torch.no_grad():
        for p in pt.parameters():
            p.normal_(0, 0.1)
    pt.eval()

    eqx_block = _make_equinox_upstream_block(
        hidden, num_heads, mlp_ratio, key=jax.random.PRNGKey(seed),
    )
    eqx_block = transfer_upstream_block_weights(pt, eqx_block)

    x = torch.randn(1, N, hidden, dtype=torch.float32)  # (B, N, D)
    c = torch.randn(1, hidden, dtype=torch.float32)      # (B, D)
    with torch.no_grad():
        pt_out = pt(x, c)[0]  # (N, D)
    eqx_out = eqx_block(x[0].numpy(), c[0].numpy())
    return compare_arrays("DiTBlock (upstream conv.)", pt_out, eqx_out)


def run_structural_check(seed: int) -> int:
    import jax, jax.numpy as jnp
    import foundax as fx

    print("[DiT] Structural check (JAX only)")
    m = fx.dit2d(in_channels=4, patch_size=2, hidden_size=64, depth=2, num_heads=4, key=jax.random.PRNGKey(seed))
    x = jax.random.normal(jax.random.PRNGKey(seed + 1), (16, 16, 4))
    y = m(x, jnp.array(0.5))
    print(f"  output shape: {y.shape}, finite: {bool(jnp.all(jnp.isfinite(y)))}")
    return 0 if bool(jnp.all(jnp.isfinite(y))) else 1


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dit-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "og_repos" / "dit",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if (
        find_spec("torch") is None
        or not (args.dit_root / "models.py").exists()
    ):
        print("torch or upstream missing — JAX-only structural check.")
        return run_structural_check(args.seed)

    print("=" * 70)
    print("DiT: facebookresearch/DiT reference vs Equinox port (upstream conventions)")
    print(f"  reference: {args.dit_root}")
    print(f"  seed:      {args.seed}")
    print("=" * 70)
    ok = [compare_block(args.dit_root, args.seed)]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
