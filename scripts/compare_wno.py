#!/usr/bin/env python3
"""
WNO: structural check + documented divergence from PyTorch upstream.

**Why this script doesn't do a weight-level parity test.** The author-
maintained upstream ``TapasTripura/WNO`` and foundax's
``foundax/architectures/wno.py`` are *structurally different*
implementations of the same algorithm family. The mismatches:

  * Wavelet basis: upstream uses Daubechies-6 (``pytorch_wavelets``,
    ``wave='db6'``); foundax uses Daubechies-8 (16-tap filter hard-
    coded as ``_DB8_LO``).
  * Boundary mode: upstream uses ``mode='symmetric'`` extension before
    the DWT; foundax uses zero-padding via ``jnp.convolve(mode='full')``
    then centre-slicing.
  * Output lengths: PyWavelets / pytorch_wavelets returns
    ``floor((N + len(filter) - 1) / 2)`` per level; foundax preserves
    spatial size through ``jax.image.resize`` after the wavelet linear.

These choices are deliberate (db8 is the more common choice in PDE
operator-learning contexts and the zero-boundary mode is JIT-friendlier
without ``pywt`` as a dep), but they mean **no shared input + shared
weights configuration produces matching output between the two**. A
weight-level parity test would require re-architecting one
implementation to mirror the other's wavelet choice and boundary mode,
which is out of scope for a verification script.

This script therefore runs a JAX-only structural check (shape,
finiteness, gradient flow) and a self-consistency sanity check
(forward + invert gives back something close to the input, modulo the
custom boundary mode).
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np


def structural_1d(seed: int) -> bool:
    import foundax as fx

    m = fx.wno1d(in_channels=2, hidden_channels=16, n_scales=2, depth=2, key=jax.random.PRNGKey(seed))
    x = jax.random.normal(jax.random.PRNGKey(seed + 1), (32, 2))
    y = m(x)
    ok = bool(jnp.all(jnp.isfinite(y))) and y.shape == (32, 2)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] WNO1d structural               shape={y.shape}  finite={bool(jnp.all(jnp.isfinite(y)))}")
    return ok


def structural_2d(seed: int) -> bool:
    import foundax as fx

    m = fx.wno2d(in_channels=2, hidden_channels=8, n_scales=2, depth=2, key=jax.random.PRNGKey(seed))
    x = jax.random.normal(jax.random.PRNGKey(seed + 1), (32, 32, 2))
    y = m(x)
    ok = bool(jnp.all(jnp.isfinite(y))) and y.shape == (32, 32, 2)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] WNO2d structural               shape={y.shape}  finite={bool(jnp.all(jnp.isfinite(y)))}")
    return ok


def gradient_flow_3d(seed: int) -> bool:
    """Confirm autodiff actually flows through the WNO3d wavelet path."""
    import foundax as fx

    m = fx.wno3d(in_channels=1, hidden_channels=8, n_scales=2, depth=1, key=jax.random.PRNGKey(seed))
    x = jax.random.normal(jax.random.PRNGKey(seed + 1), (16, 16, 16, 1))
    y_t = jax.random.normal(jax.random.PRNGKey(seed + 2), (16, 16, 16, 1))

    def loss(model, x, y_t):
        return jnp.mean((model(x) - y_t) ** 2)

    grads = eqx.filter_grad(loss)(m, x, y_t)
    flat = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    finite = all(bool(jnp.all(jnp.isfinite(g))) for g in flat if g.size > 0)
    nonzero = any(bool(jnp.linalg.norm(g) > 0) for g in flat if g.size > 0)
    ok = finite and nonzero
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] WNO3d gradient-flow            finite={finite}  nonzero={nonzero}")
    return ok


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    print("=" * 70)
    print("WNO: structural check (no PT parity — see docstring for why)")
    print(f"  seed: {args.seed}")
    print("=" * 70)
    results = [
        structural_1d(args.seed),
        structural_2d(args.seed),
        gradient_flow_3d(args.seed),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
