#!/usr/bin/env python3
"""
FNO: ``neuraloperator/neuraloperator`` reference vs foundax Equinox port.

Uses the **upstream** ``neuralop.layers.legacy_spectral_convolution.SpectralConv``
with ``factorization=None`` (dense weights), ``fft_norm='ortho'``, and
``bias=False`` to match foundax's convention. With that configuration
the quadrant-by-quadrant weight tensors map directly to ours:

  * neuralop ``n_modes = (2·m_0, 2·m_1, …, 2·m_{d-2}, 2·m_{d-1})`` →
    ``half_n_modes = [m_0, …, m_{d-1}]`` = foundax ``n_modes_i``.
  * For each FFT axis except the last, neuralop keeps both the
    ``(:m_i)`` and ``(-m_i:)`` halves → 2^(d-1) weights total. The
    quadrant ordering ``itertools.product([(+, −)]·(d−1))`` lines up
    one-for-one with foundax's ``weight_1, …, weight_{2^(d-1)}``.
  * For the last (rfft) axis, only the first ``m_{d-1}`` modes are
    kept — same in both.

So this script does true full-upstream-code parity: tensors copied
out of the actual neuralop module are placed into foundax's
``SpectralConv{1,2,3}d`` and forward outputs are compared element-wise.

Foundax's ``linear_conv=True`` mode (pad-to-2N−1 ⇒ linear instead of
circular convolution) is foundax-specific and not part of upstream FNO,
so we set ``linear_conv=False`` for the parity test.

Usage::
    python scripts/compare_fno.py --neuralop-root og_repos/neuraloperator
"""

from __future__ import annotations

import argparse
import sys
from importlib.util import find_spec
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pt2eqx import compare_arrays, set_eqx_array


def _import_upstream(neuralop_root: Path):
    sys.path.insert(0, str(neuralop_root))
    from neuralop.layers.legacy_spectral_convolution import SpectralConv
    return SpectralConv


# ── weight transfer ───────────────────────────────────────────────────────


def _copy_complex_weight(eqx_module, prefix_real, prefix_imag, complex_array):
    """Split complex ndarray into real/imag and write to foundax tensor."""
    eqx_module = set_eqx_array(eqx_module, prefix_real, complex_array.real)
    eqx_module = set_eqx_array(eqx_module, prefix_imag, complex_array.imag)
    return eqx_module


# ── comparison drivers ────────────────────────────────────────────────────


def compare_1d(neuralop_root: Path, seed: int) -> bool:
    import torch
    import jax
    from foundax.architectures.fno import SpectralConv1d

    SpectralConv = _import_upstream(neuralop_root)
    in_ch, out_ch, n_modes_foundax, W = 4, 6, 5, 32

    torch.manual_seed(seed)
    sc = SpectralConv(
        in_channels=in_ch, out_channels=out_ch,
        n_modes=(2 * n_modes_foundax,),
        factorization=None, bias=False, fft_norm="ortho",
    ).eval()
    w0 = sc.weight[0].to_tensor().detach().cpu().numpy()  # (in, out, m)

    eqx_conv = SpectralConv1d(in_ch, out_ch, n_modes_foundax,
                              linear_conv=False, key=jax.random.PRNGKey(seed))
    eqx_conv = _copy_complex_weight(
        eqx_conv, [("weight_real", None)], [("weight_imag", None)], w0,
    )

    x_t = torch.randn(1, in_ch, W, dtype=torch.float32)        # (B, C, W)
    x_j = x_t[0].permute(1, 0).numpy()                          # (W, C)
    with torch.no_grad():
        pt_out = sc(x_t)[0].permute(1, 0).detach().cpu().numpy()  # (W, out)
    eqx_out = eqx_conv(x_j)
    return compare_arrays("SpectralConv1d", pt_out, eqx_out)


def compare_2d(neuralop_root: Path, seed: int) -> bool:
    import torch
    import jax
    from foundax.architectures.fno import SpectralConv2d

    SpectralConv = _import_upstream(neuralop_root)
    in_ch, out_ch = 3, 5
    m1, m2 = 4, 5
    H, W = 16, 20

    torch.manual_seed(seed)
    sc = SpectralConv(
        in_channels=in_ch, out_channels=out_ch,
        n_modes=(2 * m1, 2 * m2),
        factorization=None, bias=False, fft_norm="ortho",
    ).eval()
    w0 = sc.weight[0].to_tensor().detach().cpu().numpy()  # ++ quadrant
    w1 = sc.weight[1].to_tensor().detach().cpu().numpy()  # -+ quadrant

    eqx_conv = SpectralConv2d(in_ch, out_ch, m1, m2,
                              linear_conv=False, key=jax.random.PRNGKey(seed))
    # foundax weight_1 = upper-H slice → neuralop quadrant 0 (++)
    # foundax weight_2 = lower-H slice → neuralop quadrant 1 (-+)
    eqx_conv = _copy_complex_weight(
        eqx_conv, [("weight_1_real", None)], [("weight_1_imag", None)], w0,
    )
    eqx_conv = _copy_complex_weight(
        eqx_conv, [("weight_2_real", None)], [("weight_2_imag", None)], w1,
    )

    x_t = torch.randn(1, in_ch, H, W, dtype=torch.float32)
    x_j = x_t[0].permute(1, 2, 0).numpy()
    with torch.no_grad():
        pt_out = sc(x_t)[0].permute(1, 2, 0).detach().cpu().numpy()
    eqx_out = eqx_conv(x_j)
    return compare_arrays("SpectralConv2d", pt_out, eqx_out)


def compare_3d(neuralop_root: Path, seed: int) -> bool:
    import torch
    import jax
    from foundax.architectures.fno import SpectralConv3d

    SpectralConv = _import_upstream(neuralop_root)
    in_ch, out_ch = 3, 4
    m1, m2, m3 = 3, 4, 5
    D, H, W = 12, 14, 16

    torch.manual_seed(seed)
    sc = SpectralConv(
        in_channels=in_ch, out_channels=out_ch,
        n_modes=(2 * m1, 2 * m2, 2 * m3),
        factorization=None, bias=False, fft_norm="ortho",
    ).eval()
    # Neuralop quadrant order over (D, H) axes: (++), (+-), (-+), (--).
    # Maps directly to foundax weight_1, weight_2, weight_3, weight_4.
    weights = [sc.weight[i].to_tensor().detach().cpu().numpy() for i in range(4)]

    eqx_conv = SpectralConv3d(in_ch, out_ch, m1, m2, m3,
                              linear_conv=False, key=jax.random.PRNGKey(seed))
    for i, w in enumerate(weights, start=1):
        eqx_conv = _copy_complex_weight(
            eqx_conv,
            [(f"weight_{i}_real", None)], [(f"weight_{i}_imag", None)],
            w,
        )

    x_t = torch.randn(1, in_ch, D, H, W, dtype=torch.float32)
    x_j = x_t[0].permute(1, 2, 3, 0).numpy()
    with torch.no_grad():
        pt_out = sc(x_t)[0].permute(1, 2, 3, 0).detach().cpu().numpy()
    eqx_out = eqx_conv(x_j)
    return compare_arrays("SpectralConv3d", pt_out, eqx_out)


def run_structural_check(seed: int) -> int:
    import jax, jax.numpy as jnp
    import foundax as fx

    print("[FNO] Structural check (JAX only)")
    m = fx.fno2d(in_features=2, hidden_channels=16, n_modes=8, key=jax.random.PRNGKey(seed))
    x = jax.random.normal(jax.random.PRNGKey(seed + 1), (16, 16, 2))
    y = m(x)
    print(f"  output shape: {y.shape}, finite: {bool(jnp.all(jnp.isfinite(y)))}")
    return 0 if bool(jnp.all(jnp.isfinite(y))) else 1


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--neuralop-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "og_repos" / "neuraloperator",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if (
        find_spec("torch") is None
        or not (args.neuralop_root / "neuralop").exists()
    ):
        print("torch or upstream missing — JAX-only structural check.")
        return run_structural_check(args.seed)

    print("=" * 70)
    print("FNO: neuraloperator/neuraloperator reference vs foundax Equinox port")
    print(f"  reference: {args.neuralop_root}")
    print(f"  seed:      {args.seed}")
    print("=" * 70)
    ok = [
        compare_1d(args.neuralop_root, args.seed),
        compare_2d(args.neuralop_root, args.seed),
        compare_3d(args.neuralop_root, args.seed),
    ]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
