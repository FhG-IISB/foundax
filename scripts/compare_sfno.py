#!/usr/bin/env python3
"""
SFNO: torch-harmonics reference vs foundax Equinox port (forward equivalence).

torch-harmonics ships only the *primitives* (``RealSHT`` /
``InverseRealSHT``) — there's no canonical "SFNO" reference class. So we
construct an SFNO mirror in PyTorch using torch-harmonics primitives
with the same architectural recipe as ours (lift → SphericalConv +
Linear skip → activation → … → project), transfer weights tensor-by-
tensor, and compare full-model forwards.

Three tests:
  1. **SHT parity** — our ``RealSHT2d`` vs ``torch_harmonics.RealSHT``
     forward and inverse on the same input (sanity check that we share
     the orthonormal spherical-harmonic convention).
  2. **Block parity** — our ``SphericalConv2d`` vs the torch-harmonics
     mirror with identical weights.
  3. **Model parity** — full ``SFNO2d`` vs PT mirror with weights
     transferred.

Falls back to a JAX-only structural check if torch-harmonics is missing.

Usage::

    python scripts/compare_sfno.py
"""

from __future__ import annotations

import argparse
import sys
from importlib.util import find_spec
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pt2eqx import (
    compare_arrays as _compare,
    copy_linear as _copy_linear,
    set_eqx_array as _set_eqx,
)


# ── PT mirror modules (built from torch-harmonics primitives) ──────────────


def _build_pt_spherical_conv(torch, th_nn, th, in_ch, out_ch, L, nlat, nlon, grid):
    """Mirror of foundax.architectures.sfno.SphericalConv2d in PyTorch."""

    class PtSphericalConv2d(th_nn.Module):
        def __init__(self):
            super().__init__()
            self.sht = th.RealSHT(nlat=nlat, nlon=nlon, lmax=L, mmax=L, grid=grid)
            self.isht = th.InverseRealSHT(nlat=nlat, nlon=nlon, lmax=L, mmax=L, grid=grid)
            # Match foundax shape (L, L, in_ch, out_ch) so weight transfer
            # is identity (no transpose).
            self.weight_real = th_nn.Parameter(torch.zeros(L, L, in_ch, out_ch))
            self.weight_imag = th_nn.Parameter(torch.zeros(L, L, in_ch, out_ch))

        def forward(self, x):
            # x: (B, nlat, nlon, in_ch) channel-last
            x_cf = x.permute(0, 3, 1, 2)            # (B, in_ch, nlat, nlon)
            f_lm = self.sht(x_cf)                    # (B, in_ch, L, L) complex
            w = self.weight_real + 1j * self.weight_imag  # (L, L, in_ch, out_ch)
            out_lm = torch.einsum("bilm,lmio->bolm", f_lm, w)
            out = self.isht(out_lm)                  # (B, out_ch, nlat, nlon)
            return out.permute(0, 2, 3, 1)            # (B, nlat, nlon, out_ch)

    return PtSphericalConv2d()


def _build_pt_sfno(torch, th_nn, th, in_ch, hidden, out_ch, L, nlat, nlon, n_layers, grid, act):
    """Mirror of foundax.architectures.sfno.SFNO2d in PyTorch."""

    act_fn = {"gelu": torch.nn.functional.gelu, "silu": torch.nn.functional.silu}[act]

    class PtSphericalBlock2d(th_nn.Module):
        def __init__(self):
            super().__init__()
            self.spectral = _build_pt_spherical_conv(
                torch, th_nn, th, hidden, hidden, L, nlat, nlon, grid
            )
            self.linear = th_nn.Linear(hidden, hidden)

        def forward(self, x):
            return act_fn(self.spectral(x) + self.linear(x))

    class PtSFNO2d(th_nn.Module):
        def __init__(self):
            super().__init__()
            self.lift = th_nn.Linear(in_ch, hidden)
            self.blocks = th_nn.ModuleList([PtSphericalBlock2d() for _ in range(n_layers)])
            self.project = th_nn.Linear(hidden, out_ch)

        def forward(self, x):
            x = self.lift(x)
            for blk in self.blocks:
                x = blk(x)
            return self.project(x)

    return PtSFNO2d()


# ── weight transfer PT → EQX ───────────────────────────────────────────────


def transfer_sfno_weights(pt_model, eqx_model):
    """Copy every parameter from PtSFNO2d into foundax SFNO2d."""
    eqx_model = _copy_linear(eqx_model, [("lift", None)], pt_model.lift)
    eqx_model = _copy_linear(eqx_model, [("project", None)], pt_model.project)
    for i, pt_blk in enumerate(pt_model.blocks):
        # spectral weights are (L, L, in, out) in both — direct copy.
        eqx_model = _set_eqx(
            eqx_model,
            [("blocks", i), ("spectral", None), ("weight_real", None)],
            pt_blk.spectral.weight_real.detach().cpu().numpy(),
        )
        eqx_model = _set_eqx(
            eqx_model,
            [("blocks", i), ("spectral", None), ("weight_imag", None)],
            pt_blk.spectral.weight_imag.detach().cpu().numpy(),
        )
        eqx_model = _copy_linear(
            eqx_model, [("blocks", i), ("linear", None)], pt_blk.linear
        )
    return eqx_model


# ── comparison drivers ─────────────────────────────────────────────────────


def compare_sht(seed: int) -> bool:
    """Test #1: SHT and inverse SHT parity, both directions."""
    import torch
    import torch_harmonics as th
    import jax.numpy as jnp
    from foundax.architectures.sfno import RealSHT2d

    L, nlat, nlon = 8, 32, 64
    torch.manual_seed(seed)

    # Forward: same input grid → matching spectral coefficients
    th_sht = th.RealSHT(nlat=nlat, nlon=nlon, lmax=L, mmax=L, grid="legendre-gauss")
    ours = RealSHT2d(L=L, nlat=nlat, nlon=nlon, grid="legendre-gauss")
    x_t = torch.randn(1, nlat, nlon, dtype=torch.float32)
    x_j = jnp.asarray(x_t.squeeze(0).unsqueeze(-1).numpy())  # (nlat, nlon, 1)
    th_coef = th_sht(x_t).detach().cpu().numpy()[0]          # (L, L)
    ours_coef = np.asarray(ours.forward(x_j))[..., 0]
    ok_fwd = _compare("SHT forward", th_coef, ours_coef)

    # Inverse: bandlimited coefficients → matching grids
    th_isht = th.InverseRealSHT(nlat=nlat, nlon=nlon, lmax=L, mmax=L, grid="legendre-gauss")
    coef_t = torch.randn(L, L, dtype=torch.complex64) / 10
    mask = (torch.arange(L)[None, :] <= torch.arange(L)[:, None]).to(torch.complex64)
    coef_t = coef_t * mask
    coef_t[:, 0] = coef_t[:, 0].real.to(torch.complex64)
    th_grid = th_isht(coef_t.unsqueeze(0)).detach().cpu().numpy()[0]
    ours_grid = np.asarray(ours.inverse(jnp.asarray(coef_t.numpy())[..., None]))[..., 0]
    ok_inv = _compare("SHT inverse", th_grid, ours_grid)
    return ok_fwd and ok_inv


def compare_spherical_conv(seed: int) -> bool:
    """Test #2: A single SphericalConv2d block with the same weights."""
    import torch
    import torch.nn as nn
    import torch_harmonics as th
    import jax
    from foundax.architectures.sfno import SphericalConv2d

    L, nlat, nlon = 8, 32, 64
    in_ch, out_ch = 2, 3
    torch.manual_seed(seed)
    pt_conv = _build_pt_spherical_conv(
        torch, nn, th, in_ch, out_ch, L, nlat, nlon, "legendre-gauss"
    )
    # Random init
    with torch.no_grad():
        pt_conv.weight_real.normal_(0, 0.1)
        pt_conv.weight_imag.normal_(0, 0.1)
    pt_conv.eval()

    eqx_conv = SphericalConv2d(
        in_ch, out_ch, L=L, nlat=nlat, nlon=nlon, key=jax.random.PRNGKey(seed)
    )
    eqx_conv = _set_eqx(
        eqx_conv,
        [("weight_real", None)],
        pt_conv.weight_real.detach().cpu().numpy(),
    )
    eqx_conv = _set_eqx(
        eqx_conv,
        [("weight_imag", None)],
        pt_conv.weight_imag.detach().cpu().numpy(),
    )

    x = torch.randn(1, nlat, nlon, in_ch, dtype=torch.float32)
    with torch.no_grad():
        pt_out = pt_conv(x)
    eqx_out = eqx_conv(x[0].numpy())  # foundax is unbatched
    return _compare("SphericalConv2d", pt_out[0], eqx_out)


def compare_sfno_full(seed: int) -> bool:
    """Test #3: full SFNO2d forward parity."""
    import torch
    import torch.nn as nn
    import torch_harmonics as th
    import jax
    from foundax.architectures.sfno import SFNO2d

    L, nlat, nlon = 6, 16, 32
    in_ch, hidden, out_ch = 3, 16, 2
    n_layers = 2

    torch.manual_seed(seed)
    pt = _build_pt_sfno(
        torch, nn, th, in_ch, hidden, out_ch, L, nlat, nlon, n_layers,
        grid="legendre-gauss", act="gelu",
    )
    pt.eval()

    eqx_model = SFNO2d(
        in_channels=in_ch, hidden_channels=hidden, out_channels=out_ch,
        L=L, nlat=nlat, nlon=nlon, n_layers=n_layers,
        grid="legendre-gauss", activation="gelu",
        key=jax.random.PRNGKey(seed),
    )
    eqx_model = transfer_sfno_weights(pt, eqx_model)

    x = torch.randn(1, nlat, nlon, in_ch, dtype=torch.float32)
    with torch.no_grad():
        pt_out = pt(x)
    eqx_out = eqx_model(x[0].numpy())
    return _compare("SFNO2d full", pt_out[0], eqx_out)


def run_structural_check(seed: int) -> int:
    import jax
    import jax.numpy as jnp
    from foundax.architectures.sfno import SFNO2d

    print("[SFNO] Structural check (JAX only, random weights)")
    m = SFNO2d(
        in_channels=2, hidden_channels=16, out_channels=1,
        L=8, nlat=16, nlon=32, n_layers=2, key=jax.random.PRNGKey(seed),
    )
    x = jax.random.normal(jax.random.PRNGKey(seed + 1), (16, 32, 2))
    y = m(x)
    print(f"  output shape: {y.shape}, finite: {bool(jnp.all(jnp.isfinite(y)))}")
    return 0 if bool(jnp.all(jnp.isfinite(y))) else 1


# ── entry point ────────────────────────────────────────────────────────────


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if find_spec("torch") is None or find_spec("torch_harmonics") is None:
        print("torch / torch-harmonics missing — running JAX-only structural check.")
        return run_structural_check(args.seed)

    print("=" * 70)
    print("SFNO: torch-harmonics reference vs foundax Equinox port")
    print(f"  seed: {args.seed}")
    print("=" * 70)
    results = [
        compare_sht(args.seed),
        compare_spherical_conv(args.seed),
        compare_sfno_full(args.seed),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
