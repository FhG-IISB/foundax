#!/usr/bin/env python3
"""
FFNO: alasdairtran/fourierflow reference vs foundax Equinox port.

**Scope note.** The upstream ``FNOFactorized2DBlock`` from fourierflow
wraps each spectral conv in per-block FeedForward MLPs and uses a
``x = x + backcast`` residual pattern with a fixed 2-layer 128-channel
output head. foundax's ``FFNO2d`` is a structurally simpler
``lift → spectral-block-stack → project`` wrapper around the same
spectral primitive. So we verify the **factorized spectral conv
primitive** — the mathematical algorithm both share — rather than the
full upstream model. A bug in the F-FNO math would surface here; a
divergence in surrounding scaffolding is by design.

Tests:
  1. ``FactorizedSpectralConv2d`` parity vs upstream ``forward_fourier``.
  2. ``FactorizedSpectralConv3d`` parity vs upstream 3-D ``forward_fourier``.

Usage::
    python scripts/compare_ffno.py --fourierflow-root og_repos/fourierflow
"""

from __future__ import annotations

import argparse
import sys
from importlib.util import find_spec
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pt2eqx import compare_arrays, set_eqx_array


def _import_pt_spectral(fourierflow_root: Path):
    """Load the upstream ``SpectralConv2d`` leaves while bypassing
    ``fourierflow/__init__.py`` (which pulls in hydra / xarray / dotenv —
    framework deps unrelated to the actual model code)."""
    import importlib.util
    import types

    ff_root = fourierflow_root / "fourierflow"
    for pkg in [
        "fourierflow",
        "fourierflow.modules",
        "fourierflow.modules.factorized_fno",
    ]:
        if pkg not in sys.modules:
            stub = types.ModuleType(pkg)
            stub.__path__ = []  # mark as package
            sys.modules[pkg] = stub

    def _load(qualname, path):
        spec = importlib.util.spec_from_file_location(qualname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[qualname] = mod
        spec.loader.exec_module(mod)
        return mod

    # Order matters: leaf files relative-import from these.
    _load("fourierflow.modules.linear", ff_root / "modules/linear.py")
    _load("fourierflow.modules.feedforward", ff_root / "modules/feedforward.py")
    grid_2d = _load(
        "fourierflow.modules.factorized_fno.grid_2d",
        ff_root / "modules/factorized_fno/grid_2d.py",
    )
    mesh_3d = _load(
        "fourierflow.modules.factorized_fno.mesh_3d",
        ff_root / "modules/factorized_fno/mesh_3d.py",
    )
    return grid_2d.SpectralConv2d, mesh_3d.SpectralConv2d


def _stuff_complex_weight(pt_param, real, imag):
    """Pack our (in, out, modes) complex weight into upstream's
    (in, out, modes, 2) real/imag layout."""
    import torch
    stacked = np.stack([real, imag], axis=-1)
    pt_param.data = torch.from_numpy(stacked.astype(np.float32))


def compare_spectral_conv_2d(fourierflow_root: Path, seed: int) -> bool:
    import torch
    import jax, jax.numpy as jnp
    from foundax.architectures.ffno import FactorizedSpectralConv2d

    Pt2D, _ = _import_pt_spectral(fourierflow_root)

    # Upstream forward_fourier hard-codes the out tensor with `in_dim`
    # channels (works only when in_dim == out_dim — the only configuration
    # FNOFactorized2DBlock actually uses, where width == width).
    in_dim, out_dim, n_modes = 8, 8, 6
    H, W = 16, 16

    # Build PT with no FeedForward layers (mode='full'); we only want the
    # forward_fourier path.
    torch.manual_seed(seed)
    pt = Pt2D(
        in_dim=in_dim,
        out_dim=out_dim,
        n_modes=n_modes,
        forecast_ff=None,
        backcast_ff=None,
        fourier_weight=None,
        factor=2,
        ff_weight_norm=False,
        n_ff_layers=2,
        layer_norm=False,
        use_fork=False,
        dropout=0.0,
        mode="full",
    ).eval()

    eqx_conv = FactorizedSpectralConv2d(
        in_dim, out_dim, n_modes, n_modes, key=jax.random.PRNGKey(seed)
    )

    # Weight transfer:
    #   upstream fourier_weight[0]  ← our weight_x  (W-axis FFT)
    #   upstream fourier_weight[1]  ← our weight_y  (H-axis FFT)
    eqx_conv = set_eqx_array(
        eqx_conv, [("weight_x_real", None)],
        pt.fourier_weight[0].detach().cpu().numpy()[..., 0],
    )
    eqx_conv = set_eqx_array(
        eqx_conv, [("weight_x_imag", None)],
        pt.fourier_weight[0].detach().cpu().numpy()[..., 1],
    )
    eqx_conv = set_eqx_array(
        eqx_conv, [("weight_y_real", None)],
        pt.fourier_weight[1].detach().cpu().numpy()[..., 0],
    )
    eqx_conv = set_eqx_array(
        eqx_conv, [("weight_y_imag", None)],
        pt.fourier_weight[1].detach().cpu().numpy()[..., 1],
    )

    # Input
    x_t = torch.randn(1, H, W, in_dim, dtype=torch.float32)
    x_j = jnp.asarray(x_t[0].numpy())

    with torch.no_grad():
        pt_out = pt.forward_fourier(x_t)[0]  # (H, W, out_dim)
    eqx_out = eqx_conv(x_j)
    return compare_arrays("FactorizedSpectralConv2d", pt_out, eqx_out)


def compare_spectral_conv_3d(fourierflow_root: Path, seed: int) -> bool:
    import torch
    import jax, jax.numpy as jnp
    from foundax.architectures.ffno import FactorizedSpectralConv3d

    _, Pt3D = _import_pt_spectral(fourierflow_root)

    in_dim, out_dim = 6, 6  # upstream 3-D variant has same in==out constraint
    modes_x, modes_y, modes_z = 4, 5, 6
    D, H, W = 8, 10, 12

    torch.manual_seed(seed)
    pt = Pt3D(
        in_dim=in_dim,
        out_dim=out_dim,
        modes_x=modes_x,
        modes_y=modes_y,
        modes_z=modes_z,
        forecast_ff=None,
        backcast_ff=None,
        fourier_weight=None,
        factor=2,
        ff_weight_norm=False,
        n_ff_layers=2,
        layer_norm=False,
        use_fork=False,
        dropout=0.0,
    ).eval()

    eqx_conv = FactorizedSpectralConv3d(
        in_dim, out_dim,
        n_modes_d=modes_x,   # upstream X (S1, axis=-3) ↔ our D (axis=0)
        n_modes_h=modes_y,   # upstream Y (S2, axis=-2) ↔ our H (axis=1)
        n_modes_w=modes_z,   # upstream Z (S3, axis=-1) ↔ our W (axis=2)
        key=jax.random.PRNGKey(seed),
    )
    # Weight transfer: upstream [0]→D, [1]→H, [2]→W
    for upstream_idx, prefix in [
        (0, "d"), (1, "h"), (2, "w"),
    ]:
        w = pt.fourier_weight[upstream_idx].detach().cpu().numpy()
        eqx_conv = set_eqx_array(eqx_conv, [(f"weight_{prefix}_real", None)], w[..., 0])
        eqx_conv = set_eqx_array(eqx_conv, [(f"weight_{prefix}_imag", None)], w[..., 1])

    x_t = torch.randn(1, D, H, W, in_dim, dtype=torch.float32)
    x_j = jnp.asarray(x_t[0].numpy())

    with torch.no_grad():
        pt_out = pt.forward_fourier(x_t)[0]
    eqx_out = eqx_conv(x_j)
    return compare_arrays("FactorizedSpectralConv3d", pt_out, eqx_out)


def run_structural_check(seed: int) -> int:
    import jax, jax.numpy as jnp
    import foundax as fx

    print("[FFNO] Structural check (JAX only)")
    m = fx.ffno2d(in_channels=2, hidden_channels=16, out_channels=1, n_modes=4, n_layers=2, key=jax.random.PRNGKey(seed))
    x = jax.random.normal(jax.random.PRNGKey(seed + 1), (16, 16, 2))
    y = m(x)
    print(f"  output shape: {y.shape}, finite: {bool(jnp.all(jnp.isfinite(y)))}")
    return 0 if bool(jnp.all(jnp.isfinite(y))) else 1


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--fourierflow-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "og_repos" / "fourierflow",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if find_spec("torch") is None or not (args.fourierflow_root / "fourierflow").exists():
        print("torch or upstream missing — JAX-only structural check.")
        return run_structural_check(args.seed)

    print("=" * 70)
    print("FFNO: alasdairtran/fourierflow reference vs foundax Equinox port")
    print(f"  reference: {args.fourierflow_root}")
    print(f"  seed:      {args.seed}")
    print("=" * 70)
    ok = [
        compare_spectral_conv_2d(args.fourierflow_root, args.seed),
        compare_spectral_conv_3d(args.fourierflow_root, args.seed),
    ]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
