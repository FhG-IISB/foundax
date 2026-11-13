#!/usr/bin/env python3
"""
Geo-FNO: ``neuraloperator/Geo-FNO`` reference vs foundax Equinox port.

The upstream model lives in training scripts (``elasticity/elas_geofno.py``
and ``elas_geofno_v2.py``) that load data and train at import time, and its
``IPHI`` hard-codes ``device="cuda"``. This script therefore parses each file,
executes only its class definitions (``SpectralConv2d``, ``FNO2d``, ``IPHI``)
with ``"cuda"`` replaced by ``"cpu"``, copies seeded PyTorch weights into
``fx.geofno`` tensor by tensor and compares forward outputs.

Checks:
  * v2 ``FNO2d`` + v2 ``IPHI`` end to end (``iphi_depth=2, iphi_act='gelu',
    iphi_center=(0.5, 0.5)``), with query points equal to and different
    from the input points.
  * v1 ``IPHI`` (the paper's network; ``iphi_depth=3, tanh``), with and
    without a design code.
  * v1 ``FNO2d`` is **not** compared end to end: its inverse transform
    builds the negative-``k1`` coefficients with ``flip(-1, -2)``, which is
    off by one row (``k1 -> -k1 - 1``). v2 fixes this with an explicit index
    and foundax follows v2.

Usage::
    python scripts/compare_geofno.py --geofno-root og_repos/geofno
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pt2eqx import compare_arrays, copy_linear, set_eqx_array  # noqa: E402

UPSTREAM_CLASSES = ("SpectralConv2d", "FNO2d", "IPHI")


def load_upstream(path: Path) -> dict:
    """Execute only the model class definitions of an upstream script."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    src = path.read_text().replace('device="cuda"', 'device="cpu"')
    tree = ast.parse(src)
    tree.body = [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name in UPSTREAM_CLASSES
    ]
    ns = {"torch": torch, "nn": nn, "F": F, "np": np}
    exec(compile(tree, str(path), "exec"), ns)
    return ns


def _np(t):
    return t.detach().cpu().numpy()


def copy_spectral(m, prefix, pt):
    for i in (1, 2):
        w = _np(getattr(pt, f"weights{i}"))
        m = set_eqx_array(m, prefix + [(f"weights{i}_real", None)], w.real)
        m = set_eqx_array(m, prefix + [(f"weights{i}_imag", None)], w.imag)
    return m


def copy_pointwise_conv(m, prefix, pt_conv):
    """PT ``Conv{1,2}d(k=1)`` weight ``(out, in, 1[, 1])`` -> Linear ``(out, in)``."""
    w = _np(pt_conv.weight).reshape(pt_conv.weight.shape[0], -1)
    m = set_eqx_array(m, prefix + [("weight", None)], w)
    return set_eqx_array(m, prefix + [("bias", None)], _np(pt_conv.bias))


def copy_iphi(m, prefix, pt, depth):
    m = copy_linear(m, prefix + [("fc0", None)], pt.fc0)
    if getattr(m_at(m, prefix), "fc_code") is not None:
        m = copy_linear(m, prefix + [("fc_code", None)], pt.fc_code)
    else:
        m = copy_linear(m, prefix + [("fc_no_code", None)], pt.fc_no_code)
    for i in range(depth):
        m = copy_linear(m, prefix + [("hidden", i)], getattr(pt, f"fc{i + 1}"))
    return copy_linear(m, prefix + [("fc_out", None)], getattr(pt, f"fc{depth + 1}"))


def m_at(m, path):
    for attr, idx in path:
        m = getattr(m, attr)
        if idx is not None:
            m = m[idx]
    return m


def copy_fno2d(m, pt):
    m = copy_linear(m, [("fc0", None)], pt.fc0)
    m = copy_spectral(m, [("conv_in", None)], pt.conv0)
    for i in range(3):
        m = copy_spectral(m, [("convs", i)], getattr(pt, f"conv{i + 1}"))
        m = copy_pointwise_conv(m, [("ws", i)], getattr(pt, f"w{i + 1}"))
    for i in range(4):
        m = copy_pointwise_conv(m, [("bs", i)], getattr(pt, f"b{i}"))
    m = copy_spectral(m, [("conv_out", None)], pt.conv4)
    m = copy_pointwise_conv(m, [("b_out", None)], pt.b4)
    m = copy_linear(m, [("fc1", None)], pt.fc1)
    return copy_linear(m, [("fc2", None)], pt.fc2)


def compare_v2_model(root: Path, seed: int) -> bool:
    import torch
    import jax
    import jax.numpy as jnp
    import foundax as fx

    up = load_upstream(root / "elasticity" / "elas_geofno_v2.py")
    modes, width, s, B, N, M = 6, 16, 20, 2, 120, 37
    torch.manual_seed(seed)
    pt_model = up["FNO2d"](modes, modes, width, 2, 1, s1=s, s2=s).eval()
    pt_iphi = up["IPHI"](width=32).eval()

    m = fx.geofno(
        modes1=modes,
        modes2=modes,
        width=width,
        s1=s,
        s2=s,
        iphi_depth=2,
        iphi_act="gelu",
        iphi_center=(0.5, 0.5),
        key=jax.random.PRNGKey(seed),
    )
    m = copy_fno2d(m, pt_model)
    m = copy_iphi(m, [("iphi", None)], pt_iphi, depth=2)

    g = torch.Generator().manual_seed(seed + 1)
    mesh = torch.rand(B, N, 2, generator=g)
    query = torch.rand(B, M, 2, generator=g)
    with torch.no_grad():
        pt_same = pt_model(mesh, iphi=pt_iphi)
        pt_query = pt_model(mesh, x_in=mesh, x_out=query, iphi=pt_iphi)

    mesh_j, query_j = jnp.asarray(_np(mesh)), jnp.asarray(_np(query))
    out_same = jax.vmap(m)(mesh_j)
    out_query = jax.vmap(lambda c, q: m(c, out_coords=q))(mesh_j, query_j)
    ok = compare_arrays("GeoFNO v2 (query = mesh)", pt_same, out_same)
    ok &= compare_arrays("GeoFNO v2 (separate query)", pt_query, out_query)
    return ok


def compare_v1_iphi(root: Path, seed: int) -> bool:
    import torch
    import jax
    import jax.numpy as jnp
    from foundax.architectures.geofno import IPHI

    up = load_upstream(root / "elasticity" / "elas_geofno.py")
    torch.manual_seed(seed)
    pt = up["IPHI"](width=32).eval()
    g = torch.Generator().manual_seed(seed + 2)
    x = torch.rand(2, 90, 2, generator=g)
    code = torch.rand(2, 42, generator=g)

    ok = True
    for code_dim in (0, 42):
        m = IPHI(32, 3, code_dim, "tanh", (1e-4, 1e-4), key=jax.random.PRNGKey(seed))
        m = copy_iphi(m, [], pt, depth=3)
        with torch.no_grad():
            ref = pt(x, code if code_dim else None)
        xj = jnp.asarray(_np(x))
        if code_dim:
            out = jax.vmap(m)(xj, jnp.asarray(_np(code)))
        else:
            out = jax.vmap(m)(xj)
        ok &= compare_arrays(f"IPHI v1 (code_dim={code_dim})", ref, out)
    return ok


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--geofno-root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    print("=" * 70)
    print("Geo-FNO: neuraloperator/Geo-FNO (PyTorch) vs foundax")
    print(f"  upstream: {args.geofno_root}   seed: {args.seed}")
    print("=" * 70)
    results = [
        compare_v2_model(args.geofno_root, args.seed),
        compare_v1_iphi(args.geofno_root, args.seed),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
