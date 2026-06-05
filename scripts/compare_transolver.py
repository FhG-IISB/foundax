#!/usr/bin/env python3
"""
Transolver: PyTorch reference vs foundax Equinox port (forward equivalence).

We don't have a published canonical Transolver checkpoint to use as a
reference (the authors publish per-benchmark weights), so this script:

  1. Seeds PyTorch and instantiates the reference ``thuml/Transolver``
     model (cloned to ``og_repos/transolver/``).
  2. Instantiates the foundax Equinox model with matching config.
  3. Tensor-by-tensor copies weights PT → EQX (handling Conv2d/Conv3d
     NCHW ↔ NHWC dim reordering).
  4. Runs both forwards on the same fixed input and compares.

Covers ``TransolverIrregular`` and ``TransolverStructured2D`` (the two
configurations most users will actually call). Falls back to a
JAX-only structural check if PyTorch / the reference repo is missing.

Usage::

    python scripts/compare_transolver.py --transolver-root og_repos/transolver
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pt2eqx import (
    compare_arrays as _compare_arrays,
    copy_conv2d as _copy_conv2d,
    copy_layernorm as _copy_layernorm,
    copy_linear as _copy_linear,
    set_eqx_array as _set_eqx_array,
)


def _copy_mlp_two_linear(eqx_module, eqx_path_prefix, pt_mlp):
    """Reference MLP(n_layers=0) is Sequential(Linear, act) → Linear.

    Map ``linear_pre`` → ``pre`` and ``linear_post`` → ``post`` to match
    foundax's ``_Lift`` / ``TransolverFFN`` structure."""
    pre = pt_mlp.linear_pre[0]  # Sequential(Linear, act)[0]
    post = pt_mlp.linear_post
    eqx_module = _copy_linear(eqx_module, eqx_path_prefix + [("pre", None)], pre)
    eqx_module = _copy_linear(eqx_module, eqx_path_prefix + [("post", None)], post)
    return eqx_module


def _copy_physics_attention(eqx_module, prefix, pt_attn, is_structured):
    """Copy a PhysicsAttention* sub-module."""
    if is_structured:
        eqx_module = _copy_conv2d(
            eqx_module, prefix + [("in_project_x", None)], pt_attn.in_project_x
        )
        eqx_module = _copy_conv2d(
            eqx_module, prefix + [("in_project_fx", None)], pt_attn.in_project_fx
        )
    else:
        eqx_module = _copy_linear(
            eqx_module, prefix + [("in_project_x", None)], pt_attn.in_project_x
        )
        eqx_module = _copy_linear(
            eqx_module, prefix + [("in_project_fx", None)], pt_attn.in_project_fx
        )
    eqx_module = _copy_linear(
        eqx_module, prefix + [("in_project_slice", None)], pt_attn.in_project_slice
    )
    eqx_module = _copy_linear(eqx_module, prefix + [("to_q", None)], pt_attn.to_q)
    eqx_module = _copy_linear(eqx_module, prefix + [("to_k", None)], pt_attn.to_k)
    eqx_module = _copy_linear(eqx_module, prefix + [("to_v", None)], pt_attn.to_v)
    # to_out is Sequential(Linear, Dropout)
    eqx_module = _copy_linear(
        eqx_module, prefix + [("to_out", None)], pt_attn.to_out[0]
    )
    # temperature: PT is (1, H, 1, 1); EQX is (H, 1, 1)
    t = pt_attn.temperature.detach().cpu().numpy().squeeze(0)
    eqx_module = _set_eqx_array(eqx_module, prefix + [("temperature", None)], t)
    return eqx_module


def transfer_transolver_weights(pt_model, eqx_model, is_structured):
    """Copy every parameter from the PyTorch Transolver Model into our
    Equinox Transolver{Irregular,Structured2D}.

    The reference's last block has an inline output head (ln_3 + mlp2);
    in our port we hoisted that to ``head_ln`` + ``head`` outside the
    block list, so we copy from ``blocks[-1]`` accordingly.
    """
    # Lift (PT: preprocess MLP, EQX: lift _Lift).
    eqx_model = _copy_mlp_two_linear(eqx_model, [("lift", None)], pt_model.preprocess)

    # placeholder
    eqx_model = _set_eqx_array(
        eqx_model,
        [("placeholder", None)],
        pt_model.placeholder.detach().cpu().numpy(),
    )

    n_blocks = len(pt_model.blocks)
    for i, pt_block in enumerate(pt_model.blocks):
        prefix = [("blocks", i)]
        eqx_model = _copy_layernorm(eqx_model, prefix + [("ln_1", None)], pt_block.ln_1)
        eqx_model = _copy_layernorm(eqx_model, prefix + [("ln_2", None)], pt_block.ln_2)
        eqx_model = _copy_physics_attention(
            eqx_model, prefix + [("physics_attn", None)], pt_block.Attn, is_structured
        )
        eqx_model = _copy_mlp_two_linear(
            eqx_model, prefix + [("ffn", None)], pt_block.mlp
        )
        # Last block's ln_3 / mlp2 → our hoisted head_ln / head.
        if i == n_blocks - 1 and getattr(pt_block, "last_layer", False):
            eqx_model = _copy_layernorm(eqx_model, [("head_ln", None)], pt_block.ln_3)
            eqx_model = _copy_linear(eqx_model, [("head", None)], pt_block.mlp2)

    return eqx_model


# ── comparison drivers ─────────────────────────────────────────────────────


def compare_irregular(transolver_root: Path, seed: int) -> bool:
    import torch

    sys.path.insert(0, str(transolver_root / "PDE-Solving-StandardBenchmark"))
    from model.Transolver_Irregular_Mesh import Model as PtIrregular

    space_dim, fun_dim, out_dim = 2, 1, 1
    hidden_dim, n_layers, n_heads, n_slices = 32, 2, 4, 8
    N = 64

    torch.manual_seed(seed)
    pt = PtIrregular(
        space_dim=space_dim,
        n_layers=n_layers,
        n_hidden=hidden_dim,
        dropout=0.0,
        n_head=n_heads,
        mlp_ratio=2,
        fun_dim=fun_dim,
        out_dim=out_dim,
        slice_num=n_slices,
        unified_pos=False,
    ).eval()

    import jax
    from foundax.architectures.transolver import TransolverIrregular

    eqx_model = TransolverIrregular(
        space_dim=space_dim,
        fun_dim=fun_dim,
        out_features=out_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_slices=n_slices,
        mlp_ratio=2,
        dropout=0.0,
        key=jax.random.PRNGKey(seed),
    )
    eqx_model = transfer_transolver_weights(pt, eqx_model, is_structured=False)

    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal((1, N, space_dim)).astype(np.float32)
    fx_np = rng.standard_normal((1, N, fun_dim)).astype(np.float32)

    with torch.no_grad():
        pt_out = pt(torch.from_numpy(x_np), torch.from_numpy(fx_np))
    eqx_out = eqx_model(x_np[0], fx_np[0])
    return _compare_arrays("Transolver Irregular", pt_out[0], eqx_out)


def compare_structured_2d(transolver_root: Path, seed: int) -> bool:
    import torch

    sys.path.insert(0, str(transolver_root / "PDE-Solving-StandardBenchmark"))
    from model.Transolver_Structured_Mesh_2D import Model as PtStructured2D

    space_dim, fun_dim, out_dim = 2, 1, 1
    hidden_dim, n_layers, n_heads, n_slices = 32, 2, 4, 8
    H, W = 16, 16

    torch.manual_seed(seed)
    pt = PtStructured2D(
        space_dim=space_dim,
        n_layers=n_layers,
        n_hidden=hidden_dim,
        dropout=0.0,
        n_head=n_heads,
        mlp_ratio=2,
        fun_dim=fun_dim,
        out_dim=out_dim,
        slice_num=n_slices,
        unified_pos=False,
        H=H,
        W=W,
    ).eval()

    import jax
    from foundax.architectures.transolver import TransolverStructured2D

    eqx_model = TransolverStructured2D(
        space_dim=space_dim,
        fun_dim=fun_dim,
        out_features=out_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_slices=n_slices,
        mlp_ratio=2,
        dropout=0.0,
        kernel=3,
        key=jax.random.PRNGKey(seed),
    )
    eqx_model = transfer_transolver_weights(pt, eqx_model, is_structured=True)

    rng = np.random.default_rng(seed)
    # PT layout: (B, N, C) flattened over H*W; EQX layout: (H, W, C)
    x_np = rng.standard_normal((1, H * W, space_dim)).astype(np.float32)
    fx_np = rng.standard_normal((1, H * W, fun_dim)).astype(np.float32)

    with torch.no_grad():
        pt_out = pt(torch.from_numpy(x_np), torch.from_numpy(fx_np))
    eqx_out = eqx_model(
        x_np[0].reshape(H, W, space_dim),
        fx_np[0].reshape(H, W, fun_dim),
    )
    # Reshape EQX output back to flat to match PT layout for comparison.
    eqx_out_flat = np.asarray(eqx_out).reshape(H * W, out_dim)
    return _compare_arrays("Transolver Structured2D", pt_out[0], eqx_out_flat)


def run_structural_check(seed: int) -> int:
    """JAX-only fallback when PyTorch / reference repo isn't available."""
    import jax
    import jax.numpy as jnp
    from foundax.architectures.transolver import TransolverIrregular

    print("[Transolver] Structural check (JAX only, random weights)")
    model = TransolverIrregular(
        space_dim=2,
        fun_dim=1,
        out_features=1,
        hidden_dim=32,
        n_layers=2,
        n_heads=4,
        n_slices=8,
        key=jax.random.PRNGKey(seed),
    )
    x = jax.random.normal(jax.random.PRNGKey(seed + 1), (64, 2))
    fx = jax.random.normal(jax.random.PRNGKey(seed + 2), (64, 1))
    y = model(x, fx)
    print(f"  output shape: {y.shape}, finite: {bool(jnp.all(jnp.isfinite(y)))}")
    return 0 if bool(jnp.all(jnp.isfinite(y))) else 1


# ── entry point ────────────────────────────────────────────────────────────


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--transolver-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "og_repos" / "transolver",
        help="Path to a clone of thuml/Transolver",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        import torch  # noqa: F401
    except ImportError:
        print("PyTorch not installed — running JAX-only structural check.")
        return run_structural_check(args.seed)

    bench = args.transolver_root / "PDE-Solving-StandardBenchmark"
    if not bench.exists():
        print(f"Reference repo missing at {bench} — running structural check.")
        return run_structural_check(args.seed)

    print("=" * 70)
    print("Transolver: PyTorch reference vs foundax Equinox port")
    print(f"  reference: {args.transolver_root}")
    print(f"  seed:      {args.seed}")
    print("=" * 70)

    ok1 = compare_irregular(args.transolver_root, args.seed)
    ok2 = compare_structured_2d(args.transolver_root, args.seed)
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
