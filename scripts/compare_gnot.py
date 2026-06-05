#!/usr/bin/env python3
"""
GNOT: HaoZhongkai/GNOT reference vs foundax Equinox port.

Tests the cross-attention transformer primitives — ``LinearAttention``,
``LinearCrossAttention``, and ``CrossAttentionBlock`` — that GNOT's
full ``CGPTNO`` is composed of.

The full upstream ``CGPTNO.forward`` requires ``dgl`` (Deep Graph
Library) for graph batching of variable-size point clouds; foundax
operates on padded tensors. We therefore bypass the graph layer and
compare the building blocks directly — that's where the actual
mathematical algorithm lives (linear attention + cross-attention +
residual block structure). The DGL batching is orchestration, not
algorithm.

Usage::
    python scripts/compare_gnot.py --gnot-root og_repos/gnot
"""

from __future__ import annotations

import argparse
import sys
from importlib.util import find_spec
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pt2eqx import compare_arrays, copy_linear, copy_layernorm


def _import_upstream(gnot_root: Path):
    """Stub ``dgl`` and ``utils.MultipleTensors`` so upstream cgpt.py
    imports without those heavy deps. We only use classes that don't
    actually invoke dgl at runtime."""
    import importlib.util
    import types

    # Stub `dgl` so `import dgl` succeeds.
    if "dgl" not in sys.modules:
        dgl_stub = types.ModuleType("dgl")
        dgl_stub.unbatch = lambda g: [g]  # not actually called in our tests
        sys.modules["dgl"] = dgl_stub

    # Stub `utils.MultipleTensors` — upstream uses this as a list wrapper.
    if "utils" not in sys.modules:
        utils_stub = types.ModuleType("utils")

        class MultipleTensors(list):
            def __init__(self, items=()):
                super().__init__(items)

        utils_stub.MultipleTensors = MultipleTensors
        sys.modules["utils"] = utils_stub

    # Stub the `models` package so `from models.mlp import MLP` resolves.
    models_root = gnot_root / "models"
    if "models" not in sys.modules:
        models_stub = types.ModuleType("models")
        models_stub.__path__ = [str(models_root)]
        sys.modules["models"] = models_stub

    def _load(qualname, path):
        spec = importlib.util.spec_from_file_location(qualname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[qualname] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("models.mlp", models_root / "mlp.py")
    cgpt = _load("models.cgpt", models_root / "cgpt.py")
    return cgpt


# ── weight transfer (upstream uses single Linear, foundax uses Linear too) ─


def transfer_linear_attention(pt_attn, eqx_attn):
    """Copy upstream LinearAttention into foundax LinearAttention."""
    eqx_attn = copy_linear(eqx_attn, [("query", None)], pt_attn.query)
    eqx_attn = copy_linear(eqx_attn, [("key", None)], pt_attn.key)
    eqx_attn = copy_linear(eqx_attn, [("value", None)], pt_attn.value)
    eqx_attn = copy_linear(eqx_attn, [("proj", None)], pt_attn.proj)
    return eqx_attn


def transfer_linear_cross_attention(pt_attn, eqx_attn):
    eqx_attn = copy_linear(eqx_attn, [("query_proj", None)], pt_attn.query)
    for i, pt_k in enumerate(pt_attn.keys):
        eqx_attn = copy_linear(eqx_attn, [("key_projs", i)], pt_k)
    for i, pt_v in enumerate(pt_attn.values):
        eqx_attn = copy_linear(eqx_attn, [("value_projs", i)], pt_v)
    eqx_attn = copy_linear(eqx_attn, [("proj", None)], pt_attn.proj)
    return eqx_attn


def transfer_block(pt_block, eqx_block):
    """Copy upstream CrossAttentionBlock into foundax CrossAttentionBlock."""
    from _pt2eqx import set_eqx_array

    eqx_block = copy_layernorm(eqx_block, [("ln1", None)], pt_block.ln1)
    for i, pt_ln in enumerate(pt_block.ln2_branch):
        eqx_block = copy_layernorm(eqx_block, [("ln2_branches", i)], pt_ln)
    eqx_block = copy_layernorm(eqx_block, [("ln3", None)], pt_block.ln3)
    eqx_block = copy_layernorm(eqx_block, [("ln4", None)], pt_block.ln4)
    eqx_block = copy_layernorm(eqx_block, [("ln5", None)], pt_block.ln5)
    # cross_attn: rewrite each sub-array
    eqx_block = set_eqx_array(
        eqx_block,
        [("cross_attn", None), ("query_proj", None), ("weight", None)],
        pt_block.crossattn.query.weight.detach().cpu().numpy(),
    )
    eqx_block = set_eqx_array(
        eqx_block,
        [("cross_attn", None), ("query_proj", None), ("bias", None)],
        pt_block.crossattn.query.bias.detach().cpu().numpy(),
    )
    for i in range(len(pt_block.crossattn.keys)):
        eqx_block = set_eqx_array(
            eqx_block,
            [("cross_attn", None), ("key_projs", i), ("weight", None)],
            pt_block.crossattn.keys[i].weight.detach().cpu().numpy(),
        )
        eqx_block = set_eqx_array(
            eqx_block,
            [("cross_attn", None), ("key_projs", i), ("bias", None)],
            pt_block.crossattn.keys[i].bias.detach().cpu().numpy(),
        )
        eqx_block = set_eqx_array(
            eqx_block,
            [("cross_attn", None), ("value_projs", i), ("weight", None)],
            pt_block.crossattn.values[i].weight.detach().cpu().numpy(),
        )
        eqx_block = set_eqx_array(
            eqx_block,
            [("cross_attn", None), ("value_projs", i), ("bias", None)],
            pt_block.crossattn.values[i].bias.detach().cpu().numpy(),
        )
    eqx_block = set_eqx_array(
        eqx_block,
        [("cross_attn", None), ("proj", None), ("weight", None)],
        pt_block.crossattn.proj.weight.detach().cpu().numpy(),
    )
    eqx_block = set_eqx_array(
        eqx_block,
        [("cross_attn", None), ("proj", None), ("bias", None)],
        pt_block.crossattn.proj.bias.detach().cpu().numpy(),
    )
    # self_attn
    for name in ["query", "key", "value", "proj"]:
        pt_layer = getattr(pt_block.selfattn, name)
        # foundax field names: query/key/value/proj
        eqx_block = set_eqx_array(
            eqx_block,
            [("self_attn", None), (name, None), ("weight", None)],
            pt_layer.weight.detach().cpu().numpy(),
        )
        eqx_block = set_eqx_array(
            eqx_block,
            [("self_attn", None), (name, None), ("bias", None)],
            pt_layer.bias.detach().cpu().numpy(),
        )
    # FFNs — upstream mlp1[0], mlp1[2] are the Linear layers
    eqx_block = set_eqx_array(
        eqx_block,
        [("ffn1", None), ("fc1", None), ("weight", None)],
        pt_block.mlp1[0].weight.detach().cpu().numpy(),
    )
    eqx_block = set_eqx_array(
        eqx_block,
        [("ffn1", None), ("fc1", None), ("bias", None)],
        pt_block.mlp1[0].bias.detach().cpu().numpy(),
    )
    eqx_block = set_eqx_array(
        eqx_block,
        [("ffn1", None), ("fc2", None), ("weight", None)],
        pt_block.mlp1[2].weight.detach().cpu().numpy(),
    )
    eqx_block = set_eqx_array(
        eqx_block,
        [("ffn1", None), ("fc2", None), ("bias", None)],
        pt_block.mlp1[2].bias.detach().cpu().numpy(),
    )
    eqx_block = set_eqx_array(
        eqx_block,
        [("ffn2", None), ("fc1", None), ("weight", None)],
        pt_block.mlp2[0].weight.detach().cpu().numpy(),
    )
    eqx_block = set_eqx_array(
        eqx_block,
        [("ffn2", None), ("fc1", None), ("bias", None)],
        pt_block.mlp2[0].bias.detach().cpu().numpy(),
    )
    eqx_block = set_eqx_array(
        eqx_block,
        [("ffn2", None), ("fc2", None), ("weight", None)],
        pt_block.mlp2[2].weight.detach().cpu().numpy(),
    )
    eqx_block = set_eqx_array(
        eqx_block,
        [("ffn2", None), ("fc2", None), ("bias", None)],
        pt_block.mlp2[2].bias.detach().cpu().numpy(),
    )
    return eqx_block


# ── drivers ────────────────────────────────────────────────────────────────


def compare_linear_attention(gnot_root: Path, seed: int) -> bool:
    import torch
    import jax
    from foundax.architectures.gnot import LinearAttention

    cgpt = _import_upstream(gnot_root)
    n_embd, n_head, T = 32, 4, 12
    torch.manual_seed(seed)
    config = cgpt.GPTConfig(n_embd=n_embd, n_head=n_head, attn_type="linear")
    pt = cgpt.LinearAttention(config).eval()

    eqx_attn = LinearAttention(
        n_embd=n_embd, n_head=n_head, attn_type="l1", key=jax.random.PRNGKey(seed)
    )
    eqx_attn = transfer_linear_attention(pt, eqx_attn)

    x = torch.randn(1, T, n_embd, dtype=torch.float32)
    with torch.no_grad():
        pt_out = pt(x)
    eqx_out = eqx_attn(x.numpy())
    return compare_arrays("LinearAttention", pt_out, eqx_out)


def compare_linear_cross_attention(gnot_root: Path, seed: int) -> bool:
    import torch
    import jax
    from foundax.architectures.gnot import LinearCrossAttention

    cgpt = _import_upstream(gnot_root)
    n_embd, n_head, n_inputs = 32, 4, 2
    T1, T2 = 10, 16
    torch.manual_seed(seed)
    config = cgpt.GPTConfig(
        n_embd=n_embd, n_head=n_head, n_inputs=n_inputs, attn_type="linear"
    )
    pt = cgpt.LinearCrossAttention(config).eval()

    eqx_attn = LinearCrossAttention(
        n_embd=n_embd,
        n_head=n_head,
        n_inputs=n_inputs,
        key=jax.random.PRNGKey(seed),
    )
    eqx_attn = transfer_linear_cross_attention(pt, eqx_attn)

    x = torch.randn(1, T1, n_embd, dtype=torch.float32)
    ys_pt = [torch.randn(1, T2, n_embd, dtype=torch.float32) for _ in range(n_inputs)]
    ys_jax = [y.numpy() for y in ys_pt]
    with torch.no_grad():
        pt_out = pt(x, ys_pt)
    eqx_out = eqx_attn(x.numpy(), ys_jax)
    return compare_arrays("LinearCrossAttention", pt_out, eqx_out)


def compare_cross_attention_block(gnot_root: Path, seed: int) -> bool:
    import torch
    import jax
    from foundax.architectures.gnot import CrossAttentionBlock, GPTConfig

    cgpt = _import_upstream(gnot_root)
    n_embd, n_head, n_inputs, n_inner = 32, 4, 2, 64
    T1, T2 = 10, 16

    torch.manual_seed(seed)
    pt_config = cgpt.GPTConfig(
        n_embd=n_embd,
        n_head=n_head,
        n_inputs=n_inputs,
        n_inner=n_inner,
        attn_type="linear",
        act="gelu",
    )
    pt = cgpt.CrossAttentionBlock(pt_config).eval()

    eqx_config = GPTConfig(
        n_embd=n_embd,
        n_head=n_head,
        n_inputs=n_inputs,
        n_inner=n_inner,
        attn_type="linear",
        act="gelu",
    )
    eqx_block = CrossAttentionBlock(eqx_config, key=jax.random.PRNGKey(seed))
    eqx_block = transfer_block(pt, eqx_block)

    x = torch.randn(1, T1, n_embd, dtype=torch.float32)
    ys_pt = [torch.randn(1, T2, n_embd, dtype=torch.float32) for _ in range(n_inputs)]

    # Upstream forward signature: forward(x, y) where y is MultipleTensors([y0, y1])
    from utils import MultipleTensors

    with torch.no_grad():
        pt_out = pt(x, MultipleTensors(ys_pt))
    eqx_out = eqx_block(x.numpy(), [y.numpy() for y in ys_pt])
    return compare_arrays("CrossAttentionBlock", pt_out, eqx_out)


def run_structural_check(seed: int) -> int:
    import jax
    import jax.numpy as jnp
    import foundax as fx

    print("[GNOT] Structural check (JAX only)")
    m = fx.cgptno(
        trunk_size=2,
        branch_sizes=[2],
        output_size=1,
        n_layers=2,
        n_hidden=32,
        n_head=4,
        key=jax.random.PRNGKey(seed),
    )
    x_trunk = jax.random.normal(jax.random.PRNGKey(seed + 1), (1, 16, 2))
    x_branch = jax.random.normal(jax.random.PRNGKey(seed + 2), (1, 8, 2))
    y = m(x_trunk, [x_branch])
    print(f"  output shape: {y.shape}, finite: {bool(jnp.all(jnp.isfinite(y)))}")
    return 0 if bool(jnp.all(jnp.isfinite(y))) else 1


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gnot-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "og_repos" / "gnot",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if (
        find_spec("torch") is None
        or not (args.gnot_root / "models" / "cgpt.py").exists()
    ):
        print("torch or upstream missing — JAX-only structural check.")
        return run_structural_check(args.seed)

    print("=" * 70)
    print("GNOT: HaoZhongkai/GNOT reference vs foundax Equinox port")
    print(f"  reference: {args.gnot_root}")
    print(f"  seed:      {args.seed}")
    print("=" * 70)
    ok = [
        compare_linear_attention(args.gnot_root, args.seed),
        compare_linear_cross_attention(args.gnot_root, args.seed),
        compare_cross_attention_block(args.gnot_root, args.seed),
    ]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
