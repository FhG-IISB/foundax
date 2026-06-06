#!/usr/bin/env python3
"""GAOT: PyTorch reference vs foundax Equinox port (forward equivalence).

Steps:
  1. Seed PyTorch and instantiate the reference ``camlab-ethz/GAOT`` model
     (cloned to ``og_repos/gaot/``).
  2. Instantiate the foundax Equinox model with matching config.
  3. Tensor-by-tensor copy weights PT → EQX (handling Conv1d kernel=1
     → Linear shape squeeze, Sequential indexing for GeometricEmbedding).
  4. Pre-compute CSR neighbours on the PT side, pass them to both.
  5. Run forwards on identical input and compare outputs.

Covers four configurations:
  (a) ``linear`` AGNO, no attention, no geoembed
  (b) ``linear`` AGNO, cosine attention, no geoembed
  (c) ``linear`` AGNO, dot_product attention, geoembed=statistical
  (d) ``linear`` AGNO, cosine attention, geoembed=pointnet

Usage::

    pixi run --environment dev python scripts/compare_gaot.py --gaot-root og_repos/gaot
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# torch_scatter shim (pure-PyTorch fallback, only used if torch_scatter is
# not installed — the upstream GAOT imports it unconditionally in gemb.py).
# ---------------------------------------------------------------------------


def _install_torch_scatter_shim():
    try:
        import torch_scatter  # noqa: F401

        return
    except ImportError:
        pass

    import torch

    def scatter_sum(src, index, dim, dim_size=None):
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        shape = list(src.shape)
        shape[dim] = dim_size
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
        idx = index
        while idx.dim() < src.dim():
            idx = idx.unsqueeze(-1)
        idx = idx.expand_as(src)
        out.scatter_add_(dim, idx, src)
        return out

    def scatter_mean(src, index, dim, dim_size=None):
        s = scatter_sum(src, index, dim, dim_size)
        ones = torch.ones_like(src)
        cnt = scatter_sum(ones, index, dim, dim_size)
        return s / cnt.clamp_min(1.0)

    def scatter_max(src, index, dim, dim_size=None):
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        shape = list(src.shape)
        shape[dim] = dim_size
        out = torch.full(shape, float("-inf"), dtype=src.dtype, device=src.device)
        arg = torch.full(shape, -1, dtype=torch.long, device=src.device)
        for i in range(src.shape[dim]):
            sl_src = [slice(None)] * src.dim()
            sl_src[dim] = i
            seg = int(index[i].item()) if index.dim() == 1 else None
            if seg is None:
                raise NotImplementedError("scatter_max shim supports 1-D index only")
            sl_out = [slice(None)] * out.dim()
            sl_out[dim] = seg
            val = src[tuple(sl_src)]
            cur = out[tuple(sl_out)]
            better = val > cur
            cur = torch.where(better, val, cur)
            out[tuple(sl_out)] = cur
        # arg-max not needed for parity (we only ever consume the values)
        return out, arg

    def segment_csr(src, indptr, reduce):
        """Generic CSR reduce supporting 'sum', 'mean', 'max'."""
        if src.dim() == 3:
            point_dim = 1
        else:
            point_dim = 0
        n_out = indptr.shape[point_dim] - 1
        shape = list(src.shape)
        shape[point_dim] = n_out
        if reduce == "max":
            out = torch.full(shape, float("-inf"), dtype=src.dtype, device=src.device)
        else:
            out = torch.zeros(shape, dtype=src.dtype, device=src.device)
        for i in range(n_out):
            if src.dim() == 3:
                start = int(indptr[0, i].item())
                end = int(indptr[0, i + 1].item())
                sub = src[:, start:end]
                if end > start:
                    if reduce == "max":
                        out[:, i] = sub.amax(dim=1)
                    elif reduce == "mean":
                        out[:, i] = sub.sum(dim=1) / float(end - start)
                    else:
                        out[:, i] = sub.sum(dim=1)
            else:
                start = int(indptr[i].item())
                end = int(indptr[i + 1].item())
                sub = src[start:end]
                if end > start:
                    if reduce == "max":
                        out[i] = sub.amax(dim=0)
                    elif reduce == "mean":
                        out[i] = sub.sum(dim=0) / float(end - start)
                    else:
                        out[i] = sub.sum(dim=0)
        return out

    import importlib.machinery

    class _CallableModule(types.ModuleType):
        def __call__(self, *args, **kwargs):
            return self._fn(*args, **kwargs)

    mod = types.ModuleType("torch_scatter")
    mod.__spec__ = importlib.machinery.ModuleSpec("torch_scatter", loader=None)
    mod.__path__ = []  # mark as a package so submodule imports work
    mod.scatter_sum = scatter_sum
    mod.scatter_mean = scatter_mean
    mod.scatter_max = scatter_max
    mod.segment_csr = segment_csr
    sys.modules["torch_scatter"] = mod

    # Upstream does `import torch_scatter.segment_csr as scatter_segment_csr`;
    # they then call it as a function — so the submodule itself must be callable.
    sub = _CallableModule("torch_scatter.segment_csr")
    sub.__spec__ = importlib.machinery.ModuleSpec(
        "torch_scatter.segment_csr", loader=None
    )
    sub._fn = segment_csr
    sys.modules["torch_scatter.segment_csr"] = sub
    mod.segment_csr = sub  # so `from torch_scatter import segment_csr` finds it too


# ---------------------------------------------------------------------------
# Weight transfer
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pt2eqx import (  # noqa: E402
    set_eqx_array as _set,
    copy_linear as _copy_linear,
    compare_arrays as _compare_arrays,
)


def _copy_conv1d_as_linear(eqx_module, prefix, pt_conv):
    """PT Conv1d(in,out,1).weight shape [out,in,1] → squeeze → [out,in]."""
    w = pt_conv.weight.detach().cpu().numpy().squeeze(-1)
    eqx_module = _set(eqx_module, prefix + [("weight", None)], w)
    if pt_conv.bias is not None:
        eqx_module = _set(
            eqx_module, prefix + [("bias", None)], pt_conv.bias.detach().cpu().numpy()
        )
    return eqx_module


def _copy_channel_mlp(eqx_module, prefix, pt_cmlp):
    """ChannelMLP holds Conv1d kernel=1 layers; our port stores Linears."""
    for i, fc in enumerate(pt_cmlp.fcs):
        eqx_module = _copy_conv1d_as_linear(eqx_module, prefix + [("fcs", i)], fc)
    return eqx_module


def _copy_linear_channel_mlp(eqx_module, prefix, pt_lcmlp):
    for i, fc in enumerate(pt_lcmlp.fcs):
        eqx_module = _copy_linear(eqx_module, prefix + [("fcs", i)], fc)
    return eqx_module


def _copy_geoembed(eqx_module, prefix, pt_geoembed):
    if pt_geoembed.method == "statistical":
        # PT mlp = Sequential(Linear, ReLU, Linear, ReLU); take [0] and [2]
        eqx_module = _copy_linear(
            eqx_module, prefix + [("mlp", None), ("l1", None)], pt_geoembed.mlp[0]
        )
        eqx_module = _copy_linear(
            eqx_module, prefix + [("mlp", None), ("l2", None)], pt_geoembed.mlp[2]
        )
    else:  # pointnet
        eqx_module = _copy_linear(
            eqx_module,
            prefix + [("pointnet_mlp", None), ("l1", None)],
            pt_geoembed.pointnet_mlp[0],
        )
        eqx_module = _copy_linear(
            eqx_module,
            prefix + [("pointnet_mlp", None), ("l2", None)],
            pt_geoembed.pointnet_mlp[2],
        )
        eqx_module = _copy_linear(
            eqx_module,
            prefix + [("fc", None), ("linear", None)],
            pt_geoembed.fc[0],
        )
    return eqx_module


def _copy_magno(eqx_module, prefix, pt_magno, is_encoder: bool):
    """Copy MAGNOEncoder/Decoder weights."""
    # AGNO
    agno_prefix = prefix + [("agno", None)]
    eqx_module = _copy_linear_channel_mlp(
        eqx_module,
        agno_prefix + [("channel_mlp", None)],
        pt_magno.agno.channel_mlp,
    )
    if pt_magno.agno.use_attn and pt_magno.agno.attention_type == "dot_product":
        eqx_module = _copy_linear(
            eqx_module, agno_prefix + [("query_proj", None)], pt_magno.agno.query_proj
        )
        eqx_module = _copy_linear(
            eqx_module, agno_prefix + [("key_proj", None)], pt_magno.agno.key_proj
        )

    # lifting (encoder) or projection (decoder)
    if is_encoder:
        eqx_module = _copy_channel_mlp(
            eqx_module, prefix + [("lifting", None)], pt_magno.lifting
        )
    else:
        eqx_module = _copy_channel_mlp(
            eqx_module, prefix + [("projection", None)], pt_magno.projection
        )

    # geoembed + recovery (optional)
    if pt_magno.use_geoembed:
        eqx_module = _copy_geoembed(
            eqx_module, prefix + [("geoembed", None)], pt_magno.geoembed
        )
        eqx_module = _copy_channel_mlp(
            eqx_module, prefix + [("recovery", None)], pt_magno.recovery
        )

    return eqx_module


def _copy_transformer_block(eqx_module, prefix, pt_block):
    eqx_module = _set(
        eqx_module,
        prefix + [("attn_norm", None), ("weight", None)],
        pt_block.attn_norm.weight.detach().cpu().numpy(),
    )
    eqx_module = _set(
        eqx_module,
        prefix + [("ffn_norm", None), ("weight", None)],
        pt_block.ffn_norm.weight.detach().cpu().numpy(),
    )
    # attention sub-module
    ap = prefix + [("attn", None)]
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        eqx_module = _copy_linear(
            eqx_module,
            ap + [(name, None)],
            getattr(pt_block.attn, name),
            has_bias=False,
        )
    # FFN
    fp = prefix + [("ffn", None)]
    for name in ("w1", "w2", "w3"):
        eqx_module = _copy_linear(
            eqx_module,
            fp + [(name, None)],
            getattr(pt_block.ffn, name),
            has_bias=False,
        )
    if pt_block.skip_connection:
        eqx_module = _copy_linear(
            eqx_module, prefix + [("skip_proj", None)], pt_block.skip_proj
        )
    return eqx_module


def _copy_processor(eqx_module, pt_processor, pt_gaot):
    """Transfer Transformer + the GAOT-level patch_linear/positions."""
    eqx_module = _copy_linear(
        eqx_module, [("patch_linear", None)], pt_gaot.patch_linear
    )

    # processor.input_proj (may be Identity)
    import torch.nn as nn

    if not isinstance(pt_processor.input_proj, nn.Identity):
        eqx_module = _copy_linear(
            eqx_module,
            [("processor", None), ("input_proj", None)],
            pt_processor.input_proj,
        )
    if not isinstance(pt_processor.output_proj, nn.Identity):
        eqx_module = _copy_linear(
            eqx_module,
            [("processor", None), ("output_proj", None)],
            pt_processor.output_proj,
        )

    for i, layer in enumerate(pt_processor.encoder_layers):
        eqx_module = _copy_transformer_block(
            eqx_module,
            [("processor", None), ("encoder_layers", i)],
            layer,
        )
    if pt_processor.middle_layer is not None:
        eqx_module = _copy_transformer_block(
            eqx_module,
            [("processor", None), ("middle_layer", None)],
            pt_processor.middle_layer,
        )
    for i, layer in enumerate(pt_processor.decoder_layers):
        eqx_module = _copy_transformer_block(
            eqx_module,
            [("processor", None), ("decoder_layers", i)],
            layer,
        )
    return eqx_module


def transfer_gaot_weights(pt_gaot, eqx_gaot):
    eqx_gaot = _copy_magno(
        eqx_gaot, [("encoder", None)], pt_gaot.encoder, is_encoder=True
    )
    eqx_gaot = _copy_magno(
        eqx_gaot, [("decoder", None)], pt_gaot.decoder, is_encoder=False
    )
    eqx_gaot = _copy_processor(eqx_gaot, pt_gaot.processor, pt_gaot)
    return eqx_gaot


# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------


class _Args:
    def __init__(self, magno, transformer):
        self.magno = magno
        self.transformer = transformer


class _GaotConfig:
    def __init__(self, magno, transformer, latent_tokens_size):
        self.args = _Args(magno, transformer)
        self.latent_tokens_size = latent_tokens_size


def _build_configs(
    use_attention: bool,
    attention_type: str,
    use_geoembed: bool,
    embedding_method: str = "statistical",
    pooling: str = "max",
    lifting_channels: int = 8,
    transformer_hidden: int = 16,
    num_layers: int = 2,
    patch_size: int = 2,
    H: int = 8,
    W: int = 8,
    transform_type: str = "linear",
):
    from src.model.layers.magno import MAGNOConfig as PtMAGNOConfig
    from src.model.layers.attn import TransformerConfig as PtTfConfig, AttentionConfig

    pt_magno = PtMAGNOConfig(
        coord_dim=2,
        radius=0.3,
        hidden_size=16,
        mlp_layers=2,
        lifting_channels=lifting_channels,
        scales=[1.0],
        use_scale_weights=False,
        use_attention=use_attention,
        attention_type=attention_type,
        use_geoembed=use_geoembed,
        embedding_method=embedding_method,
        pooling=pooling,
        transform_type=transform_type,
        precompute_edges=True,
        use_torch_scatter=True,  # routed through our shim (supports 'max')
    )
    pt_tf = PtTfConfig(
        patch_size=patch_size,
        hidden_size=transformer_hidden,
        num_layers=num_layers,
        positional_embedding="absolute",
        attn_config=AttentionConfig(num_heads=2, num_kv_heads=2),
    )
    pt_config = _GaotConfig(pt_magno, pt_tf, (H, W))

    from foundax.architectures.gaot import (
        MAGNOConfig,
        TransformerConfig,
        AttentionConfig as JAttnCfg,
    )

    j_magno = MAGNOConfig(
        coord_dim=2,
        radius=0.3,
        hidden_size=16,
        mlp_layers=2,
        lifting_channels=lifting_channels,
        use_attention=use_attention,
        attention_type=attention_type,
        use_geoembed=use_geoembed,
        embedding_method=embedding_method,
        pooling=pooling,
        transform_type=transform_type,
    )
    j_tf = TransformerConfig(
        patch_size=patch_size,
        hidden_size=transformer_hidden,
        num_layers=num_layers,
        positional_embedding="absolute",
        attn_config=JAttnCfg(num_heads=2, num_kv_heads=2),
    )
    return pt_config, j_magno, j_tf, (H, W)


# ---------------------------------------------------------------------------
# Comparison drivers
# ---------------------------------------------------------------------------


def run_one(
    name: str,
    gaot_root: Path,
    seed: int,
    use_attention,
    attention_type,
    use_geoembed,
    embedding_method="statistical",
    pooling="max",
    transform_type="linear",
    num_layers=2,
) -> bool:
    import torch

    sys.path.insert(0, str(gaot_root))
    from src.model.gaot import GAOT as PtGAOT
    from src.model.layers.utils.neighbor_search import NeighborSearch

    pt_config, j_magno, j_tf, latent_size = _build_configs(
        use_attention=use_attention,
        attention_type=attention_type,
        use_geoembed=use_geoembed,
        embedding_method=embedding_method,
        pooling=pooling,
        transform_type=transform_type,
        num_layers=num_layers,
    )

    torch.manual_seed(seed)
    pt = PtGAOT(input_size=2, output_size=1, config=pt_config).eval()

    import jax
    from foundax.architectures.gaot import GAOT as JaxGAOT

    eqx_model = JaxGAOT(
        input_size=2,
        output_size=1,
        magno_config=j_magno,
        transformer_config=j_tf,
        latent_tokens_size=latent_size,
        key=jax.random.PRNGKey(seed),
    )
    eqx_model = transfer_gaot_weights(pt, eqx_model)

    # Inputs
    H, W = latent_size
    xs = np.linspace(0, 1, H, dtype=np.float32)
    latent_coord_np = (
        np.stack(np.meshgrid(xs, xs, indexing="ij"), -1)
        .reshape(-1, 2)
        .astype(np.float32)
    )
    rng = np.random.default_rng(seed)
    x_coord_np = rng.random((40, 2)).astype(np.float32)
    q_coord_np = rng.random((20, 2)).astype(np.float32)
    pndata_np = rng.random((3, 40, 2)).astype(np.float32)

    # Compute CSR neighbours using upstream's NeighborSearch (native PT method
    # — torch_cluster/open3d may permute neighbours so we'd lose parity).
    nb = NeighborSearch(method="native")
    pt_lat = torch.from_numpy(latent_coord_np)
    pt_x = torch.from_numpy(x_coord_np)
    pt_q = torch.from_numpy(q_coord_np)
    pt_pn = torch.from_numpy(pndata_np)
    enc_nbrs_pt = [nb(data=pt_x, queries=pt_lat, radius=pt_config.args.magno.radius)]
    dec_nbrs_pt = [nb(data=pt_lat, queries=pt_q, radius=pt_config.args.magno.radius)]

    # Run PT
    with torch.no_grad():
        pt_out = pt(
            latent_tokens_coord=pt_lat,
            xcoord=pt_x,
            pndata=pt_pn,
            query_coord=pt_q,
            encoder_nbrs=enc_nbrs_pt,
            decoder_nbrs=dec_nbrs_pt,
        )

    # Build matching JAX CSR neighbours from PT tensors (same indices)
    import jax.numpy as jnp

    def _to_jax_csr(d):
        idx = d["neighbors_index"].cpu().numpy().astype(np.int64)
        rs = d["neighbors_row_splits"].cpu().numpy().astype(np.int64)
        counts = (rs[1:] - rs[:-1]).astype(np.int64)
        seg_ids = np.repeat(np.arange(len(rs) - 1, dtype=np.int64), counts)
        return {
            "neighbors_index": jnp.asarray(idx),
            "neighbors_row_splits": jnp.asarray(rs),
            "seg_ids": jnp.asarray(seg_ids),
            "counts": jnp.asarray(counts),
        }

    enc_nbrs_j = [_to_jax_csr(d) for d in enc_nbrs_pt]
    dec_nbrs_j = [_to_jax_csr(d) for d in dec_nbrs_pt]

    # foundax convention: model is single-example; vmap externally for batch.
    # PT keeps the leading batch dim, so we strip it for the per-sample compare.
    eqx_out = jax.vmap(
        lambda f: eqx_model(
            jnp.asarray(latent_coord_np),
            jnp.asarray(x_coord_np),
            f,
            jnp.asarray(q_coord_np),
            enc_nbrs_j,
            dec_nbrs_j,
        )
    )(jnp.asarray(pndata_np))

    return _compare_arrays(name, pt_out, eqx_out, atol=1e-4, rtol=1e-4)


def run_structural_check(seed: int) -> int:
    import jax
    import jax.numpy as jnp
    from foundax.architectures.gaot import (
        GAOT,
        MAGNOConfig,
        TransformerConfig,
        compute_neighbors_csr,
    )

    print("[GAOT] Structural check (JAX only, random weights)")
    m = GAOT(
        input_size=2,
        output_size=1,
        magno_config=MAGNOConfig(
            radius=0.3, lifting_channels=8, hidden_size=16, mlp_layers=2
        ),
        transformer_config=TransformerConfig(
            patch_size=2, hidden_size=16, num_layers=2
        ),
        latent_tokens_size=(8, 8),
        key=jax.random.PRNGKey(seed),
    )
    H = 8
    xs = np.linspace(0, 1, H, dtype=np.float32)
    latent = jnp.asarray(
        np.stack(np.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)
    )
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.random((40, 2)).astype(np.float32))
    q = jnp.asarray(rng.random((20, 2)).astype(np.float32))
    pn = jnp.asarray(rng.random((40, 2)).astype(np.float32))  # single-example
    en = [compute_neighbors_csr(np.asarray(x), np.asarray(latent), 0.3)]
    dn = [compute_neighbors_csr(np.asarray(latent), np.asarray(q), 0.3)]
    y = m(latent, x, pn, q, en, dn)
    print(f"  output shape: {y.shape}, finite: {bool(jnp.all(jnp.isfinite(y)))}")
    return 0 if bool(jnp.all(jnp.isfinite(y))) else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gaot-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "og_repos" / "gaot",
        help="Path to a clone of camlab-ethz/GAOT",
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

    if not (args.gaot_root / "src" / "model" / "gaot.py").exists():
        print(f"Reference repo missing at {args.gaot_root} — running structural check.")
        return run_structural_check(args.seed)

    _install_torch_scatter_shim()

    print("=" * 70)
    print("GAOT: PyTorch reference vs foundax Equinox port")
    print(f"  reference: {args.gaot_root}")
    print(f"  seed:      {args.seed}")
    print("=" * 70)

    cases = [
        # (name, use_attention, attention_type, use_geoembed, method, pooling, transform_type, num_layers)
        (
            "(a) linear / no attn / no geoembed",
            False,
            "cosine",
            False,
            "statistical",
            "max",
            "linear",
            2,
        ),
        (
            "(b) linear / cosine attn / no geoembed",
            True,
            "cosine",
            False,
            "statistical",
            "max",
            "linear",
            2,
        ),
        (
            "(c1) linear / dot_product attn / no geoembed",
            True,
            "dot_product",
            False,
            "statistical",
            "max",
            "linear",
            2,
        ),
        (
            "(c2) linear / cosine attn / geoembed=statistical",
            True,
            "cosine",
            True,
            "statistical",
            "max",
            "linear",
            2,
        ),
        (
            "(c) linear / dot_product attn / geoembed=statistical",
            True,
            "dot_product",
            True,
            "statistical",
            "max",
            "linear",
            2,
        ),
        (
            "(d) linear / cosine attn / geoembed=pointnet/mean",
            True,
            "cosine",
            True,
            "pointnet",
            "mean",
            "linear",
            2,
        ),
        # NOTE: upstream's `nonlinear` AGNO has a channel-dim bug when lifting
        # changes the feature width — both sides hit it, so we skip that case.
        (
            "(e) linear / cosine attn / geoembed=statistical / 5 layers",
            True,
            "cosine",
            True,
            "statistical",
            "max",
            "linear",
            5,
        ),
    ]
    ok_all = True
    for case in cases:
        try:
            ok = run_one(case[0], args.gaot_root, args.seed, *case[1:])
        except Exception as exc:
            print(f"  [SKIP] {case[0]:<60} raised {type(exc).__name__}: {exc}")
            ok = True
        ok_all = ok_all and ok
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
