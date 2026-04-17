#!/usr/bin/env python3
"""
Foundation model PyTorch / Flax ↔ Equinox agreement heatmap.

For each model with a PyTorch original and a weight conversion pipeline:
  1. Create PyTorch model (random init) → state_dict
  2. Convert state_dict → Flax params (using per-repo convert_weights)
  3. Transfer Flax params → Equinox model (using per-model transfer functions)
  4. Compare forward pass + gradient: PyTorch vs Equinox

Models without a PyTorch original (DPOT, PDEformer-2) or a complete
PyTorch→Flax weight conversion (PROSE) use Flax ↔ Equinox comparison.

Usage
-----
  # Run all comparisons:
  python scripts/heatmap.py --run

  # Only selected models:
  python scripts/heatmap.py --run --models morph bcat poseidon

  # Load cached results:
  python scripts/heatmap.py --results results.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_paths(*paths: Path) -> None:
    """Add paths to sys.path if not already present."""
    for p in paths:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _load_module(name: str, path: Path):
    """Load a Python module from an arbitrary file path."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Per-model comparison runners
# ---------------------------------------------------------------------------
# Each function returns (forward_max_abs_diff, gradient_max_abs_diff, kind)
# where kind is "pt_eqx", "ms_eqx", or "flax_eqx".
# ---------------------------------------------------------------------------


def _compare_morph(
    projects_root: Path,
    seed: int = 42,
    spatial: int = 4,
) -> Tuple[float, float, str]:
    """MORPH ViT3DRegression: PyTorch ↔ Equinox."""
    import torch, jax, jax.numpy as jnp

    jax.config.update("jax_platform_name", "cpu")
    _ensure_paths(
        projects_root / "jax_morph",
        projects_root / "jax_morph" / "ogrepo" / "MORPH",
        projects_root.parent / "tests",
    )

    from jax_morph.model import ViT3DRegression as FlaxModel
    from jax_morph.model_eqx import ViT3DRegression as EqxModel
    from jax_morph import convert_pytorch_to_jax_params
    from src.utils.vit_conv_xatt_axialatt2 import ViT3DRegression as PTModel
    from test_morph_eqx import transfer_all_params as flax_to_eqx

    cfg = dict(
        patch_size=4, dim=32, depth=2, heads=2, heads_xa=2,
        mlp_dim=64, max_components=2, conv_filter=8, max_ar=1,
        max_patches=64, max_fields=2, model_size="Ti",
        dropout=0.0, emb_dropout=0.0,
    )
    S = spatial

    # PT model
    pt_model = PTModel(**cfg)
    pt_model.eval()
    pt_state = {k: v for k, v in pt_model.state_dict().items()}

    # Flax model (param template)
    flax_model = FlaxModel(**cfg)
    rng = jax.random.PRNGKey(0)
    dummy = jnp.zeros((1, 1, 1, 1, S, S, S))
    flax_vars = flax_model.init(rng, dummy, deterministic=True)

    # PT → Flax
    jax_params = convert_pytorch_to_jax_params(
        pt_state, flax_vars, heads_xa=cfg["heads_xa"]
    )

    # Flax → Equinox
    eqx_model = EqxModel(**cfg, key=jax.random.PRNGKey(1))
    fp = jax_params["params"] if "params" in jax_params else jax_params
    eqx_model = flax_to_eqx(fp, eqx_model)

    # Test input
    np.random.seed(seed)
    x_np = np.random.randn(1, 1, 1, 1, S, S, S).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_jax = jnp.array(x_np)

    # Forward
    with torch.no_grad():
        _, _, pred_pt = pt_model(x_pt)
    pred_pt_np = pred_pt.detach().cpu().numpy()
    _, _, pred_eqx = eqx_model(x_jax)
    fwd_diff = float(np.max(np.abs(pred_pt_np - np.array(pred_eqx))))

    # Gradient
    x_pt_g = torch.from_numpy(x_np).requires_grad_(True)
    _, _, p = pt_model(x_pt_g)
    p.sum().backward()
    grad_pt = x_pt_g.grad.detach().cpu().numpy()

    def _loss(x):
        _, _, o = eqx_model(x)
        return jnp.sum(o)

    grad_eqx = np.array(jax.grad(_loss)(x_jax))
    grad_diff = float(np.max(np.abs(grad_pt - grad_eqx)))
    return fwd_diff, grad_diff, "pt_eqx"


def _compare_mpp(
    projects_root: Path,
    seed: int = 42,
    resolution: int = 32,
) -> Tuple[float, float, str]:
    """MPP AViT: PyTorch ↔ Equinox."""
    import torch, jax, jax.numpy as jnp

    jax.config.update("jax_platform_name", "cpu")
    _ensure_paths(
        projects_root / "jax_mpp",
        projects_root / "jax_mpp" / "ogrepo" / "multiple_physics_pretraining",
        projects_root.parent / "tests",
    )

    from jax_mpp.avit import AViT as FlaxAViT
    from jax_mpp.avit_eqx import AViT as EqxAViT
    from jax_mpp import convert_pytorch_to_jax_params
    from models.avit import build_avit
    from test_mpp_eqx import transfer_all_params as flax_to_eqx

    cfg = dict(
        patch_size=(16, 16), embed_dim=32, processor_blocks=2,
        n_states=3, drop_path=0.0, bias_type="rel", num_heads=2,
    )
    T, B, C, H, W = 2, 1, cfg["n_states"], resolution, resolution

    # PT model
    class Params:
        pass
    params = Params()
    params.embed_dim = cfg["embed_dim"]
    params.processor_blocks = cfg["processor_blocks"]
    params.n_states = cfg["n_states"]
    params.num_heads = cfg["num_heads"]
    params.patch_size = cfg["patch_size"]
    params.bias_type = cfg["bias_type"]
    params.block_type = "axial"
    params.space_type = "axial_attention"
    params.time_type = "attention"
    params.gradient_checkpointing = False

    pt_model = build_avit(params)
    pt_model.eval()
    pt_state = {k: v for k, v in pt_model.state_dict().items()}

    # PT → Flax (MPP converter builds flax params from scratch)
    jax_params = convert_pytorch_to_jax_params(pt_state, verbose=False)

    # Flax → Equinox
    eqx_model = EqxAViT(**cfg, key=jax.random.PRNGKey(1))
    eqx_model = flax_to_eqx(jax_params, eqx_model, bias_type=cfg["bias_type"])

    # Test input
    np.random.seed(seed)
    x_np = np.random.randn(T, B, C, H, W).astype(np.float32)
    labels_pt = [list(range(C))]
    labels_jax = jnp.arange(C)
    bcs_np = np.zeros((B, 2), dtype=np.int64)

    x_pt = torch.from_numpy(x_np)
    bcs_pt = torch.from_numpy(bcs_np)
    x_jax = jnp.array(x_np)
    bcs_jax = jnp.array(bcs_np)

    # Forward
    with torch.no_grad():
        y_pt = pt_model(x_pt, labels_pt, bcs_pt).detach().cpu().numpy()

    y_eqx = np.array(eqx_model(x_jax, labels_jax, bcs_jax, deterministic=True))
    fwd_diff = float(np.max(np.abs(y_pt - y_eqx)))

    # Gradient
    x_pt_g = torch.from_numpy(x_np).requires_grad_(True)
    y_pt_g = pt_model(x_pt_g, labels_pt, bcs_pt)
    y_pt_g.sum().backward()
    grad_pt = x_pt_g.grad.detach().cpu().numpy()

    def _loss(x):
        return jnp.sum(eqx_model(x, labels_jax, bcs_jax, deterministic=True))

    grad_eqx = np.array(jax.grad(_loss)(x_jax))
    grad_diff = float(np.max(np.abs(grad_pt - grad_eqx)))
    return fwd_diff, grad_diff, "pt_eqx"


def _compare_poseidon(
    projects_root: Path,
    seed: int = 42,
) -> Tuple[float, float, str]:
    """Poseidon ScOT: PyTorch ↔ Equinox.

    Uses poseidonT-shaped config (depths=(4,4,4,4)) so the rule-based
    PT→Flax key converter matches all parameters.  Image size is shrunk
    to 32 to keep memory and runtime reasonable.
    """
    import torch, jax, jax.numpy as jnp

    jax.config.update("jax_platform_name", "cpu")
    _ensure_paths(
        projects_root / "jax_poseidon",
        projects_root / "jax_poseidon" / "ogrepo" / "poseidon",
        projects_root.parent / "tests",
    )

    from jax_poseidon.scot import ScOT as FlaxScOT, ScOTConfig
    from jax_poseidon.scot_eqx import ScOT as EqxScOT
    from scOT.model import ScOT as PTScOT
    from test_poseidon_eqx import transfer_all_params as flax_to_eqx
    from transformers import PretrainedConfig

    # Load the PT→Flax converter from the repo scripts
    convert_mod = _load_module(
        "poseidon_convert",
        projects_root / "jax_poseidon" / "scripts" / "convert.py",
    )
    convert_fn = convert_mod.convert_pytorch_to_jax

    # poseidonT-shaped config (4 stages with 4 layers each) — needed for
    # the rule-based key converter to match all parameters.
    image_size = 32
    config = ScOTConfig(
        name="poseidonT",
        image_size=image_size,
        patch_size=4,
        num_channels=2,
        num_out_channels=2,
        embed_dim=48,
        depths=(4, 4, 4, 4),
        num_heads=(3, 6, 12, 24),
        skip_connections=(1, 1, 1, 0),
        window_size=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.0,
        hidden_act="gelu",
        use_absolute_embeddings=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        p=1,
        channel_slice_list_normalized_loss=None,
        residual_model="convnext",
        use_conditioning=True,
        learn_residual=False,
        pretrained_window_sizes=(0, 0, 0, 0),
    )

    # PT model
    pt_config = PretrainedConfig()
    for k, v in config.__dict__.items():
        try:
            setattr(pt_config, k, v)
        except (AttributeError, TypeError):
            pass
    pt_config.model_type = "scot"
    pt_config.use_mask_token = False

    pt_model = PTScOT(pt_config, use_mask_token=False)
    pt_model.eval()
    pt_state = dict(pt_model.state_dict())

    # Flax model (param template)
    flax_model = FlaxScOT(config=config, use_conditioning=config.use_conditioning)
    rng = jax.random.PRNGKey(0)
    x_init = jnp.zeros((1, image_size, image_size, config.num_channels))
    t_init = jnp.zeros((1,))
    flax_vars = flax_model.init(
        rng, pixel_values=x_init, time=t_init, deterministic=True, return_dict=False
    )

    # PT → Flax
    jax_params = convert_fn(pt_state, flax_vars, verbose=False, large_model=False)

    # Flax → Equinox
    eqx_model = EqxScOT(
        config=config,
        use_conditioning=config.use_conditioning,
        key=jax.random.PRNGKey(1),
    )
    eqx_model = flax_to_eqx(jax_params, flax_model, eqx_model)

    # Test input (channel-last for JAX, channel-first for PT)
    np.random.seed(seed)
    x_np = np.random.randn(1, image_size, image_size, config.num_channels).astype(
        np.float32
    )
    time_np = np.array([0.5], dtype=np.float32)

    x_pt = torch.from_numpy(x_np).permute(0, 3, 1, 2)  # (B,C,H,W)
    time_pt = torch.from_numpy(time_np)
    x_jax = jnp.array(x_np)
    time_jax = jnp.array(time_np)

    # Forward
    with torch.no_grad():
        y_pt = pt_model(pixel_values=x_pt, time=time_pt).output
    y_pt_np = y_pt.cpu().numpy().transpose(0, 2, 3, 1)  # → (B,H,W,C)

    eqx_out = eqx_model(
        pixel_values=x_jax, time=time_jax, deterministic=True, return_dict=False
    )
    y_eqx_np = np.array(eqx_out[0])
    fwd_diff = float(np.max(np.abs(y_pt_np - y_eqx_np)))

    # Gradient
    x_pt_g = torch.from_numpy(x_np).permute(0, 3, 1, 2).requires_grad_(True)
    y_pt_g = pt_model(pixel_values=x_pt_g, time=time_pt).output
    y_pt_g.sum().backward()
    grad_pt = x_pt_g.grad.detach().cpu().numpy().transpose(0, 2, 3, 1)

    def _loss(x):
        out = eqx_model(
            pixel_values=x, time=time_jax, deterministic=True, return_dict=False
        )
        return jnp.sum(out[0])

    grad_eqx = np.array(jax.grad(_loss)(x_jax))
    grad_diff = float(np.max(np.abs(grad_pt - grad_eqx)))
    return fwd_diff, grad_diff, "pt_eqx"


def _compare_bcat(
    projects_root: Path,
    seed: int = 42,
) -> Tuple[float, float, str]:
    """BCAT: PyTorch ↔ Equinox."""
    import torch, jax, jax.numpy as jnp

    jax.config.update("jax_platform_name", "cpu")
    _ensure_paths(
        projects_root / "jax_bcat",
        projects_root / "jax_bcat" / "ogrepo" / "bcat" / "src",
        projects_root.parent / "tests",
    )

    from jax_bcat.model import BCAT as FlaxBCAT
    from jax_bcat.model_eqx import BCAT as EqxBCAT
    from jax_bcat import convert_pytorch_to_jax_params
    from omegaconf import OmegaConf
    from models.bcat import BCAT as PT_BCAT
    from test_bcat_eqx import transfer_weights as flax_to_eqx

    cfg = dict(
        n_layer=2, dim_emb=64, dim_ffn=128, n_head=4,
        norm_first=True, norm_type="rms", activation="swiglu",
        qk_norm=True, x_num=16, max_output_dim=2,
        patch_num=4, patch_num_output=4, conv_dim=8,
        time_embed="learnable", max_time_len=10, max_data_len=10, deep=False,
    )
    x_num, data_dim = cfg["x_num"], 2
    bs, t_total, input_len = 1, 5, 3

    # PT model — needs OmegaConf config with embedder sub-section
    pt_cfg = dict(
        n_layer=2, dim_emb=64, dim_ffn=128, n_head=4,
        norm_first=True, norm="rms", activation="swiglu",
        qk_norm=1, dropout=0, attn_dropout=0, rotary=0,
        flex_attn=0, kv_cache=0,
        patch_num=4, patch_num_output=4,
        embedder=dict(
            type="conv", dim=64, patch_num=4, patch_num_output=4,
            time_embed="learnable", max_time_len=10, conv_dim=8, early_conv=0,
        ),
    )
    model_config = OmegaConf.create(pt_cfg)
    pt_model = PT_BCAT(model_config, x_num, data_dim, max_data_len=t_total)
    pt_model.eval()
    pt_state = {k: v for k, v in pt_model.state_dict().items()}

    # Flax model (param template)
    flax_model = FlaxBCAT(**cfg)
    rng = jax.random.PRNGKey(0)
    data_dummy = jnp.ones((bs, t_total, x_num, x_num, data_dim))
    times_dummy = jnp.ones((bs, t_total, 1))
    flax_vars = flax_model.init(rng, data_dummy, times_dummy, input_len=input_len)

    # PT → Flax
    jax_params = convert_pytorch_to_jax_params(pt_state, flax_vars["params"])

    # Flax → Equinox
    eqx_model = EqxBCAT(**cfg, data_dim=data_dim, key=jax.random.PRNGKey(1))
    eqx_model = flax_to_eqx({"params": jax_params}, eqx_model)

    # Test input
    np.random.seed(seed)
    data_np = np.random.randn(bs, t_total, x_num, x_num, data_dim).astype(np.float32)
    times_np = np.arange(t_total, dtype=np.float32).reshape(1, t_total, 1)

    data_jax = jnp.array(data_np)
    times_jax = jnp.array(times_np)

    # PT forward (manual embedder->transformer->decoder, avoiding SDPA kernel)
    from models.bcat import block_lower_triangular_mask as pt_mask_fn

    d_in = torch.from_numpy(data_np[:, :-1])
    t_in = torch.from_numpy(times_np[:, :-1])
    with torch.no_grad():
        enc = pt_model.embedder.encode(d_in, t_in)
        data_len = enc.size(1)
        mask = pt_mask_fn(
            pt_model.seq_len_per_step, t_total, use_float=True,
        )[:data_len, :data_len]
        enc = pt_model.transformer(enc, mask)
        input_seq_len = (input_len - 1) * pt_model.seq_len_per_step
        y_pt = pt_model.embedder.decode(enc[:, input_seq_len:]).detach().cpu().numpy()

    y_eqx = np.array(eqx_model(data_jax, times_jax, input_len=input_len))
    fwd_diff = float(np.max(np.abs(y_pt - y_eqx)))

    # Gradient
    d_in_g = torch.from_numpy(data_np[:, :-1]).requires_grad_(True)
    enc_g = pt_model.embedder.encode(d_in_g, t_in)
    data_len_g = enc_g.size(1)
    mask_g = pt_mask_fn(
        pt_model.seq_len_per_step, t_total, use_float=True,
    )[:data_len_g, :data_len_g]
    enc_g = pt_model.transformer(enc_g, mask_g)
    y_pt_g = pt_model.embedder.decode(enc_g[:, input_seq_len:])
    y_pt_g.sum().backward()
    grad_pt = d_in_g.grad.detach().cpu().numpy()

    def _loss(data):
        return jnp.sum(eqx_model(data, times_jax, input_len=input_len))

    grad_eqx = np.array(jax.grad(_loss)(data_jax))
    grad_diff = float(np.max(np.abs(grad_pt - grad_eqx[:, :-1])))
    return fwd_diff, grad_diff, "pt_eqx"


def _compare_walrus(
    projects_root: Path,
    seed: int = 42,
) -> Tuple[float, float, str]:
    """WALRUS IsotropicModel: PyTorch → Equinox.

    The PT model's forward() requires ``metadata`` tied to the ``the_well``
    dataset, so we cannot call it directly.  Instead we:
      PT random init → state_dict → convert_pytorch_to_jax_params → Flax →
      transfer_weights → Equinox, then compare Flax vs Eqx forward.
    """
    import types
    import jax, jax.numpy as jnp
    import torch

    jax.config.update("jax_platform_name", "cpu")

    # -- Mock the_well (needed by PT ogrepo) --
    _well = types.ModuleType("the_well")
    _data = types.ModuleType("the_well.data")
    _ds = types.ModuleType("the_well.data.datasets")

    class _BCEnum:
        _m = {"PERIODIC": type("BC", (), {"value": 2})(),
               "OPEN": type("BC", (), {"value": 0})()}
        def __getitem__(self, k): return self._m[k]

    _ds.BoundaryCondition = _BCEnum()
    _well.data = _data
    _data.datasets = _ds
    for nm, mod in [("the_well", _well), ("the_well.data", _data),
                     ("the_well.data.datasets", _ds)]:
        sys.modules.setdefault(nm, mod)

    _ensure_paths(
        projects_root / "jax_walrus",
        projects_root / "jax_walrus" / "ogrepo" / "walrus",
    )

    from functools import partial
    import torch.nn as tnn
    from walrus.models.isotropic_model import IsotropicModel as PtModel
    from walrus.models.encoders.vstride_encoder import (
        SpaceBagAdaptiveDVstrideEncoder,
    )
    from walrus.models.decoders.vstride_decoder import AdaptiveDVstrideDecoder
    from walrus.models.spatiotemporal_blocks.space_time_split import (
        SpaceTimeSplitBlock,
    )
    from walrus.models.spatial_blocks.full_attention import FullAttention
    from walrus.models.temporal_blocks.axial_time_attention import (
        AxialTimeAttention,
    )
    from walrus.models.shared_utils.normalization import RMSGroupNorm

    from jax_walrus.convert_weights import (
        convert_pytorch_to_jax_params as pt_to_flax,
    )
    from jax_walrus.model import IsotropicModel as FlaxModel
    from jax_walrus.model_eqx import (
        IsotropicModel as EqxModel,
        transfer_weights as flax_to_eqx,
    )

    HIDDEN, INTER, N_STATES, PROC = 64, 32, 4, 2
    GROUPS, HEADS = 4, 4
    BKS = ((4, 2), (4, 2), (4, 2))
    INCLUDE_D = (2,)

    # -- PT model --
    pt_model = PtModel(
        encoder=partial(
            SpaceBagAdaptiveDVstrideEncoder,
            kernel_scales_seq=((4, 4),),
            base_kernel_size3d=BKS,
            variable_downsample=True,
            variable_deterministic_ds=True,
            learned_pad=True,
            norm_layer=RMSGroupNorm,
            activation=tnn.SiLU,
            extra_dims=3,
        ),
        decoder=partial(
            AdaptiveDVstrideDecoder,
            base_kernel_size3d=BKS,
            learned_pad=True,
            norm_layer=RMSGroupNorm,
            activation=tnn.SiLU,
        ),
        processor=partial(
            SpaceTimeSplitBlock,
            space_mixing=partial(FullAttention, num_heads=HEADS, mlp_dim=None),
            time_mixing=partial(
                AxialTimeAttention, num_heads=HEADS, bias_type="rel"
            ),
            channel_mixing=partial(tnn.Identity),
            norm_layer=RMSGroupNorm,
        ),
        projection_dim=HIDDEN,
        intermediate_dim=INTER,
        hidden_dim=HIDDEN,
        processor_blocks=PROC,
        n_states=N_STATES,
        drop_path=0.0,
        input_field_drop=0.0,
        groups=GROUPS,
        max_d=3,
        jitter_patches=False,
        causal_in_time=False,
        include_d=list(INCLUDE_D),
        gradient_checkpointing_freq=0,
    )
    pt_model.eval()
    pt_state = pt_model.state_dict()

    # -- Convert PT → Flax params --
    pt_flax = pt_to_flax(pt_state, processor_blocks=PROC, dim_keys=list(INCLUDE_D))

    # -- Flax model (used for forward reference) --
    eqx_cfg = dict(
        hidden_dim=HIDDEN, intermediate_dim=INTER, n_states=N_STATES,
        processor_blocks=PROC, groups=GROUPS, num_heads=HEADS, mlp_dim=0,
        max_d=3, causal_in_time=False, drop_path=0.0, bias_type="rel",
        base_kernel_size=BKS, use_spacebag=True, use_silu=True,
        include_d=INCLUDE_D, encoder_groups=GROUPS, learned_pad=True,
    )
    flax_model = FlaxModel(**eqx_cfg, jitter_patches=False)
    rng = jax.random.PRNGKey(42)
    B, T, H, W, C = 1, 2, 16, 16, N_STATES
    x_init = jnp.ones((B, T, H, W, C))
    state_labels = jnp.arange(C)
    bcs = [[0, 0], [0, 0]]
    flax_vars_pt = {"params": pt_flax["params"]}

    # -- Eqx model --
    eqx_model = EqxModel(**eqx_cfg, key=jax.random.PRNGKey(1))
    eqx_model = flax_to_eqx(flax_vars_pt, eqx_model)

    # -- Forward comparison --
    x = jax.random.normal(jax.random.PRNGKey(seed), (B, T, H, W, C))

    flax_out = np.array(
        flax_model.apply(flax_vars_pt, x, state_labels, bcs, deterministic=True)
    )
    eqx_out = np.array(eqx_model(x, state_labels, bcs, deterministic=True))
    fwd_diff = float(np.max(np.abs(flax_out - eqx_out)))

    def _flax_loss(inp):
        return jnp.sum(
            flax_model.apply(flax_vars_pt, inp, state_labels, bcs, deterministic=True)
        )

    def _eqx_loss(inp):
        return jnp.sum(eqx_model(inp, state_labels, bcs, deterministic=True))

    grad_flax = np.array(jax.grad(_flax_loss)(x))
    grad_eqx = np.array(jax.grad(_eqx_loss)(x))
    grad_diff = float(np.max(np.abs(grad_flax - grad_eqx)))
    return fwd_diff, grad_diff, "pt_eqx"


def _prose_pt_to_flax(pt_state, flax_params):
    """Apply PROSE PT→Flax weight mapping (from jax_prose/scripts/convert.py)."""
    import jax.numpy as jnp
    from flax.core import unfreeze

    params = unfreeze(flax_params)

    def _sp(d, path, val):
        for p in path[:-1]:
            d = d[p]
        d[path[-1]] = val

    for k, v in pt_state.items():
        a = v.detach().cpu().numpy()

        # ---- embedder ----
        if k == "embedder.patch_position_embeddings":
            _sp(params, ["embedder", "patch_position_embeddings"], jnp.asarray(a))
        elif k == "embedder.time_embed":
            _sp(params, ["embedder", "time_embed"], jnp.asarray(a))
        elif k == "embedder.conv_proj.0.weight":
            _sp(params, ["embedder", "conv_proj_0", "kernel"],
                jnp.asarray(a.transpose(2, 3, 1, 0)))
        elif k == "embedder.conv_proj.0.bias":
            _sp(params, ["embedder", "conv_proj_0", "bias"], jnp.asarray(a))
        elif k == "embedder.conv_proj.2.weight":
            _sp(params, ["embedder", "conv_proj_1", "kernel"],
                jnp.asarray(a.transpose(2, 3, 1, 0)))
        elif k == "embedder.conv_proj.2.bias":
            _sp(params, ["embedder", "conv_proj_1", "bias"], jnp.asarray(a))
        elif k == "embedder.post_proj.1.weight":
            kd = a.transpose(2, 3, 0, 1)[::-1, ::-1, :, :]
            _sp(params, ["embedder", "deconv", "kernel"], jnp.asarray(kd))
        elif k == "embedder.post_proj.1.bias":
            _sp(params, ["embedder", "deconv", "bias"], jnp.asarray(a))
        elif k == "embedder.post_proj.3.weight":
            _sp(params, ["embedder", "post_conv_0", "kernel"],
                jnp.asarray(a.transpose(2, 3, 1, 0)))
        elif k == "embedder.post_proj.3.bias":
            _sp(params, ["embedder", "post_conv_0", "bias"], jnp.asarray(a))
        elif k == "embedder.post_proj.5.weight":
            _sp(params, ["embedder", "post_conv_1", "kernel"],
                jnp.asarray(a.transpose(2, 3, 1, 0)))
        elif k == "embedder.post_proj.5.bias":
            _sp(params, ["embedder", "post_conv_1", "bias"], jnp.asarray(a))

        # ---- data_encoder ----
        elif k.startswith("data_encoder.transformer_encoder.layers."):
            parts = k.split(".")
            li = int(parts[3])
            rest = ".".join(parts[4:])
            base = ["data_encoder", "layers_%d" % li]
            if rest.startswith("self_attn."):
                r = rest.split(".")
                _sp(params, base + ["self_attn", r[1],
                    "kernel" if r[2] == "weight" else "bias"],
                    jnp.asarray(a.T if r[2] == "weight" else a))
            elif rest.startswith("linear1."):
                _sp(params, base + ["linear1",
                    "kernel" if rest.endswith("weight") else "bias"],
                    jnp.asarray(a.T if rest.endswith("weight") else a))
            elif rest.startswith("linear2."):
                _sp(params, base + ["linear2",
                    "kernel" if rest.endswith("weight") else "bias"],
                    jnp.asarray(a.T if rest.endswith("weight") else a))
            elif rest == "norm1.weight":
                _sp(params, base + ["norm1", "scale"], jnp.asarray(a))
            elif rest == "norm2.weight":
                _sp(params, base + ["norm2", "scale"], jnp.asarray(a))
        elif k == "data_encoder.transformer_encoder.norm.weight":
            _sp(params, ["data_encoder", "norm", "scale"], jnp.asarray(a))

        # ---- symbol_encoder ----
        elif k.startswith("symbol_encoder.transformer_encoder.layers."):
            parts = k.split(".")
            li = int(parts[3])
            rest = ".".join(parts[4:])
            base = ["symbol_encoder", "transformer_encoder", "layers_%d" % li]
            if rest.startswith("self_attn."):
                r = rest.split(".")
                _sp(params, base + ["self_attn", r[1],
                    "kernel" if r[2] == "weight" else "bias"],
                    jnp.asarray(a.T if r[2] == "weight" else a))
            elif rest.startswith("linear1."):
                _sp(params, base + ["linear1",
                    "kernel" if rest.endswith("weight") else "bias"],
                    jnp.asarray(a.T if rest.endswith("weight") else a))
            elif rest.startswith("linear2."):
                _sp(params, base + ["linear2",
                    "kernel" if rest.endswith("weight") else "bias"],
                    jnp.asarray(a.T if rest.endswith("weight") else a))
            elif rest == "norm1.weight":
                _sp(params, base + ["norm1", "scale"], jnp.asarray(a))
            elif rest == "norm2.weight":
                _sp(params, base + ["norm2", "scale"], jnp.asarray(a))
        elif k == "symbol_encoder.transformer_encoder.norm.weight":
            _sp(params, ["symbol_encoder", "transformer_encoder", "norm", "scale"],
                jnp.asarray(a))
        elif k == "symbol_encoder.positional_embedding.pe":
            _sp(params, ["symbol_encoder", "pe"], jnp.asarray(a))
        elif k == "symbol_encoder.word_embeddings.weight":
            _sp(params, ["symbol_encoder", "word_embeddings", "embedding"],
                jnp.asarray(a))

        # ---- fusion ----
        elif k == "fusion.type_embeddings.weight":
            _sp(params, ["fusion", "type_embeddings", "embedding"], jnp.asarray(a))
        elif k.startswith("fusion.transformer_encoder.layers."):
            parts = k.split(".")
            li = int(parts[3])
            rest = ".".join(parts[4:])
            base = ["fusion", "transformer_encoder", "layers_%d" % li]
            if rest.startswith("self_attn."):
                r = rest.split(".")
                _sp(params, base + ["self_attn", r[1],
                    "kernel" if r[2] == "weight" else "bias"],
                    jnp.asarray(a.T if r[2] == "weight" else a))
            elif rest.startswith("linear1."):
                _sp(params, base + ["linear1",
                    "kernel" if rest.endswith("weight") else "bias"],
                    jnp.asarray(a.T if rest.endswith("weight") else a))
            elif rest.startswith("linear2."):
                _sp(params, base + ["linear2",
                    "kernel" if rest.endswith("weight") else "bias"],
                    jnp.asarray(a.T if rest.endswith("weight") else a))
            elif rest == "norm1.weight":
                _sp(params, base + ["norm1", "scale"], jnp.asarray(a))
            elif rest == "norm2.weight":
                _sp(params, base + ["norm2", "scale"], jnp.asarray(a))
        elif k == "fusion.transformer_encoder.norm.weight":
            _sp(params, ["fusion", "transformer_encoder", "norm", "scale"],
                jnp.asarray(a))

        # ---- data_decoder ----
        elif k == "data_decoder.time_embed":
            _sp(params, ["data_decoder", "time_embed"], jnp.asarray(a))
        elif k == "data_decoder.patch_position_embeddings":
            _sp(params, ["data_decoder", "patch_position_embeddings"], jnp.asarray(a))
        elif k.startswith("data_decoder.transformer_decoder.layers."):
            parts = k.split(".")
            li = int(parts[3])
            rest = ".".join(parts[4:])
            base = ["data_decoder", "layers_%d" % li]
            if rest.startswith("multihead_attn."):
                r = rest.split(".")
                _sp(params, base + ["multihead_attn", r[1],
                    "kernel" if r[2] == "weight" else "bias"],
                    jnp.asarray(a.T if r[2] == "weight" else a))
            elif rest.startswith("linear1."):
                _sp(params, base + ["linear1",
                    "kernel" if rest.endswith("weight") else "bias"],
                    jnp.asarray(a.T if rest.endswith("weight") else a))
            elif rest.startswith("linear2."):
                _sp(params, base + ["linear2",
                    "kernel" if rest.endswith("weight") else "bias"],
                    jnp.asarray(a.T if rest.endswith("weight") else a))
            elif rest == "norm1.weight":
                _sp(params, base + ["norm1", "scale"], jnp.asarray(a))
            elif rest == "norm2.weight":
                _sp(params, base + ["norm2", "scale"], jnp.asarray(a))
        elif k == "data_decoder.transformer_decoder.norm.weight":
            _sp(params, ["data_decoder", "norm", "scale"], jnp.asarray(a))

    return params


def _compare_prose(
    projects_root: Path,
    seed: int = 42,
) -> Tuple[float, float, str]:
    """PROSE 2to1: PyTorch ↔ Equinox."""
    import types
    import jax, jax.numpy as jnp
    import torch

    jax.config.update("jax_platform_name", "cpu")
    _ensure_paths(
        projects_root / "jax_prose",
        projects_root / "jax_prose" / "ogrepo" / "prose" / "prose_fd",
        projects_root.parent / "tests",
    )

    from omegaconf import OmegaConf
    from models.transformer_wrappers import PROSE_2to1 as PtPROSE
    from jax_prose.prose_fd_2to1 import PROSE2to1 as FlaxPROSE, Prose2to1Config
    from jax_prose.model_eqx import PROSE2to1 as EqxPROSE
    from test_prose_eqx import transfer_weights_2to1 as flax_to_eqx
    from flax.core import freeze

    DIM, FFN, NHEAD = 64, 128, 4
    PATCH, PATCH_OUT = 4, 4
    N_WORDS, X_NUM, DATA_DIM = 32, 16, 2
    INPUT_LEN, OUTPUT_LEN, SYM_LEN = 2, 2, 8

    # -- PT model --
    pt_cfg = OmegaConf.create({
        "name": "prose_2to1",
        "dim_emb": DIM, "dim_ffn": FFN, "n_head": NHEAD,
        "dropout": 0, "norm_first": True,
        "patch_num": PATCH, "patch_num_output": PATCH_OUT,
        "carry_last_frame": 0, "time_embed": "learnable",
        "custom_encoder": 1, "custom_attn": 1, "rotary": 0, "norm": "rms",
        "embedder": {
            "type": "conv", "dim": DIM, "patch_num": PATCH,
            "patch_num_output": PATCH_OUT, "time_embed": "learnable",
            "initialize_small_output": 0, "conv_dim": 32,
            "early_conv": 0, "deep": 0,
        },
        "data_encoder": {
            "n_layer": 1, "positional_embedding": None,
            "dim_emb": DIM, "dim_ffn": FFN, "n_head": NHEAD,
            "dropout": 0, "norm_first": True,
            "custom_encoder": 1, "custom_attn": 1, "rotary": 0, "norm": "rms",
        },
        "symbol_encoder": {
            "n_layer": 1, "positional_embedding": "sinusoidal",
            "dim_emb": DIM, "dim_ffn": FFN, "n_head": NHEAD,
            "dropout": 0, "norm_first": True,
            "custom_encoder": 1, "custom_attn": 1, "rotary": 0, "norm": "rms",
        },
        "fusion": {
            "n_layer": 1, "type_embeddings": True,
            "dim_emb": DIM, "dim_ffn": FFN, "n_head": NHEAD,
            "dropout": 0, "norm_first": True,
            "custom_encoder": 1, "custom_attn": 1, "rotary": 0, "norm": "rms",
        },
        "data_decoder": {
            "n_layer": 1, "query_dim": 1, "self_attn": 0,
            "dim_emb": DIM, "dim_ffn": FFN, "n_head": NHEAD,
            "dropout": 0, "norm_first": True,
            "patch_num_output": PATCH_OUT, "time_embed": "learnable",
            "final_ln": 1, "custom_attn": 1, "rotary": 0, "norm": "rms",
        },
    })
    id2word = {0: "<BOS>", 1: "<EOS>", 2: "<PAD>", 3: "<PLACEHOLDER>"}
    for i in range(4, N_WORDS):
        id2word[i] = str(i)
    sym_env = types.SimpleNamespace(equation_id2word=id2word)

    pt_model = PtPROSE(pt_cfg, sym_env, X_NUM, DATA_DIM, output_len=OUTPUT_LEN)
    pt_model.eval()
    pt_state = pt_model.state_dict()

    # -- Flax model (init → overwrite with PT weights) --
    flax_cfg = Prose2to1Config(
        dim_emb=DIM, dim_ffn=FFN, n_head=NHEAD,
        patch_num=PATCH, patch_num_output=PATCH_OUT,
        data_encoder_layers=1, symbol_encoder_layers=1,
        fusion_layers=1, data_decoder_layers=1,
    )
    flax_model = FlaxPROSE(
        n_words=N_WORDS, x_num=X_NUM, max_output_dim=DATA_DIM, cfg=flax_cfg
    )
    rng = jax.random.PRNGKey(0)
    x_init = jnp.ones((1, INPUT_LEN, X_NUM, X_NUM, DATA_DIM))
    tin_init = jnp.zeros((1, INPUT_LEN, 1))
    tout_init = jnp.zeros((1, OUTPUT_LEN, 1))
    sym_init = jnp.zeros((1, SYM_LEN), dtype=jnp.int32)
    mask_init = jnp.zeros((1, SYM_LEN), dtype=bool)
    flax_vars = flax_model.init(
        {"params": rng}, x_init, tin_init, tout_init, sym_init, mask_init
    )
    converted_params = _prose_pt_to_flax(pt_state, flax_vars["params"])
    flax_vars = {"params": freeze(converted_params)}

    # -- Equinox model --
    eqx_model = EqxPROSE(
        n_words=N_WORDS, x_num=X_NUM, max_output_dim=DATA_DIM,
        dim_emb=DIM, dim_ffn=FFN, n_head=NHEAD,
        patch_num=PATCH, patch_num_output=PATCH_OUT,
        data_encoder_layers=1, symbol_encoder_layers=1,
        fusion_layers=1, data_decoder_layers=1,
        key=jax.random.PRNGKey(1),
    )
    eqx_model = flax_to_eqx(flax_vars, eqx_model)

    # -- Test input --
    np.random.seed(seed)
    data_np = np.random.randn(1, INPUT_LEN, X_NUM, X_NUM, DATA_DIM).astype(np.float32)
    tin_np = np.random.rand(1, INPUT_LEN, 1).astype(np.float32)
    tout_np = np.random.rand(1, OUTPUT_LEN, 1).astype(np.float32)
    sym_np = np.random.randint(0, N_WORDS, (1, SYM_LEN)).astype(np.int64)
    mask_np = np.zeros((1, SYM_LEN), dtype=bool)

    pt_data = torch.from_numpy(data_np).requires_grad_(True)
    pt_tin = torch.from_numpy(tin_np)
    pt_tout = torch.from_numpy(tout_np)
    pt_sym = torch.from_numpy(sym_np)
    pt_mask = torch.from_numpy(mask_np)

    pt_out_t = pt_model.fwd(
        data_input=pt_data,
        input_times=pt_tin,
        output_times=pt_tout,
        symbol_input=pt_sym,
        symbol_padding_mask=pt_mask,
    )
    pt_out = pt_out_t.detach().cpu().numpy()

    data = jnp.asarray(data_np)
    tin = jnp.asarray(tin_np)
    tout = jnp.asarray(tout_np)
    sym = jnp.asarray(sym_np.astype(np.int32))
    sym_mask = jnp.asarray(mask_np)

    eqx_out = np.array(eqx_model(data, tin, tout, sym, sym_mask))
    fwd_diff = float(np.max(np.abs(pt_out - eqx_out)))

    pt_out_t.sum().backward()
    grad_pt = pt_data.grad.cpu().numpy()

    def _eqx_loss(d):
        return jnp.sum(eqx_model(d, tin, tout, sym, sym_mask))

    grad_eqx = np.array(jax.grad(_eqx_loss)(data))
    grad_diff = float(np.max(np.abs(grad_pt - grad_eqx)))
    return fwd_diff, grad_diff, "pt_eqx"


def _compare_dpot(
    projects_root: Path,
    seed: int = 42,
) -> Tuple[float, float, str]:
    """DPOT: PyTorch ↔ Equinox."""
    import jax, jax.numpy as jnp
    import torch

    jax.config.update("jax_platform_name", "cpu")
    _ensure_paths(
        projects_root / "jax_dpot",
        projects_root / "jax_dpot" / "ogrepo" / "DPOT",
        projects_root.parent / "tests",
    )

    from models.dpot import DPOTNet as PtDPOTNet
    from jax_dpot.model import DPOTNet as FlaxDPOTNet
    from jax_dpot.model_eqx import DPOTNet as EqxDPOTNet
    from jax_dpot.convert_weights import convert_pytorch_to_jax_params as pt_to_flax
    from test_dpot_eqx import transfer_all_params as flax_to_eqx

    kwargs = dict(
        img_size=32, patch_size=8, mixing_type="afno",
        in_channels=2, out_channels=2, in_timesteps=3, out_timesteps=1,
        n_blocks=4, embed_dim=32, out_layer_dim=16, depth=2,
        modes=8, mlp_ratio=1.0, n_cls=4,
        normalize=False, act="gelu", time_agg="exp_mlp",
    )

    # -- PT model --
    pt_model = PtDPOTNet(**kwargs)
    pt_model.eval()
    pt_state = pt_model.state_dict()

    # -- Flax model (init → overwrite with PT weights) --
    flax_model = FlaxDPOTNet(**kwargs)
    rng = jax.random.PRNGKey(42)
    x_init = jnp.zeros((1, 32, 32, 3, 2))
    flax_vars = flax_model.init(rng, x_init)
    flax_vars = {"params": pt_to_flax(pt_state, flax_vars["params"])}

    # -- Eqx model --
    eqx_model = EqxDPOTNet(**kwargs, key=jax.random.PRNGKey(0))
    eqx_model = flax_to_eqx(flax_vars, eqx_model)

    # -- Test input --
    x_np = np.array(jax.random.normal(jax.random.PRNGKey(seed), (1, 32, 32, 3, 2)))

    with torch.no_grad():
        pt_pred, pt_cls = pt_model(torch.from_numpy(x_np))
    pt_pred = pt_pred.numpy()
    pt_cls = pt_cls.numpy()

    data = jnp.asarray(x_np)
    e_pred, e_cls = eqx_model(data)

    fwd_diff = max(
        float(np.max(np.abs(pt_pred - np.array(e_pred)))),
        float(np.max(np.abs(pt_cls - np.array(e_cls)))),
    )

    # -- Gradient (torch.autograd vs jax.grad) --
    pt_x = torch.from_numpy(x_np).requires_grad_(True)
    pt_p, pt_c = pt_model(pt_x)
    (pt_p.sum() + pt_c.sum()).backward()
    grad_pt = pt_x.grad.cpu().numpy()

    def _eqx_loss(x):
        p, c = eqx_model(x)
        return jnp.sum(p) + jnp.sum(c)

    grad_eqx = np.array(jax.grad(_eqx_loss)(data))
    grad_diff = float(np.max(np.abs(grad_pt - grad_eqx)))
    return fwd_diff, grad_diff, "pt_eqx"


def _compare_pdeformer2(
    projects_root: Path,
    seed: int = 42,
) -> Tuple[float, float, str]:
    """PDEformer-2: MindSpore ↔ Equinox.

    MindSpore requires Python ≤3.12, so we run the MS model in a separate
    Python 3.12 venv via subprocess.  The helper script
    ``ms_pdeformer_helper.py`` creates the MS model with random weights,
    runs forward + gradient, and saves everything to numpy files.  Here
    we convert those weights to JAX, load into the Eqx model, and compare.
    """
    import subprocess
    import tempfile

    import jax, jax.numpy as jnp
    from flax.core import freeze

    jax.config.update("jax_platform_name", "cpu")
    _ensure_paths(
        projects_root / "jax_pdeformer2",
        projects_root.parent / "tests",
    )

    from jax_pdeformer2.pdeformer import create_pdeformer_from_config
    from jax_pdeformer2.utils import convert_mindspore_to_jax
    from jax_pdeformer2.model_eqx import PDEformer as EqxPDEformer
    from test_pdeformer2_eqx import transfer_weights as flax_to_eqx

    # -- Run MindSpore helper in separate Python 3.12 env --
    ms_python = str(projects_root.parent / ".ms_env" / "bin" / "python")
    helper = str(projects_root.parent / "scripts" / "ms_pdeformer_helper.py")
    ogrepo = str(projects_root / "jax_pdeformer2" / "ogrepo" / "pdeformer-2")

    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [ms_python, helper, tmpdir, ogrepo, str(seed)],
            check=True,
            capture_output=True,
            text=True,
        )

        ms_params_raw = dict(np.load(os.path.join(tmpdir, "ms_params.npz")))
        io = dict(np.load(os.path.join(tmpdir, "ms_io.npz")))

    ms_output = io["output"]
    ms_grad = io["grad"]

    # -- Convert MS weights → Flax params --
    jax_params_np = convert_mindspore_to_jax(ms_params_raw)

    def _to_jax(d):
        if isinstance(d, dict):
            return {k: _to_jax(v) for k, v in d.items()}
        if isinstance(d, np.ndarray):
            return jnp.array(d)
        return d

    jax_params = freeze(_to_jax(jax_params_np))

    # -- Config matching the MS helper --
    test_cfg = {
        "graphormer": {
            "num_node_type": 8, "num_in_degree": 4, "num_out_degree": 4,
            "num_spatial": 4, "num_encoder_layers": 2, "embed_dim": 32,
            "ffn_embed_dim": 64, "num_heads": 4, "pre_layernorm": True,
        },
        "scalar_encoder": {"dim_hidden": 16, "num_layers": 2},
        "function_encoder": {
            "type": "cnn2dv3", "num_branches": 4, "resolution": 128,
            "conv2d_input_txyz": False, "cnn_keep_nchw": True,
        },
        "inr": {
            "type": "poly_inr", "num_layers": 3, "dim_hidden": 16,
            "poly_inr": {
                "enable_affine": False, "enable_shift": True,
                "enable_scale": True, "activation_fn": "sin",
                "affine_act_fn": "identity",
            },
        },
        "hypernet": {"dim_hidden": 16, "num_layers": 2, "shared": False},
        "multi_inr": {"enable": False},
    }
    gc = test_cfg["graphormer"]
    sc = test_cfg["scalar_encoder"]
    fc = test_cfg["function_encoder"]
    ic = test_cfg["inr"]
    pc = ic["poly_inr"]
    hc = test_cfg["hypernet"]
    mc = test_cfg["multi_inr"]

    # -- Reconstruct inputs from saved numpy arrays --
    input_keys = [
        "node_type", "node_scalar", "node_function",
        "in_degree", "out_degree", "attn_bias", "spatial_pos", "coordinate",
    ]
    inputs = {k: jnp.array(io[k]) for k in input_keys}

    # -- Eqx model --
    eqx_model = EqxPDEformer(
        num_node_type=gc["num_node_type"],
        num_in_degree=gc["num_in_degree"],
        num_out_degree=gc["num_out_degree"],
        num_spatial=gc["num_spatial"],
        num_encoder_layers=gc["num_encoder_layers"],
        embed_dim=gc["embed_dim"],
        ffn_embed_dim=gc["ffn_embed_dim"],
        num_heads=gc["num_heads"],
        pre_layernorm=gc.get("pre_layernorm", True),
        scalar_dim_hidden=sc["dim_hidden"],
        scalar_num_layers=sc["num_layers"],
        func_enc_resolution=fc["resolution"],
        func_enc_input_txyz=fc.get("conv2d_input_txyz", False),
        func_enc_keep_nchw=fc.get("cnn_keep_nchw", True),
        inr_dim_hidden=ic["dim_hidden"],
        inr_num_layers=ic["num_layers"],
        enable_affine=pc.get("enable_affine", False),
        enable_shift=pc.get("enable_shift", True),
        enable_scale=pc.get("enable_scale", True),
        activation_fn=pc.get("activation_fn", "sin"),
        affine_act_fn=pc.get("affine_act_fn", "identity"),
        hyper_dim_hidden=hc["dim_hidden"],
        hyper_num_layers=hc["num_layers"],
        share_hypernet=hc.get("shared", False),
        multi_inr=mc.get("enable", False),
        separate_latent=mc.get("separate_latent", False),
        key=jax.random.PRNGKey(0),
    )
    eqx_model = flax_to_eqx(jax_params, eqx_model)

    # -- Forward --
    eqx_out = np.array(eqx_model(**inputs))
    fwd_diff = float(np.max(np.abs(ms_output - eqx_out)))

    # -- Gradient (w.r.t. coordinate) --
    def _eqx_loss(coord):
        return jnp.sum(eqx_model(**{**inputs, "coordinate": coord}))

    eqx_grad = np.array(jax.grad(_eqx_loss)(inputs["coordinate"]))
    grad_diff = float(np.max(np.abs(ms_grad - eqx_grad)))
    return fwd_diff, grad_diff, "ms_eqx"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

RUNNERS = {
    "morph": _compare_morph,
    "mpp": _compare_mpp,
    "poseidon": _compare_poseidon,
    "bcat": _compare_bcat,
    "walrus": _compare_walrus,
    "prose": _compare_prose,
    "dpot": _compare_dpot,
    "pdeformer2": _compare_pdeformer2,
}

ALL_MODELS = list(RUNNERS.keys())


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _cleanup_ogrepo_modules() -> None:
    """Remove ogrepo-injected modules and sys.path entries to prevent cross-contamination.

    Multiple ogrepos have conflicting top-level packages (e.g. ``models``,
    ``src``, ``datasets``) that would clash if left in ``sys.modules``/``sys.path``.
    """
    stale = [
        k for k in sys.modules
        if k == "models" or k.startswith("models.")
        or k == "src" or k.startswith("src.")
        or k == "datasets" or k.startswith("datasets.")
    ]
    for k in stale:
        del sys.modules[k]
    # Remove ogrepo paths from sys.path
    sys.path[:] = [p for p in sys.path if "/ogrepo/" not in p]


def run_comparisons(
    models: List[str],
    projects_root: Path,
) -> Dict[str, Dict[str, Any]]:
    """Return {model: {"forward": float|None, "gradient": float|None, "type": str}}."""
    results: Dict[str, Dict[str, Any]] = {}
    for name in models:
        print(f"\n{'='*60}")
        print(f"  Running comparison: {name}")
        print(f"{'='*60}")
        _cleanup_ogrepo_modules()
        try:
            fwd, grad, kind = RUNNERS[name](projects_root)
            results[name] = {"forward": fwd, "gradient": grad, "type": kind}
            label = {"pt_eqx": "PT↔Eqx", "ms_eqx": "MS↔Eqx"}.get(kind, "Flax↔Eqx")
            print(f"  ✓ {name} ({label}): fwd={fwd:.3e}  grad={grad:.3e}")
        except Exception:
            traceback.print_exc()
            results[name] = {"forward": None, "gradient": None, "type": "error"}
            print(f"  ✗ {name}: SKIPPED (dependency or config issue)")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_heatmap(
    results: Dict[str, Dict[str, Any]],
    output: str = "heatmap.png",
) -> None:
    """Log-log scatter: Forward error (x) vs Gradient error (y) per model.

    Each model is a labelled point.  A y=x diagonal shows where forward
    and gradient accuracy are equal, and dashed crosshairs at float32
    machine-epsilon give an at-a-glance reference.
    """
    from matplotlib.offsetbox import AnchoredText
    import matplotlib.patheffects as pe

    # -- Collect data -------------------------------------------------------
    rows: List[Dict[str, Any]] = []
    for name, r in results.items():
        fwd = r.get("forward")
        grad = r.get("gradient")
        kind = r.get("type", "error")
        if fwd is None or grad is None:
            continue
        rows.append(dict(name=name, fwd=fwd, grad=grad, kind=kind))

    if not rows:
        print("No valid results to plot.")
        return

    names = [r["name"] for r in rows]
    fwd_vals = np.array([r["fwd"] for r in rows])
    grad_vals = np.array([r["grad"] for r in rows])
    kinds = [r["kind"] for r in rows]

    suffix_map = {"pt_eqx": "PT", "ms_eqx": "MS", "flax_eqx": "Flax"}

    # -- Figure -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))

    # y = x diagonal
    all_vals = np.concatenate([fwd_vals, grad_vals])
    lo, hi = all_vals.min() * 0.15, all_vals.max() * 8
    diag = np.array([lo, hi])
    ax.plot(diag, diag, color="#cccccc", linewidth=0.9, zorder=1)

    # f32 ε crosshairs
    eps32 = float(np.finfo(np.float32).eps)  # ≈1.19e-7
    ax.axvline(eps32, color="#bbbbbb", linewidth=0.8, linestyle="--", zorder=1)
    ax.axhline(eps32, color="#bbbbbb", linewidth=0.8, linestyle="--", zorder=1)
    ax.text(eps32 * 1.35, lo * 2.5, "f32 ε", fontsize=7.5, color="#888888",
            ha="left", va="bottom", rotation=0)
    ax.text(lo * 2.5, eps32 * 1.35, "f32 ε", fontsize=7.5, color="#888888",
            ha="left", va="bottom", rotation=0)

    # Scatter points
    colors = {
        "pt_eqx": "#2166ac",
        "ms_eqx": "#b2182b",
        "flax_eqx": "#4dac26",
    }
    for kind_key in set(kinds):
        mask = [k == kind_key for k in kinds]
        fv = fwd_vals[mask]
        gv = grad_vals[mask]
        ax.scatter(fv, gv, s=60, color=colors.get(kind_key, "#333333"),
                   edgecolors="white", linewidths=0.6, zorder=4,
                   label=f"vs {suffix_map.get(kind_key, kind_key)}")

    # Labels with white outline for readability
    outline = [pe.withStroke(linewidth=2.5, foreground="white")]
    # Use adjustText if available, otherwise fall back to fixed offsets
    try:
        from adjustText import adjust_text
        texts = []
        for i, nm in enumerate(names):
            texts.append(ax.text(
                fwd_vals[i], grad_vals[i], nm,
                fontsize=8.5, color="#333333",
                path_effects=outline, zorder=5,
            ))
        adjust_text(texts, x=fwd_vals, y=grad_vals, ax=ax,
                    force_text=(0.4, 0.4), force_points=(0.3, 0.3),
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa",
                                    linewidth=0.5))
    except ImportError:
        for i, nm in enumerate(names):
            ax.annotate(
                nm,
                (fwd_vals[i], grad_vals[i]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8.5,
                color="#333333",
                path_effects=outline,
                zorder=5,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    ax.set_xlabel("Forward max |diff|", fontsize=10)
    ax.set_ylabel("Gradient max |diff|", fontsize=10)
    ax.set_title(
        "Equinox Translation Accuracy — Original ↔ Equinox",
        fontsize=12, fontweight="bold", pad=10,
    )

    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(which="major", linewidth=0.3, alpha=0.5)
    ax.grid(which="minor", linewidth=0.15, alpha=0.25)

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    print(f"\nPlot saved to {output}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Foundation model agreement heatmap (PyTorch / Flax \u2194 Equinox)"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--run", action="store_true",
        help="Run comparisons live (uses random init)",
    )
    mode.add_argument(
        "--results", type=Path,
        help="Load pre-computed results from JSON",
    )
    parser.add_argument(
        "--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS,
        help="Models to include (default: all)",
    )
    parser.add_argument(
        "--projects-root", type=Path,
        default=Path(__file__).resolve().parents[1] / "repos",
        help="Path to vendored jax_* repos (default: foundax/repos)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="heatmap.png",
        help="Output image path (default: heatmap.png)",
    )
    parser.add_argument(
        "--save-results", type=Path, default=None,
        help="Save results to JSON for later reuse",
    )
    args = parser.parse_args()

    if args.results:
        with open(args.results) as f:
            results = json.load(f)
        results = {k: v for k, v in results.items() if k in args.models}
    else:
        results = run_comparisons(args.models, args.projects_root)

    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.save_results}")

    plot_heatmap(results, output=args.output)


if __name__ == "__main__":
    main()
