"""PDEformer-2 -- Foundation Model for Two-Dimensional PDEs.

**Paper:** Shi et al., *"PDEformer-2"* (2025)
https://arxiv.org/abs/2502.14844

Architecture: Graphormer encoder + Implicit Neural Representation (INR)
decoder with a hyper-network bridge.

Usage::

    model = foundax.pdeformer2.small(inr_dim_hidden=256)
    model = foundax.pdeformer2(num_encoder_layers=12, embed_dim=768)
"""

import importlib

from ._vendors import ensure_repo_on_path
from . import _callable_module

try:
    import jax.numpy as jnp
    _f32 = jnp.float32
except Exception:
    _f32 = None


def _build(
    num_node_type=128,
    num_in_degree=32,
    num_out_degree=32,
    num_spatial=16,
    num_encoder_layers=12,
    embed_dim=768,
    ffn_embed_dim=1536,
    num_heads=32,
    pre_layernorm=True,
    scalar_dim_hidden=256,
    scalar_num_layers=3,
    func_enc_type="cnn2dv3",
    func_enc_num_branches=4,
    func_enc_resolution=128,
    func_enc_input_txyz=False,
    func_enc_keep_nchw=True,
    inr_dim_hidden=768,
    inr_num_layers=12,
    enable_affine=False,
    enable_shift=True,
    enable_scale=True,
    activation_fn="sin",
    affine_act_fn="identity",
    hyper_dim_hidden=512,
    hyper_num_layers=2,
    share_hypernet=False,
    multi_inr=False,
    separate_latent=False,
    dtype=None,
):
    ensure_repo_on_path("jax_pdeformer2")
    mod = importlib.import_module("jax_pdeformer2")
    kw = {k: v for k, v in locals().items() if k != "mod"}
    if kw["dtype"] is None:
        import jax.numpy as _jnp
        kw["dtype"] = _jnp.float32
    return mod.PDEformer(**kw)


def small(
    num_node_type=128,
    num_in_degree=32,
    num_out_degree=32,
    num_spatial=16,
    num_encoder_layers=9,
    embed_dim=512,
    ffn_embed_dim=1024,
    num_heads=32,
    pre_layernorm=True,
    scalar_dim_hidden=256,
    scalar_num_layers=3,
    func_enc_type="cnn2dv3",
    func_enc_num_branches=4,
    func_enc_resolution=128,
    func_enc_input_txyz=False,
    func_enc_keep_nchw=True,
    inr_dim_hidden=128,
    inr_num_layers=12,
    enable_affine=False,
    enable_shift=True,
    enable_scale=True,
    activation_fn="sin",
    affine_act_fn="identity",
    hyper_dim_hidden=512,
    hyper_num_layers=2,
    share_hypernet=False,
    multi_inr=False,
    separate_latent=False,
    dtype=None,
):
    """PDEformer-2 Small ~27.7 M params. Graphormer(9, 512) + INR(128)."""
    return _build(**{k: v for k, v in locals().items()})


def base(
    num_node_type=128,
    num_in_degree=32,
    num_out_degree=32,
    num_spatial=16,
    num_encoder_layers=12,
    embed_dim=768,
    ffn_embed_dim=1536,
    num_heads=32,
    pre_layernorm=True,
    scalar_dim_hidden=256,
    scalar_num_layers=3,
    func_enc_type="cnn2dv3",
    func_enc_num_branches=4,
    func_enc_resolution=128,
    func_enc_input_txyz=False,
    func_enc_keep_nchw=True,
    inr_dim_hidden=768,
    inr_num_layers=12,
    enable_affine=False,
    enable_shift=True,
    enable_scale=True,
    activation_fn="sin",
    affine_act_fn="identity",
    hyper_dim_hidden=512,
    hyper_num_layers=2,
    share_hypernet=False,
    multi_inr=False,
    separate_latent=False,
    dtype=None,
):
    """PDEformer-2 Base. Graphormer(12, 768) + INR(768)."""
    return _build(**{k: v for k, v in locals().items()})


def fast(
    num_node_type=128,
    num_in_degree=32,
    num_out_degree=32,
    num_spatial=16,
    num_encoder_layers=12,
    embed_dim=768,
    ffn_embed_dim=1536,
    num_heads=32,
    pre_layernorm=True,
    scalar_dim_hidden=256,
    scalar_num_layers=3,
    func_enc_type="cnn2dv3",
    func_enc_num_branches=4,
    func_enc_resolution=128,
    func_enc_input_txyz=False,
    func_enc_keep_nchw=True,
    inr_dim_hidden=256,
    inr_num_layers=12,
    enable_affine=False,
    enable_shift=True,
    enable_scale=True,
    activation_fn="sin",
    affine_act_fn="identity",
    hyper_dim_hidden=512,
    hyper_num_layers=2,
    share_hypernet=False,
    multi_inr=False,
    separate_latent=False,
    dtype=None,
):
    """PDEformer-2 Fast. Graphormer(12, 768) + INR(256) -- smaller INR than Base."""
    return _build(**{k: v for k, v in locals().items()})


__all__ = ["small", "base", "fast"]

_callable_module.install(__name__, _build)
