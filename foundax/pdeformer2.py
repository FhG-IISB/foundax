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
from typing import Callable

from ._vendors import ensure_repo_on_path
from . import _callable_module

try:
    import jax.numpy as jnp

    _f32 = jnp.float32
except Exception:
    _f32 = None


def _identity(x):
    """Identity activation (no-op)."""
    return x


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
    func_enc_resolution=128,
    func_enc_input_txyz=False,
    func_enc_keep_nchw=True,
    inr_dim_hidden=768,
    inr_num_layers=12,
    enable_affine=False,
    enable_shift=True,
    enable_scale=True,
    activation_fn: Callable = jnp.sin,
    affine_act_fn: Callable = _identity,
    hyper_dim_hidden=512,
    hyper_num_layers=2,
    share_hypernet=False,
    multi_inr=False,
    separate_latent=False,
    *,
    key=None,
):
    import jax

    ensure_repo_on_path("jax_pdeformer2")
    mod = importlib.import_module("jax_pdeformer2.model_eqx")
    if key is None:
        key = jax.random.PRNGKey(0)
    kw = {k: v for k, v in locals().items() if k not in ("mod", "jax")}
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
    func_enc_resolution=128,
    func_enc_input_txyz=False,
    func_enc_keep_nchw=True,
    inr_dim_hidden=128,
    inr_num_layers=12,
    enable_affine=False,
    enable_shift=True,
    enable_scale=True,
    activation_fn: Callable = jnp.sin,
    affine_act_fn: Callable = _identity,
    hyper_dim_hidden=512,
    hyper_num_layers=2,
    share_hypernet=False,
    multi_inr=False,
    separate_latent=False,
):
    """PDEformer-2 Small ~27.7 M params.

    Graphormer encoder + Implicit Neural Representation (INR) decoder
    with a hyper-network bridge for two-dimensional PDE solving.

    Reference:
        Shi et al., *PDEformer-2: A Foundation Model for Two-
        Dimensional PDEs* (2025). https://arxiv.org/abs/2502.14844

    Example::

        model = foundax.pdeformer2.small(inr_dim_hidden=256)

    Shape:
        - Input: graph-structured (``node_type``, ``node_scalar``,
          ``node_function``, ``degrees``, ``attn_bias``,
          ``spatial_pos``, ``coordinate``).
        - Output: ``(n_graph, num_points, 1)``

    See Also:
        :func:`base`, :func:`fast`

    Args:
        num_node_type: Number of node-type categories in the PDE graph.
        num_in_degree: Number of in-degree bins for degree encoding.
        num_out_degree: Number of out-degree bins for degree encoding.
        num_spatial: Spatial embedding dimension.
        num_encoder_layers: Number of Graphormer encoder layers.
        embed_dim: Graphormer embedding dimension.
        ffn_embed_dim: Feedforward hidden dimension inside Graphormer.
        num_heads: Number of attention heads.
        pre_layernorm: Apply LayerNorm before attention (pre-norm).
        scalar_dim_hidden: Hidden dimension of the scalar encoder MLP.
        scalar_num_layers: Number of scalar encoder MLP layers.
        func_enc_resolution: Input resolution for the function encoder.
        func_enc_input_txyz: Include time in function-encoder input
            channels.
        func_enc_keep_nchw: Keep NCHW layout after function encoding.
        inr_dim_hidden: INR MLP hidden dimension.
        inr_num_layers: INR MLP depth.
        enable_affine: Enable full affine transformation in INR.
        enable_shift: Enable additive shift modulation in INR.
        enable_scale: Enable multiplicative scale modulation in INR.
        activation_fn: INR activation callable (e.g. ``jnp.sin``,
            ``jax.nn.relu``).
        affine_act_fn: Hypernet output activation callable
            (e.g. ``_identity``).
        hyper_dim_hidden: Hyper-network hidden dimension.
        hyper_num_layers: Hyper-network depth.
        share_hypernet: Share a single hyper-network across all INR
            layers.
        multi_inr: Use a second INR head for multi-output prediction.
        separate_latent: Use a separate latent vector for the second
            INR head.

    Returns:
        An ``equinox.Module`` (PDEformer).
    """
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
    func_enc_resolution=128,
    func_enc_input_txyz=False,
    func_enc_keep_nchw=True,
    inr_dim_hidden=768,
    inr_num_layers=12,
    enable_affine=False,
    enable_shift=True,
    enable_scale=True,
    activation_fn: Callable = jnp.sin,
    affine_act_fn: Callable = _identity,
    hyper_dim_hidden=512,
    hyper_num_layers=2,
    share_hypernet=False,
    multi_inr=False,
    separate_latent=False,
):
    """PDEformer-2 Base.

    Graphormer encoder + Implicit Neural Representation (INR) decoder
    with a hyper-network bridge for two-dimensional PDE solving.

    Reference:
        Shi et al., *PDEformer-2: A Foundation Model for Two-
        Dimensional PDEs* (2025). https://arxiv.org/abs/2502.14844

    Example::

        model = foundax.pdeformer2.base()

    Shape:
        - Input: graph-structured (``node_type``, ``node_scalar``,
          ``node_function``, ``degrees``, ``attn_bias``,
          ``spatial_pos``, ``coordinate``).
        - Output: ``(n_graph, num_points, 1)``

    See Also:
        :func:`small`, :func:`fast`

    Args:
        num_node_type: Number of node-type categories in the PDE graph.
        num_in_degree: Number of in-degree bins for degree encoding.
        num_out_degree: Number of out-degree bins for degree encoding.
        num_spatial: Spatial embedding dimension.
        num_encoder_layers: Number of Graphormer encoder layers.
        embed_dim: Graphormer embedding dimension.
        ffn_embed_dim: Feedforward hidden dimension inside Graphormer.
        num_heads: Number of attention heads.
        pre_layernorm: Apply LayerNorm before attention (pre-norm).
        scalar_dim_hidden: Hidden dimension of the scalar encoder MLP.
        scalar_num_layers: Number of scalar encoder MLP layers.
        func_enc_resolution: Input resolution for the function encoder.
        func_enc_input_txyz: Include time in function-encoder input
            channels.
        func_enc_keep_nchw: Keep NCHW layout after function encoding.
        inr_dim_hidden: INR MLP hidden dimension.
        inr_num_layers: INR MLP depth.
        enable_affine: Enable full affine transformation in INR.
        enable_shift: Enable additive shift modulation in INR.
        enable_scale: Enable multiplicative scale modulation in INR.
        activation_fn: INR activation callable (e.g. ``jnp.sin``,
            ``jax.nn.relu``).
        affine_act_fn: Hypernet output activation callable
            (e.g. ``_identity``).
        hyper_dim_hidden: Hyper-network hidden dimension.
        hyper_num_layers: Hyper-network depth.
        share_hypernet: Share a single hyper-network across all INR
            layers.
        multi_inr: Use a second INR head for multi-output prediction.
        separate_latent: Use a separate latent vector for the second
            INR head.

    Returns:
        An ``equinox.Module`` (PDEformer).
    """
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
    func_enc_resolution=128,
    func_enc_input_txyz=False,
    func_enc_keep_nchw=True,
    inr_dim_hidden=256,
    inr_num_layers=12,
    enable_affine=False,
    enable_shift=True,
    enable_scale=True,
    activation_fn: Callable = jnp.sin,
    affine_act_fn: Callable = _identity,
    hyper_dim_hidden=512,
    hyper_num_layers=2,
    share_hypernet=False,
    multi_inr=False,
    separate_latent=False,
):
    """PDEformer-2 Fast -- smaller INR than Base.

    Graphormer encoder + Implicit Neural Representation (INR) decoder
    with a hyper-network bridge for two-dimensional PDE solving.
    Uses a narrower INR (256 vs 768) for faster inference.

    Reference:
        Shi et al., *PDEformer-2: A Foundation Model for Two-
        Dimensional PDEs* (2025). https://arxiv.org/abs/2502.14844

    Example::

        model = foundax.pdeformer2.fast()

    Shape:
        - Input: graph-structured (``node_type``, ``node_scalar``,
          ``node_function``, ``degrees``, ``attn_bias``,
          ``spatial_pos``, ``coordinate``).
        - Output: ``(n_graph, num_points, 1)``

    See Also:
        :func:`small`, :func:`base`

    Args:
        num_node_type: Number of node-type categories in the PDE graph.
        num_in_degree: Number of in-degree bins for degree encoding.
        num_out_degree: Number of out-degree bins for degree encoding.
        num_spatial: Spatial embedding dimension.
        num_encoder_layers: Number of Graphormer encoder layers.
        embed_dim: Graphormer embedding dimension.
        ffn_embed_dim: Feedforward hidden dimension inside Graphormer.
        num_heads: Number of attention heads.
        pre_layernorm: Apply LayerNorm before attention (pre-norm).
        scalar_dim_hidden: Hidden dimension of the scalar encoder MLP.
        scalar_num_layers: Number of scalar encoder MLP layers.
        func_enc_resolution: Input resolution for the function encoder.
        func_enc_input_txyz: Include time in function-encoder input
            channels.
        func_enc_keep_nchw: Keep NCHW layout after function encoding.
        inr_dim_hidden: INR MLP hidden dimension.
        inr_num_layers: INR MLP depth.
        enable_affine: Enable full affine transformation in INR.
        enable_shift: Enable additive shift modulation in INR.
        enable_scale: Enable multiplicative scale modulation in INR.
        activation_fn: INR activation callable (e.g. ``jnp.sin``,
            ``jax.nn.relu``).
        affine_act_fn: Hypernet output activation callable
            (e.g. ``_identity``).
        hyper_dim_hidden: Hyper-network hidden dimension.
        hyper_num_layers: Hyper-network depth.
        share_hypernet: Share a single hyper-network across all INR
            layers.
        multi_inr: Use a second INR head for multi-output prediction.
        separate_latent: Use a separate latent vector for the second
            INR head.

    Returns:
        An ``equinox.Module`` (PDEformer).
    """
    return _build(**{k: v for k, v in locals().items()})


__all__ = ["small", "base", "fast"]

_callable_module.install(__name__, _build)
