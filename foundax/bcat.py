"""BCAT -- Block Causal Transformer for PDE Foundation Models.

**Paper:** "BCAT: A Block Causal Transformer for PDE Foundation Models
for Fluid Dynamics" (2025)
https://arxiv.org/abs/2501.18972

Architecture: Causal transformer with patched spatio-temporal input,
RMSNorm, SwiGLU activation, and learnable time embeddings.

Usage::

    model = foundax.bcat.base(x_num=64, max_output_dim=2)
    model = foundax.bcat(n_layer=12, dim_emb=1024)
"""

import importlib

from ._vendors import ensure_repo_on_path
from . import _callable_module


def _build(
    n_layer=12,
    dim_emb=1024,
    dim_ffn=2752,
    n_head=8,
    norm_first=True,
    norm_type="rms",
    activation="swiglu",
    qk_norm=True,
    x_num=128,
    max_output_dim=4,
    patch_num=16,
    patch_num_output=16,
    conv_dim=32,
    time_embed="learnable",
    max_time_len=20,
    max_data_len=20,
    deep=False,
    data_dim=1,
    *,
    key=None,
):
    import jax
    ensure_repo_on_path("jax_bcat")
    mod = importlib.import_module("jax_bcat.model_eqx")
    if key is None:
        key = jax.random.PRNGKey(0)
    kw = {k: v for k, v in locals().items() if k not in ("mod", "jax")}
    return mod.BCAT(**kw)


def base(
    n_layer=12,
    dim_emb=1024,
    dim_ffn=2752,
    n_head=8,
    norm_first=True,
    norm_type="rms",
    activation="swiglu",
    qk_norm=True,
    x_num=128,
    max_output_dim=4,
    patch_num=16,
    patch_num_output=16,
    conv_dim=32,
    time_embed="learnable",
    max_time_len=20,
    max_data_len=20,
    deep=False,
    data_dim=1,
    *,
    key=None,
):
    """Default BCAT model. 12 layers, dim_emb=1024, 8 heads, SwiGLU, RMSNorm."""
    return _build(**{k: v for k, v in locals().items()})


def v1(
    n_layer=12,
    dim_emb=1024,
    dim_ffn=2752,
    n_head=8,
    norm_first=True,
    norm_type="rms",
    activation="swiglu",
    qk_norm=True,
    x_num=128,
    max_output_dim=4,
    patch_num=16,
    patch_num_output=16,
    conv_dim=32,
    time_embed="learnable",
    max_time_len=20,
    max_data_len=20,
    deep=False,
    data_dim=1,
    *,
    key=None,
):
    """BCAT v1 -- alias of base."""
    return base(**{k: v for k, v in locals().items()})


default = base

__all__ = ["base", "v1", "default"]

_callable_module.install(__name__, _build)
