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
from typing import Callable

import jax

from ._vendors import ensure_repo_on_path
from . import _callable_module


def _build(
    n_layer=12,
    dim_emb=1024,
    dim_ffn=2752,
    n_head=8,
    norm_first=True,
    norm_type="rms",
    activation: Callable = jax.nn.silu,
    gated: bool = True,
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
    activation: Callable = jax.nn.silu,
    gated: bool = True,
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
    """Default BCAT model.

    Block Causal Transformer with patched spatio-temporal input, RMSNorm,
    SwiGLU activation, and learnable time embeddings for PDE fluid dynamics.

    Reference:
        *BCAT: A Block Causal Transformer for PDE Foundation Models
        for Fluid Dynamics* (2025).
        https://arxiv.org/abs/2501.18972

    Example::

        model = foundax.bcat.base(x_num=64, max_output_dim=2)

    Shape:
        - Input: ``(bs, seq_len, x_num, x_num, data_dim)`` + times
        - Output: ``(bs, output_len, x_num, x_num, data_dim)``
        - ``x_num`` must be divisible by ``patch_num``.

    See Also:
        :func:`v1`

    Args:
        n_layer: Number of transformer layers.
        dim_emb: Token embedding dimension.
        dim_ffn: Feedforward network hidden dimension.
        n_head: Number of attention heads.
        norm_first: Apply normalisation before attention/MLP (pre-norm).
        norm_type: Normalisation type (``"rms"`` or ``"layer"``).
        activation: Activation callable (e.g. ``jax.nn.silu``,
            ``jax.nn.gelu``).  Combined with ``gated=True`` for
            SwiGLU / GeGLU style gating.
        gated: Use gated linear unit (GLU) in the feedforward block.
            When ``True``, the FFN multiplies ``activation(h)`` by a
            learned gate projection (e.g. SwiGLU when
            ``activation=jax.nn.silu``).
        qk_norm: Apply LayerNorm to query and key before attention.
        x_num: Spatial grid resolution (number of grid points).
        max_output_dim: Maximum number of output physical channels.
        patch_num: Number of patches per spatial dimension (input).
        patch_num_output: Number of patches per spatial dimension (output).
        conv_dim: Convolutional filter width in the patch embedding.
        time_embed: Time embedding type (``"learnable"``,
            ``"continuous"``).
        max_time_len: Maximum time-sequence length.
        max_data_len: Maximum data-sequence length.
        deep: Use a deeper / multi-scale architecture variant.
        data_dim: Data / channel dimension of the raw input.
        key: JAX PRNG key (``None`` → ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (BCAT).
    """
    return _build(**{k: v for k, v in locals().items()})


def v1(
    n_layer=12,
    dim_emb=1024,
    dim_ffn=2752,
    n_head=8,
    norm_first=True,
    norm_type="rms",
    activation: Callable = jax.nn.silu,
    gated: bool = True,
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
    """BCAT v1 -- alias of ``base``.

    See :func:`base` for full parameter documentation.
    """
    return base(**{k: v for k, v in locals().items()})


default = base

__all__ = ["base", "v1", "default"]

_callable_module.install(__name__, _build)
