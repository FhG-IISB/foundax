"""BCAT — Block Causal Transformer for PDE Foundation Models.

Lazy-loading factory for the vendored ``jax_bcat`` package.

**Paper:** *"BCAT: A Block Causal Transformer for PDE Foundation Models
for Fluid Dynamics"*
https://arxiv.org/abs/2501.18972

Architecture: Causal transformer with patched spatio-temporal input,
RMSNorm, SwiGLU activation, and learnable time embeddings.

Default configuration (``bcat_default``)::

    n_layer          = 12
    dim_emb          = 1024
    dim_ffn          = 2752
    n_head           = 8
    norm_first       = True
    norm_type        = "rms"
    activation       = "swiglu"
    qk_norm          = True
    x_num            = 128
    max_output_dim   = 4
    patch_num        = 16
    patch_num_output = 16
    conv_dim         = 32
    time_embed       = "learnable"
    max_time_len     = 20
    max_data_len     = 20
    deep             = False
"""

import importlib
from typing import Any

from ._vendors import ensure_repo_on_path


def base() -> Any:
    """Create the default BCAT model.

    Delegates to ``jax_bcat.bcat_default``.

    Takes no arguments — the default ``BCATConfig`` is applied
    internally.

    Returns
    -------
    BCAT
        Uninitialised Flax ``nn.Module``.

        Forward call signature::

            model.__call__(
                data,          # (bs, input_len + output_len, x_num, x_num, data_dim)
                times,         # (bs, input_len + output_len, 1)
                input_len=10,  # number of input time-steps
            ) -> jnp.ndarray  # (bs, output_len, x_num, x_num, data_dim)

    References
    ----------
    "BCAT: A Block Causal Transformer for PDE Foundation Models for
    Fluid Dynamics", 2025.  https://arxiv.org/abs/2501.18972
    """
    ensure_repo_on_path("jax_bcat")
    return importlib.import_module("jax_bcat").bcat_default()


def v1() -> Any:
    """Create BCAT v1 — alias of :func:`base`."""
    return base()


default = base


__all__ = ["base", "v1", "default"]
