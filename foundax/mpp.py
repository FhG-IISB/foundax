"""MPP — Multiple Physics Pretraining for Physical Surrogate Models.

Lazy-loading factories for the vendored ``jax_mpp`` package.

**Paper:** McCabe et al., *"Multiple Physics Pretraining for Physical
Surrogate Models"* (NeurIPS 2024)
https://openreview.net/forum?id=DKSI3bULiZ

**Weights:** https://drive.google.com/drive/folders/1Qaqa-RnzUDOO8-Gi4zlf4BE53SfWqDwx

Architecture: ``AViT`` (Adaptive Vision Transformer) with variable-
resolution patching for multi-physics operator learning.

Variants:

=========  ======  =====  ======  ==========  =========
Variant    embed   heads  blocks  n_states    Params
=========  ======  =====  ======  ==========  =========
Ti         192     3      12      12          ~5.5 M
S          384     6      12      12          ~21 M
B          768     12     12      12          ~83 M
L          1024    16     24      12          ~300 M
=========  ======  =====  ======  ==========  =========

Lowercase names (``ti``, ``s``, ``b``, ``l``) are aliases.
"""

import importlib
from typing import Any

from ._vendors import ensure_repo_on_path


def Ti(**overrides: Any) -> Any:
    """Create AViT-Tiny — ~5.5 M params.

    Delegates to ``jax_mpp.avit_Ti``.

    Config: embed_dim=192, num_heads=3, processor_blocks=12, n_states=12.

    Parameters
    ----------
    **overrides
        Override any ``AViT`` configuration parameter.

    Returns
    -------
    AViT
        Uninitialised Flax ``nn.Module``.

    References
    ----------
    McCabe et al., NeurIPS 2024.
    https://openreview.net/forum?id=DKSI3bULiZ
    """
    ensure_repo_on_path("jax_mpp")
    return importlib.import_module("jax_mpp").avit_Ti(**overrides)


def S(**overrides: Any) -> Any:
    """Create AViT-Small — ~21 M params.

    Delegates to ``jax_mpp.avit_S``.

    Config: embed_dim=384, num_heads=6, processor_blocks=12, n_states=12.

    Parameters
    ----------
    **overrides
        Override any ``AViT`` configuration parameter.

    Returns
    -------
    AViT
        Uninitialised Flax ``nn.Module``.

    References
    ----------
    McCabe et al., NeurIPS 2024.
    https://openreview.net/forum?id=DKSI3bULiZ
    """
    ensure_repo_on_path("jax_mpp")
    return importlib.import_module("jax_mpp").avit_S(**overrides)


def B(**overrides: Any) -> Any:
    """Create AViT-Base — ~83 M params.

    Delegates to ``jax_mpp.avit_B``.

    Config: embed_dim=768, num_heads=12, processor_blocks=12, n_states=12.

    Parameters
    ----------
    **overrides
        Override any ``AViT`` configuration parameter.

    Returns
    -------
    AViT
        Uninitialised Flax ``nn.Module``.

    References
    ----------
    McCabe et al., NeurIPS 2024.
    https://openreview.net/forum?id=DKSI3bULiZ
    """
    ensure_repo_on_path("jax_mpp")
    return importlib.import_module("jax_mpp").avit_B(**overrides)


def L(**overrides: Any) -> Any:
    """Create AViT-Large — ~300 M params.

    Delegates to ``jax_mpp.avit_L``.

    Config: embed_dim=1024, num_heads=16, processor_blocks=24, n_states=12.

    Parameters
    ----------
    **overrides
        Override any ``AViT`` configuration parameter.

    Returns
    -------
    AViT
        Uninitialised Flax ``nn.Module``.

    References
    ----------
    McCabe et al., NeurIPS 2024.
    https://openreview.net/forum?id=DKSI3bULiZ
    """
    ensure_repo_on_path("jax_mpp")
    return importlib.import_module("jax_mpp").avit_L(**overrides)


ti = Ti
s = S
b = B
l = L


__all__ = ["Ti", "S", "B", "L", "ti", "s", "b", "l"]
