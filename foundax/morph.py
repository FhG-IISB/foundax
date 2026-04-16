"""MORPH — PDE Foundation Models with Arbitrary Data Modality.

Lazy-loading factories for the vendored ``jax_morph`` package.

**Paper:** Rautela et al., *"MORPH: PDE Foundation Models with Arbitrary
Data Modality"* (2025)
https://arxiv.org/abs/2509.21670

**Weights:** https://huggingface.co/mahindrautela/MORPH
**Code:**    https://github.com/lanl/MORPH

Architecture: ``ViT3DRegression`` — a 3-D Vision Transformer for
regression over PDE data.

Variants:

=========  ======  =====  =====  ======  =========
Variant    embed   depth  heads  mlp     Params
=========  ======  =====  =====  ======  =========
Ti         256     4      4      1024    ~9.9 M
S          512     4      8      2048    ~32.8 M
M          768     8      12     3072    ~125.6 M
L          1024    16     16     4096    ~483.3 M
=========  ======  =====  =====  ======  =========

All variants share: ``patch_size=8``, ``conv_filter=8``,
``heads_xa=32``, ``max_patches=4096``, ``max_fields=3``.
L additionally uses ``max_ar=16`` (others ``max_ar=1``).

Lowercase names (``ti``, ``s``, ``m``, ``l``) are aliases.
"""

import importlib
from typing import Any

from ._vendors import ensure_repo_on_path


def Ti(**overrides: Any) -> Any:
    """Create MORPH-Ti (Tiny) — ~9.9 M params.

    Delegates to ``jax_morph.morph_Ti``.

    Config: embed_dim=256, depth=4, heads=4, mlp_dim=1024, max_ar=1.

    Parameters
    ----------
    **overrides
        Override any ``ViT3DRegression`` attribute, e.g.
        ``dropout=0.0``, ``max_patches=512``.

    Returns
    -------
    ViT3DRegression
        Uninitialised Flax ``nn.Module``.

    References
    ----------
    Rautela et al., "MORPH", 2025.  https://arxiv.org/abs/2509.21670
    """
    ensure_repo_on_path("jax_morph")
    return importlib.import_module("jax_morph").morph_Ti(**overrides)


def S(**overrides: Any) -> Any:
    """Create MORPH-S (Small) — ~32.8 M params.

    Delegates to ``jax_morph.morph_S``.

    Config: embed_dim=512, depth=4, heads=8, mlp_dim=2048, max_ar=1.

    Parameters
    ----------
    **overrides
        Override any ``ViT3DRegression`` attribute.

    Returns
    -------
    ViT3DRegression
        Uninitialised Flax ``nn.Module``.

    References
    ----------
    Rautela et al., "MORPH", 2025.  https://arxiv.org/abs/2509.21670
    """
    ensure_repo_on_path("jax_morph")
    return importlib.import_module("jax_morph").morph_S(**overrides)


def M(**overrides: Any) -> Any:
    """Create MORPH-M (Medium) — ~125.6 M params.

    Delegates to ``jax_morph.morph_M``.

    Config: embed_dim=768, depth=8, heads=12, mlp_dim=3072, max_ar=1.

    Parameters
    ----------
    **overrides
        Override any ``ViT3DRegression`` attribute.

    Returns
    -------
    ViT3DRegression
        Uninitialised Flax ``nn.Module``.

    References
    ----------
    Rautela et al., "MORPH", 2025.  https://arxiv.org/abs/2509.21670
    """
    ensure_repo_on_path("jax_morph")
    return importlib.import_module("jax_morph").morph_M(**overrides)


def L(**overrides: Any) -> Any:
    """Create MORPH-L (Large) — ~483.3 M params.

    Delegates to ``jax_morph.morph_L``.

    Config: embed_dim=1024, depth=16, heads=16, mlp_dim=4096, max_ar=16.

    Parameters
    ----------
    **overrides
        Override any ``ViT3DRegression`` attribute.

    Returns
    -------
    ViT3DRegression
        Uninitialised Flax ``nn.Module``.

    References
    ----------
    Rautela et al., "MORPH", 2025.  https://arxiv.org/abs/2509.21670
    """
    ensure_repo_on_path("jax_morph")
    return importlib.import_module("jax_morph").morph_L(**overrides)


ti = Ti
s = S
m = M
l = L


__all__ = ["Ti", "S", "M", "L", "ti", "s", "m", "l"]
