"""DPOT — DPOTNet (2-D) model wrappers.

Lazy-loading factories for the vendored ``jax_dpot`` package.

Architecture: ``DPOTNet`` — a patched Fourier / AFNO-based neural
operator with temporal aggregation for 2-D PDE surrogate modelling.

Variants:

=========  ======  =====  ========  ==========  ===========
Variant    embed   depth  n_blocks  img_size    patch_size
=========  ======  =====  ========  ==========  ===========
Ti         512     4      4         128         8
S          1024    6      8         128         8
M          1024    12     8         128         8
L          1536    24     16        128         8
H          2048    27     8         128         8
=========  ======  =====  ========  ==========  ===========

All share: ``in_channels=4``, ``out_channels=4``, ``in_timesteps=10``,
``out_timesteps=1``, ``modes=32``, ``n_cls=12``, ``normalize=False``,
``act="gelu"``, ``time_agg="exp_mlp"``, ``mixing_type="afno"``.

Lowercase names (``ti``, ``s``, ``m``, ``l``, ``h``) are aliases.
"""

import importlib
from typing import Any

from ._vendors import ensure_repo_on_path


def Ti() -> Any:
    """Create DPOTNet-Ti (Tiny).

    Delegates to ``jax_dpot.dpot_ti``.

    Config: embed_dim=512, depth=4, n_blocks=4, img_size=128, patch_size=8.

    Returns
    -------
    DPOTNet
        Uninitialised Flax ``nn.Module``.
    """
    ensure_repo_on_path("jax_dpot")
    return importlib.import_module("jax_dpot").dpot_ti()


def S() -> Any:
    """Create DPOTNet-S (Small).

    Delegates to ``jax_dpot.dpot_s``.

    Config: embed_dim=1024, depth=6, n_blocks=8, img_size=128, patch_size=8.

    Returns
    -------
    DPOTNet
        Uninitialised Flax ``nn.Module``.
    """
    ensure_repo_on_path("jax_dpot")
    return importlib.import_module("jax_dpot").dpot_s()


def M() -> Any:
    """Create DPOTNet-M (Medium).

    Delegates to ``jax_dpot.dpot_m``.

    Config: embed_dim=1024, depth=12, n_blocks=8, img_size=128, patch_size=8.

    Returns
    -------
    DPOTNet
        Uninitialised Flax ``nn.Module``.
    """
    ensure_repo_on_path("jax_dpot")
    return importlib.import_module("jax_dpot").dpot_m()


def L() -> Any:
    """Create DPOTNet-L (Large).

    Delegates to ``jax_dpot.dpot_l``.

    Config: embed_dim=1536, depth=24, n_blocks=16, img_size=128, patch_size=8.

    Returns
    -------
    DPOTNet
        Uninitialised Flax ``nn.Module``.
    """
    ensure_repo_on_path("jax_dpot")
    return importlib.import_module("jax_dpot").dpot_l()


def H() -> Any:
    """Create DPOTNet-H (Huge).

    Delegates to ``jax_dpot.dpot_h``.

    Config: embed_dim=2048, depth=27, n_blocks=8, img_size=128, patch_size=8.

    Returns
    -------
    DPOTNet
        Uninitialised Flax ``nn.Module``.
    """
    ensure_repo_on_path("jax_dpot")
    return importlib.import_module("jax_dpot").dpot_h()


ti = Ti
s = S
m = M
l = L
h = H


__all__ = ["Ti", "S", "M", "L", "H", "ti", "s", "m", "l", "h"]
