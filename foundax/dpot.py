"""DPOT -- DPOTNet (2-D) model wrappers.

Architecture: ``DPOTNet`` -- a patched Fourier / AFNO-based neural
operator with temporal aggregation for 2-D PDE surrogate modelling.

Usage::

    model = foundax.dpot.Ti(in_channels=1, out_channels=1)
    model = foundax.dpot(embed_dim=512, depth=4, in_channels=1, out_channels=1)
"""

import importlib

from ._vendors import ensure_repo_on_path
from . import _callable_module


def _build(
    img_size=224,
    patch_size=16,
    mixing_type="afno",
    in_channels=1,
    out_channels=4,
    in_timesteps=1,
    out_timesteps=1,
    n_blocks=4,
    embed_dim=768,
    out_layer_dim=32,
    depth=12,
    modes=32,
    mlp_ratio=1.0,
    n_cls=12,
    normalize=False,
    act="gelu",
    time_agg="exp_mlp",
    *,
    key=None,
):
    import jax
    ensure_repo_on_path("jax_dpot")
    mod = importlib.import_module("jax_dpot.model_eqx")
    if key is None:
        key = jax.random.PRNGKey(0)
    kw = {k: v for k, v in locals().items() if k not in ("mod", "jax")}
    return mod.DPOTNet(**kw)


def Ti(
    img_size=128,
    patch_size=8,
    mixing_type="afno",
    in_channels=4,
    out_channels=4,
    in_timesteps=10,
    out_timesteps=1,
    n_blocks=4,
    embed_dim=512,
    out_layer_dim=32,
    depth=4,
    modes=32,
    mlp_ratio=1.0,
    n_cls=12,
    normalize=False,
    act="gelu",
    time_agg="exp_mlp",
):
    """DPOTNet-Ti (Tiny). embed=512, depth=4, n_blocks=4."""
    return _build(**{k: v for k, v in locals().items()})


def S(
    img_size=128,
    patch_size=8,
    mixing_type="afno",
    in_channels=4,
    out_channels=4,
    in_timesteps=10,
    out_timesteps=1,
    n_blocks=8,
    embed_dim=1024,
    out_layer_dim=32,
    depth=6,
    modes=32,
    mlp_ratio=1.0,
    n_cls=12,
    normalize=False,
    act="gelu",
    time_agg="exp_mlp",
):
    """DPOTNet-S (Small). embed=1024, depth=6, n_blocks=8."""
    return _build(**{k: v for k, v in locals().items()})


def M(
    img_size=128,
    patch_size=8,
    mixing_type="afno",
    in_channels=4,
    out_channels=4,
    in_timesteps=10,
    out_timesteps=1,
    n_blocks=8,
    embed_dim=1024,
    out_layer_dim=32,
    depth=12,
    modes=32,
    mlp_ratio=4.0,
    n_cls=12,
    normalize=False,
    act="gelu",
    time_agg="exp_mlp",
):
    """DPOTNet-M (Medium). embed=1024, depth=12, n_blocks=8."""
    return _build(**{k: v for k, v in locals().items()})


def L(
    img_size=128,
    patch_size=8,
    mixing_type="afno",
    in_channels=4,
    out_channels=4,
    in_timesteps=10,
    out_timesteps=1,
    n_blocks=16,
    embed_dim=1536,
    out_layer_dim=128,
    depth=24,
    modes=32,
    mlp_ratio=4.0,
    n_cls=12,
    normalize=False,
    act="gelu",
    time_agg="exp_mlp",
):
    """DPOTNet-L (Large). embed=1536, depth=24, n_blocks=16."""
    return _build(**{k: v for k, v in locals().items()})


def H(
    img_size=128,
    patch_size=8,
    mixing_type="afno",
    in_channels=4,
    out_channels=4,
    in_timesteps=10,
    out_timesteps=1,
    n_blocks=8,
    embed_dim=2048,
    out_layer_dim=128,
    depth=27,
    modes=32,
    mlp_ratio=4.0,
    n_cls=12,
    normalize=False,
    act="gelu",
    time_agg="exp_mlp",
):
    """DPOTNet-H (Huge). embed=2048, depth=27, n_blocks=8."""
    return _build(**{k: v for k, v in locals().items()})


ti = Ti
s = S
m = M
l = L
h = H

__all__ = ["Ti", "S", "M", "L", "H", "ti", "s", "m", "l", "h"]

_callable_module.install(__name__, _build)
