"""Composable operator-layer primitives for building custom neural operators.

All blocks follow the channel-last, unbatched convention:

    1-D fields : ``(W, C)``
    2-D fields : ``(H, W, C)``
    3-D fields : ``(D, H, W, C)``

Batching is handled by the caller via ``jax.vmap``.

Example::

    import foundax as fx
    import foundax.layers as fl
    import jax

    ks = jax.random.split(jax.random.PRNGKey(0), 3)

    # Build a custom operator from individual blocks
    b1 = fx.block(fl.SpectralBlock2d(3,  32, n_modes=16, key=ks[0]))
    b2 = fx.block(fl.SpectralBlock2d(32, 32, n_modes=16, key=ks[1]))
    b3 = fx.block(fl.SpectralBlock2d(32,  1, n_modes=16, key=ks[2]))

    model = b1 > b2 > b3
"""

from .blocks import SpectralBlock1d, SpectralBlock2d, SpectralBlock3d

# Re-export raw spectral convolutions and common primitives so users can
# access everything from a single namespace.
from foundax.architectures.fno import (
    SpectralConv1d,
    SpectralConv2d,
    SpectralConv3d,
)
from foundax.architectures.mlp import MLP
from foundax.architectures.linear import Linear
from foundax.architectures.ffno import (
    FactorizedSpectralBlock2d,
    FactorizedSpectralBlock3d,
)
from foundax.architectures.wno import (
    WaveletBlock1d,
    WaveletBlock2d,
    WaveletBlock3d,
)

from foundax.architectures.kan import (
    KANLayer,
    EfficientKANLayer,
    FastKANLayer,
    FourierKANLayer,
    ChebyshevKANLayer,
    JacobiKANLayer,
    LegendreKANLayer,
    WaveletKANLayer,
    TaylorKANLayer,
    HermiteKANLayer,
    LaguerreKANLayer,
    BernsteinKANLayer,
    ReLUKANLayer,
    RationalKANLayer,
    SincKANLayer,
    GramKANLayer,
    BSRBFKANLayer,
    KANConv1d,
    KANConv2d,
    KANConv3d,
    KANResBlock,
    KANSpectralBlock1d,
    KANSpectralBlock2d,
    KANSpectralBlock3d,
    KANAttentionBlock,
)

__all__ = [
    "SpectralBlock1d",
    "SpectralBlock2d",
    "SpectralBlock3d",
    "SpectralConv1d",
    "SpectralConv2d",
    "SpectralConv3d",
    "MLP",
    "Linear",
    # Flow-matching backbones
    "FactorizedSpectralBlock2d",
    "FactorizedSpectralBlock3d",
    "WaveletBlock1d",
    "WaveletBlock2d",
    "WaveletBlock3d",
    # KAN layer primitives
    "KANLayer",
    "EfficientKANLayer",
    "FastKANLayer",
    "FourierKANLayer",
    "ChebyshevKANLayer",
    "JacobiKANLayer",
    "LegendreKANLayer",
    "WaveletKANLayer",
    "TaylorKANLayer",
    "HermiteKANLayer",
    "LaguerreKANLayer",
    "BernsteinKANLayer",
    "ReLUKANLayer",
    "RationalKANLayer",
    "SincKANLayer",
    "GramKANLayer",
    "BSRBFKANLayer",
    # KAN convolutional + structural blocks
    "KANConv1d",
    "KANConv2d",
    "KANConv3d",
    "KANResBlock",
    "KANSpectralBlock1d",
    "KANSpectralBlock2d",
    "KANSpectralBlock3d",
    "KANAttentionBlock",
]
