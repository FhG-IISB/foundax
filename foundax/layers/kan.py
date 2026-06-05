"""KAN layer + structural-block namespace.

Access pattern (names drop the redundant ``KAN``/``Layer`` parts that the
namespace already conveys)::

    import foundax as fx

    # 17 KAN layer variants
    layer = fx.layers.kan.kan(in_features=4, out_features=8, key=k)        # original B-spline
    layer = fx.layers.kan.efficient(...)
    layer = fx.layers.kan.fast(...)
    layer = fx.layers.kan.fourier(...)
    layer = fx.layers.kan.chebyshev(...)
    layer = fx.layers.kan.jacobi(...)
    layer = fx.layers.kan.legendre(...)
    layer = fx.layers.kan.wavelet(...)
    layer = fx.layers.kan.taylor(...)
    layer = fx.layers.kan.hermite(...)
    layer = fx.layers.kan.laguerre(...)
    layer = fx.layers.kan.bernstein(...)
    layer = fx.layers.kan.relu(...)
    layer = fx.layers.kan.rational(...)
    layer = fx.layers.kan.sinc(...)
    layer = fx.layers.kan.gram(...)
    layer = fx.layers.kan.bsrbf(...)

    # Convolutional + structural blocks
    block = fx.layers.kan.conv1d(...); fx.layers.kan.conv2d(...); fx.layers.kan.conv3d(...)
    block = fx.layers.kan.res_block(...)
    block = fx.layers.kan.spectral_block1d/2d/3d(...)
    block = fx.layers.kan.attention_block(...)
"""

from foundax.architectures.kan import (
    KANLayer as kan,
    EfficientKANLayer as efficient,
    FastKANLayer as fast,
    FourierKANLayer as fourier,
    ChebyshevKANLayer as chebyshev,
    JacobiKANLayer as jacobi,
    LegendreKANLayer as legendre,
    WaveletKANLayer as wavelet,
    TaylorKANLayer as taylor,
    HermiteKANLayer as hermite,
    LaguerreKANLayer as laguerre,
    BernsteinKANLayer as bernstein,
    ReLUKANLayer as relu,
    RationalKANLayer as rational,
    SincKANLayer as sinc,
    GramKANLayer as gram,
    BSRBFKANLayer as bsrbf,
    KANConv1d as conv1d,
    KANConv2d as conv2d,
    KANConv3d as conv3d,
    KANResBlock as res_block,
    KANSpectralBlock1d as spectral_block1d,
    KANSpectralBlock2d as spectral_block2d,
    KANSpectralBlock3d as spectral_block3d,
    KANAttentionBlock as attention_block,
)


__all__ = [
    # 17 KAN layer variants
    "kan",
    "efficient",
    "fast",
    "fourier",
    "chebyshev",
    "jacobi",
    "legendre",
    "wavelet",
    "taylor",
    "hermite",
    "laguerre",
    "bernstein",
    "relu",
    "rational",
    "sinc",
    "gram",
    "bsrbf",
    # convolutional + structural blocks
    "conv1d",
    "conv2d",
    "conv3d",
    "res_block",
    "spectral_block1d",
    "spectral_block2d",
    "spectral_block3d",
    "attention_block",
]
