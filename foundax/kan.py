"""KAN namespace — Kolmogorov-Arnold Networks.

Access pattern::

    import foundax as fx

    # Factories
    m = fx.kan(in_features=4, output_dim=1, key=k)            # original B-spline
    m = fx.kan.kan(in_features=4, output_dim=1, key=k)        # explicit original
    m = fx.kan.efficient(in_features=4, output_dim=1, key=k)
    m = fx.kan.fast(in_features=4, output_dim=1, key=k)
    m = fx.kan.fourier(...); m = fx.kan.chebyshev(...); ...   # 17 variants total

    # Structural blocks
    m = fx.kan.conv2d(in_channels=3, out_channels=8, key=k)
    m = fx.kan.res_block(...)
    m = fx.kan.spectral_block2d(...)
    m = fx.kan.attention_block(...)

The 17 variants are all thin constructors over the same ``_BasisKANLayer``
implementation in ``foundax/architectures/kan.py``; they differ only in
the basis-function family used to interpolate edge functions.
"""

from . import _callable_module
from . import nn as _nn

# ── 17 KAN variants (drop the _kan suffix — the namespace conveys it) ──────

kan = _nn.kan
efficient = _nn.efficient_kan
fast = _nn.fastkan
fourier = _nn.fourier_kan
chebyshev = _nn.chebyshev_kan
jacobi = _nn.jacobi_kan
legendre = _nn.legendre_kan
wavelet = _nn.wavelet_kan
taylor = _nn.taylor_kan
hermite = _nn.hermite_kan
laguerre = _nn.laguerre_kan
bernstein = _nn.bernstein_kan
relu = _nn.relu_kan
rational = _nn.rational_kan
sinc = _nn.sinc_kan
gram = _nn.gram_kan
bsrbf = _nn.bsrbf_kan

# ── structural blocks ───────────────────────────────────────────────────────

conv1d = _nn.kan_conv1d
conv2d = _nn.kan_conv2d
conv3d = _nn.kan_conv3d
res_block = _nn.kan_res_block
spectral_block1d = _nn.kan_spectral_block1d
spectral_block2d = _nn.kan_spectral_block2d
spectral_block3d = _nn.kan_spectral_block3d
attention_block = _nn.kan_attention_block

# ── default (callable-module shortcut: fx.kan(...) == fx.kan.kan(...)) ─────

default = kan


__all__ = [
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
    "conv1d",
    "conv2d",
    "conv3d",
    "res_block",
    "spectral_block1d",
    "spectral_block2d",
    "spectral_block3d",
    "attention_block",
    "default",
]

_callable_module.install(__name__, kan)
