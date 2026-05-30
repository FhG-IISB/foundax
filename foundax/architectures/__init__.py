"""Equinox architecture implementations for neural operators."""

from .linear import Linear as Linear
from .common import (
    BatchNorm as BatchNorm,
    get_activation as get_activation,
    compute_Fourier_modes as compute_Fourier_modes,
    Conv2d as Conv2d,
    ConvTranspose2d as ConvTranspose2d,
)
from .mlp import MLP as MLP
from .fno import (
    SpectralConv1d as SpectralConv1d,
    SpectralConv2d as SpectralConv2d,
    SpectralConv3d as SpectralConv3d,
    SpectralLayers1d as SpectralLayers1d,
    SpectralLayers2d as SpectralLayers2d,
    SpectralLayers3d as SpectralLayers3d,
    FNO1D as FNO1D,
    FNO2D as FNO2D,
    FNO3D as FNO3D,
)
from .unet import UNet1D as UNet1D, UNet2D as UNet2D, UNet3D as UNet3D
from .transformer import (
    Transformer as Transformer,
    TransformerEncoder as TransformerEncoder,
    TransformerDecoder as TransformerDecoder,
)
from .deeponet import DeepONet as DeepONet
from .cno import CNO2D as CNO2D
from .mgno import MgNO as MgNO, MgNO1D as MgNO1D
from .geofno import GeoFNO as GeoFNO
from .pcno import PCNO as PCNO
from .gnot import CGPTNO as CGPTNO, GNOT as GNOT, MoEGPTNO as MoEGPTNO
from .pit import PiT as PiT, PiTWithCoords as PiTWithCoords
from .pointnet import PointNet as PointNet
from .time_embed import (
    SinusoidalTimeEmbedding as SinusoidalTimeEmbedding,
    FiLMLayer as FiLMLayer,
    AdaLayerNorm as AdaLayerNorm,
    AdaLayerNormZero as AdaLayerNormZero,
)
from .dit import DiT2d as DiT2d, DiT3d as DiT3d
from .ffno import (
    FactorizedSpectralConv2d as FactorizedSpectralConv2d,
    FactorizedSpectralConv3d as FactorizedSpectralConv3d,
    FactorizedSpectralBlock2d as FactorizedSpectralBlock2d,
    FactorizedSpectralBlock3d as FactorizedSpectralBlock3d,
    FFNO2d as FFNO2d,
    FFNO3d as FFNO3d,
)
from .wno import (
    WaveletBlock1d as WaveletBlock1d,
    WaveletBlock2d as WaveletBlock2d,
    WaveletBlock3d as WaveletBlock3d,
    WNO1d as WNO1d,
    WNO2d as WNO2d,
    WNO3d as WNO3d,
)
