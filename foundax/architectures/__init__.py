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
from .kan import (
    # bases
    BSplineBasis as BSplineBasis,
    RBFBasis as RBFBasis,
    FourierBasis as FourierBasis,
    ChebyshevBasis as ChebyshevBasis,
    JacobiBasis as JacobiBasis,
    LegendreBasis as LegendreBasis,
    WaveletBasis as WaveletBasis,
    TaylorBasis as TaylorBasis,
    HermiteBasis as HermiteBasis,
    LaguerreBasis as LaguerreBasis,
    BernsteinBasis as BernsteinBasis,
    ReLUKANBasis as ReLUKANBasis,
    RationalBasis as RationalBasis,
    SincBasis as SincBasis,
    GramBasis as GramBasis,
    BSRBFBasis as BSRBFBasis,
    # layers
    KANLayer as KANLayer,
    EfficientKANLayer as EfficientKANLayer,
    FastKANLayer as FastKANLayer,
    FourierKANLayer as FourierKANLayer,
    ChebyshevKANLayer as ChebyshevKANLayer,
    JacobiKANLayer as JacobiKANLayer,
    LegendreKANLayer as LegendreKANLayer,
    WaveletKANLayer as WaveletKANLayer,
    TaylorKANLayer as TaylorKANLayer,
    HermiteKANLayer as HermiteKANLayer,
    LaguerreKANLayer as LaguerreKANLayer,
    BernsteinKANLayer as BernsteinKANLayer,
    ReLUKANLayer as ReLUKANLayer,
    RationalKANLayer as RationalKANLayer,
    SincKANLayer as SincKANLayer,
    GramKANLayer as GramKANLayer,
    BSRBFKANLayer as BSRBFKANLayer,
    # convolutional + structural blocks
    KANConv1d as KANConv1d,
    KANConv2d as KANConv2d,
    KANConv3d as KANConv3d,
    KANResBlock as KANResBlock,
    KANSpectralBlock1d as KANSpectralBlock1d,
    KANSpectralBlock2d as KANSpectralBlock2d,
    KANSpectralBlock3d as KANSpectralBlock3d,
    KANAttentionBlock as KANAttentionBlock,
    # networks
    KAN as KAN,
    EfficientKAN as EfficientKAN,
    FastKAN as FastKAN,
    FourierKAN as FourierKAN,
    ChebyshevKAN as ChebyshevKAN,
    JacobiKAN as JacobiKAN,
    LegendreKAN as LegendreKAN,
    WaveletKAN as WaveletKAN,
    TaylorKAN as TaylorKAN,
    HermiteKAN as HermiteKAN,
    LaguerreKAN as LaguerreKAN,
    BernsteinKAN as BernsteinKAN,
    ReLUKAN as ReLUKAN,
    RationalKAN as RationalKAN,
    SincKAN as SincKAN,
    GramKAN as GramKAN,
    BSRBFKAN as BSRBFKAN,
)
