"""Equinox architecture implementations for neural operators."""

from .linear import Linear
from .common import BatchNorm, get_activation, compute_Fourier_modes, Conv2d, ConvTranspose2d
from .mlp import MLP
from .fno import SpectralConv1d, SpectralConv2d, SpectralConv3d, SpectralLayers1d, SpectralLayers2d, SpectralLayers3d, FNO1D, FNO2D, FNO3D
from .unet import UNet1D, UNet2D, UNet3D
from .transformer import Transformer, TransformerEncoder, TransformerDecoder
from .deeponet import DeepONet
from .cno import CNO2D
from .mgno import MgNO, MgNO1D
from .geofno import GeoFNO
from .pcno import PCNO
from .gnot import CGPTNO, GNOT, MoEGPTNO
from .pit import PiT, PiTWithCoords
from .pointnet import PointNet
