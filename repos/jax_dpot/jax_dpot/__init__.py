from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jax_dpot")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from .configs import DPOT_CONFIGS, dpot_h, dpot_l, dpot_m, dpot_s, dpot_ti
from .convert_weights import convert_pytorch_to_jax_params, load_pytorch_state_dict
from .model import DPOTNet
from .model_3d import DPOTNet3D
from .model_res import CDPOTNet
from .utils import resize_pos_embed

__all__ = [
    "__version__",
    "CDPOTNet",
    "DPOT_CONFIGS",
    "DPOTNet",
    "DPOTNet3D",
    "convert_pytorch_to_jax_params",
    "dpot_h",
    "dpot_l",
    "dpot_m",
    "dpot_s",
    "dpot_ti",
    "load_pytorch_state_dict",
    "resize_pos_embed",
]
