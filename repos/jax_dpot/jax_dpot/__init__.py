from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jax_dpot")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from .configs import DPOT_CONFIGS
from .model_eqx import DPOTNet
from .utils import resize_pos_embed

__all__ = [
    "__version__",
    "DPOT_CONFIGS",
    "DPOTNet",
    "resize_pos_embed",
]
