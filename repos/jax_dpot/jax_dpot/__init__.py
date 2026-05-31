from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jax_dpot")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from .configs import DPOT_CONFIGS
from .model_eqx import DPOTNet
from .utils import resize_pos_embed
from .convert_weights import (
    load_pytorch_state_dict,
    convert_pytorch_to_jax_params,
    load_jax_params,
)

__all__ = [
    "__version__",
    "DPOT_CONFIGS",
    "DPOTNet",
    "resize_pos_embed",
    "load_pytorch_state_dict",
    "convert_pytorch_to_jax_params",
    "load_jax_params",
]
