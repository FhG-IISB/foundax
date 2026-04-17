from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jax_bcat")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from .configs import BCAT_CONFIGS, bcat_default
from .model import BCAT
from .model_eqx import BCAT as BCATEqx
from .convert_weights import load_pytorch_state_dict, convert_pytorch_to_jax_params

__all__ = [
    "__version__",
    "BCAT",
    "BCATEqx",
    "BCAT_CONFIGS",
    "bcat_default",
    "load_pytorch_state_dict",
    "convert_pytorch_to_jax_params",
]
