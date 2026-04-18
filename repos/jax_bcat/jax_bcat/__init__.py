from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jax_bcat")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from .configs import BCAT_CONFIGS
from .model_eqx import BCAT

__all__ = [
    "__version__",
    "BCAT",
    "BCAT_CONFIGS",
]
