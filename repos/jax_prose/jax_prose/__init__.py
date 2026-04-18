from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("jax_prose")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from .config import (
    PROSE1to1Config,
    prose_fd_1to1_default_config,
)
from .model_eqx import (
    PROSE1to1,
    PROSE2to1,
    PROSEODE2to1,
    PROSEPDE2to1,
)

__all__ = [
    "__version__",
    "PROSE1to1Config",
    "prose_fd_1to1_default_config",
    "PROSE1to1",
    "PROSE2to1",
    "PROSEODE2to1",
    "PROSEPDE2to1",
]
