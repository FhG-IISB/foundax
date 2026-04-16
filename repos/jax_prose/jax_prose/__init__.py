from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("jax_prose")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from .config import (
    PROSE1to1Config,
    prose_fd_1to1_default_config,
)
from .prose_fd import PROSE1to1
from .prose_fd_2to1 import PROSE2to1
from .prose_ode_pde_2to1 import PROSEODE2to1, PROSEPDE2to1, ProseTextData2to1Config
from .load import prose_fd_1to1, prose_fd_2to1, prose_ode_2to1, prose_pde_2to1

__all__ = [
    "__version__",
    "PROSE1to1Config",
    "prose_fd_1to1_default_config",
    "PROSE1to1",
    "PROSE2to1",
    "ProseTextData2to1Config",
    "PROSEODE2to1",
    "PROSEPDE2to1",
    "prose_fd_1to1",
    "prose_fd_2to1",
    "prose_ode_2to1",
    "prose_pde_2to1",
]
