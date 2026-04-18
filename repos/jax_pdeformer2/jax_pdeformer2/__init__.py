"""PDEformer-2 neural operator implemented in JAX/Equinox.

Reference:
    Shi et al., "PDEformer-2: A Foundation Model for Two-Dimensional PDEs" (2025)
    https://arxiv.org/abs/2502.14844
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("pdeformer2-jax")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from .model_eqx import PDEformer

__all__ = [
    "__version__",
    "PDEformer",
]
