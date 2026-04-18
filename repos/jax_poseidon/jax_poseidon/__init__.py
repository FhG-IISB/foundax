"""
jax_poseidon: JAX/Equinox implementation of the Poseidon PDE foundation model.

Reference:
    Herde et al., "Poseidon: Efficient Foundation Models for PDEs" (2024)
    https://arxiv.org/abs/2405.19101
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("jax_poseidon")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from .scot_eqx import ScOT, ScOTConfig

__all__ = [
    "__version__",
    "ScOT",
    "ScOTConfig",
]
