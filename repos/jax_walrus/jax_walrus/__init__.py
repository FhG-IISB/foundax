"""
Walrus-JAX: JAX/Equinox implementation of the Walrus PDE foundation model.

Reference:
    Bodner et al., "Aurora: A Foundation Model of the Atmosphere" (2024)
    https://github.com/PolymathicAI/the_well
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("jax_walrus")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from jax_walrus.model_eqx import IsotropicModel, transfer_weights

__all__ = ["__version__", "IsotropicModel", "transfer_weights"]
