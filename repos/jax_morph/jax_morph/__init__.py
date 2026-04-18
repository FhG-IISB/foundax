"""
jax_morph: JAX/Equinox implementation of the MORPH PDE foundation model.

Reference:
    Rautela et al., "MORPH: PDE Foundation Models with Arbitrary Data Modality" (2025)
    https://arxiv.org/abs/2509.21670
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("jax_morph")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from jax_morph.model_eqx import ViT3DRegression
from jax_morph.configs import (
    MORPH_CONFIGS,
    CHECKPOINT_NAMES,
    HF_REPO_ID,
)

__all__ = [
    "__version__",
    "ViT3DRegression",
    "MORPH_CONFIGS",
    "CHECKPOINT_NAMES",
    "HF_REPO_ID",
]
