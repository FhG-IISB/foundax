"""
jax_mpp: JAX/Equinox implementation of the Multiple Physics Pretraining (MPP) model.

Reference: McCabe et al., "Multiple Physics Pretraining" (NeurIPS 2024)
https://openreview.net/forum?id=DKSI3bULiZ
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jax_mpp")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from jax_mpp.avit_eqx import AViT
from jax_mpp.configs import AVIT_CONFIGS
from jax_mpp.convert_weights import (
    convert_pytorch_to_jax_params,
    load_pytorch_state_dict,
)

__all__ = [
    "__version__",
    "AViT",
    "AVIT_CONFIGS",
    "load_pytorch_state_dict",
    "convert_pytorch_to_jax_params",
]
