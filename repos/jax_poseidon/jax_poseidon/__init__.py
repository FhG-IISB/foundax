"""
jax_poseidon: JAX/Flax translation of the Poseidon PDE foundation model.

A 1-to-1 translation of the Poseidon model architecture (ScOT / Swin Transformer V2
based encoder-decoder) from PyTorch to JAX/Flax, maintaining exact weight
compatibility for pretrained checkpoint conversion.

Reference:
    Herde et al., "Poseidon: Efficient Foundation Models for PDEs" (2024)
    https://arxiv.org/abs/2405.19101

.. note::
    This package is designed to be used with jNO
    (https://github.com/FhG-IISB/jNO).

.. warning::
    This is a research-level repository. It may contain bugs and is subject
    to continuous change without notice.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("jax_poseidon")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from .load import (
    poseidonB,
    poseidonL,
    poseidonT,
    scot,
    init_poseidon_with_weights,
    merge_pretrained_params,
)
from .scot import ScOT, ScOTConfig
from . import scot_eqx

__all__ = [
    "__version__",
    "poseidonB",
    "poseidonL",
    "poseidonT",
    "scot",
    "init_poseidon_with_weights",
    "merge_pretrained_params",
    "ScOT",
    "ScOTConfig",
    "scot_eqx",
]
