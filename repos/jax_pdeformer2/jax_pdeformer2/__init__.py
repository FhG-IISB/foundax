"""PDEformer-2 neural operator implemented in JAX/Flax.

This package provides a JAX/Flax implementation of the PDEformer-2 model,
originally implemented in MindSpore. It includes:

- Full model architecture (PDEformer, PDEEncoder)
- Weight conversion utilities
- Configurations for all three pretrained variants (Small, Base, Fast)

Usage:
    from jax_pdeformer2 import PDEformer, create_pdeformer_from_config, PDEFORMER_SMALL_CONFIG
    from jax_pdeformer2.utils import load_pdeformer_weights, create_dummy_inputs

    # Create model from config
    model = create_pdeformer_from_config({"model": PDEFORMER_SMALL_CONFIG})

    # Or load with weights (requires converted .npz file)
    model, params = load_pdeformer_weights("pdeformer2-small.npz")

Reference:
    Shi et al., "PDEformer-2: A Foundation Model for Two-Dimensional PDEs" (2025)
    https://arxiv.org/abs/2502.14844

.. note::
    This package is designed to be used with jNO
    (https://github.com/FhG-IISB/jNO).

.. warning::
    This is a research-level repository. It may contain bugs and is subject
    to continuous change without notice.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("pdeformer2-jax")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from .pdeformer import (
    PDEformer,
    PDEEncoder,
    create_pdeformer_from_config,
    PDEFORMER_SMALL_CONFIG,
    PDEFORMER_BASE_CONFIG,
    PDEFORMER_FAST_CONFIG,
)
from .graphormer import (
    GraphormerEncoder,
    GraphormerEncoderLayer,
    MultiheadAttention,
    GraphNodeFeature,
    GraphAttnBias,
)
from .inr_with_hypernet import (
    PolyINR,
    PolyINRWithHypernet,
    get_inr_with_hypernet,
)
from .function_encoder import (
    Conv2dFuncEncoderV3,
    get_function_encoder,
)
from .basic_block import MLP, Sine, Scale, Clamp
from .utils import (
    load_pdeformer_weights,
    create_dummy_inputs,
    convert_mindspore_to_jax,
    load_numpy_weights,
)

__all__ = [
    # Main models
    "PDEformer",
    "PDEEncoder",
    "create_pdeformer_from_config",
    "PDEFORMER_SMALL_CONFIG",
    "PDEFORMER_BASE_CONFIG",
    "PDEFORMER_FAST_CONFIG",
    # Graphormer components
    "GraphormerEncoder",
    "GraphormerEncoderLayer",
    "MultiheadAttention",
    "GraphNodeFeature",
    "GraphAttnBias",
    # INR components
    "PolyINR",
    "PolyINRWithHypernet",
    "get_inr_with_hypernet",
    # Function encoder
    "Conv2dFuncEncoderV3",
    "get_function_encoder",
    # Basic blocks
    "MLP",
    "Sine",
    "Scale",
    "Clamp",
    # Utilities
    "load_pdeformer_weights",
    "create_dummy_inputs",
    "convert_mindspore_to_jax",
    "load_numpy_weights",
]
