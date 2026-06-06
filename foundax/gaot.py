"""GAOT -- Geometry-Aware Operator Transformer (NeurIPS 2025).

**Paper:** Gao et al., *"GAOT: Geometry-Aware Operator Transformer for
Arbitrary-Geometry PDE Problems"* (2025).
https://arxiv.org/abs/2505.18781

Architecture: MAGNO encoder → UViT transformer → MAGNO decoder.
This module re-exports a faithful JAX/Equinox port of the upstream
``camlab-ethz/GAOT`` PyTorch implementation. Forward-pass numerical
equivalence is verified by ``scripts/compare_gaot.py``.

.. warning::

    The upstream GAOT repository (https://github.com/camlab-ethz/GAOT)
    carries **no code license** (all rights reserved by default).
    No pretrained weights have been released as of 2026-06.

Usage::

    model = foundax.gaot.S(input_size=2, output_size=1)
    model = foundax.gaot.M(input_size=2, output_size=1)
    model = foundax.gaot.L(input_size=2, output_size=1)
    model = foundax.gaot(input_size=2, output_size=1)
"""

from .architectures.gaot import (
    GAOT,
    MAGNOConfig,
    TransformerConfig,
    AttentionConfig,
    AGNO,
    GeometricEmbedding,
    MAGNOEncoder,
    MAGNODecoder,
    compute_neighbors_csr,
    compute_neighbors,
    S,
    M,
    L,
    s,
    m,
    l,  # noqa: E741
)
from .nn import gaot as _build

__all__ = [
    "compute_neighbors_csr",
    "compute_neighbors",
    "MAGNOConfig",
    "TransformerConfig",
    "AttentionConfig",
    "GAOT",
    "AGNO",
    "GeometricEmbedding",
    "MAGNOEncoder",
    "MAGNODecoder",
    "S",
    "M",
    "L",
    "s",
    "m",
    "l",
]

from . import _callable_module

_callable_module.install(__name__, _build)
