"""Poseidon — Efficient Foundation Models for PDEs.

Lazy-loading factories for the vendored ``jax_poseidon`` package.
Each function forwards all arguments to the corresponding upstream
constructor after adding the vendored repository to ``sys.path``.

**Paper:** Herde et al., *"Poseidon: Efficient Foundation Models for PDEs"* (2024)
https://arxiv.org/abs/2405.19101

Architecture: Scalable Operator Transformer (ScOT) — Swin-Transformer
backbone with U-Net-style skip connections for multi-scale operator
learning. Weights are stored as ``.msgpack`` files.

Variants (parameter counts):

===========  ===========
Variant      Parameters
===========  ===========
Poseidon-T   ~20.8 M
Poseidon-B   ~157.7 M
Poseidon-L   ~628.6 M
===========  ===========
"""

import importlib
from typing import Any, Optional, Tuple, Union

from ._vendors import ensure_repo_on_path


def T(
    rng: Optional[Any] = None,
    weight_path: Optional[str] = None,
    verbose: bool = True,
) -> Any:
    """Create the Poseidon-T (Tiny) foundation model (~20.8 M params).

    Delegates to ``jax_poseidon.poseidonT``.

    Parameters
    ----------
    rng : jax.random.PRNGKey, optional
        PRNG key for parameter initialisation.  Required together with
        *weight_path* to obtain loaded weights.
    weight_path : str, optional
        Path to a ``.msgpack`` checkpoint.  When provided together with
        *rng*, the function returns ``(model, params)`` with pretrained
        weights merged into freshly initialised parameters.
    verbose : bool, default True
        Print model / weight-loading information.

    Returns
    -------
    ScOT or tuple[ScOT, dict]
        If *weight_path* **and** *rng* are both given: ``(model, params)``.
        Otherwise: an uninitialised ``ScOT`` module.

    References
    ----------
    Herde et al., "Poseidon: Efficient Foundation Models for PDEs", 2024.
    https://arxiv.org/abs/2405.19101
    """
    ensure_repo_on_path("jax_poseidon")
    return importlib.import_module("jax_poseidon").poseidonT(rng=rng, weight_path=weight_path, verbose=verbose)


def B(
    rng: Optional[Any] = None,
    weight_path: Optional[str] = None,
    verbose: bool = True,
) -> Any:
    """Create the Poseidon-B (Base) foundation model (~157.7 M params).

    Delegates to ``jax_poseidon.poseidonB``.

    Parameters
    ----------
    rng : jax.random.PRNGKey, optional
        PRNG key for parameter initialisation.
    weight_path : str, optional
        Path to a ``.msgpack`` checkpoint.
    verbose : bool, default True
        Print model / weight-loading information.

    Returns
    -------
    ScOT or tuple[ScOT, dict]
        If *weight_path* **and** *rng* are both given: ``(model, params)``.
        Otherwise: an uninitialised ``ScOT`` module.

    References
    ----------
    Herde et al., "Poseidon: Efficient Foundation Models for PDEs", 2024.
    https://arxiv.org/abs/2405.19101
    """
    ensure_repo_on_path("jax_poseidon")
    return importlib.import_module("jax_poseidon").poseidonB(rng=rng, weight_path=weight_path, verbose=verbose)


def L(
    rng: Optional[Any] = None,
    weight_path: Optional[str] = None,
    verbose: bool = True,
) -> Any:
    """Create the Poseidon-L (Large) foundation model (~628.6 M params).

    Delegates to ``jax_poseidon.poseidonL``.

    Parameters
    ----------
    rng : jax.random.PRNGKey, optional
        PRNG key for parameter initialisation.
    weight_path : str, optional
        Path to a ``.msgpack`` checkpoint.
    verbose : bool, default True
        Print model / weight-loading information.

    Returns
    -------
    ScOT or tuple[ScOT, dict]
        If *weight_path* **and** *rng* are both given: ``(model, params)``.
        Otherwise: an uninitialised ``ScOT`` module.

    References
    ----------
    Herde et al., "Poseidon: Efficient Foundation Models for PDEs", 2024.
    https://arxiv.org/abs/2405.19101
    """
    ensure_repo_on_path("jax_poseidon")
    return importlib.import_module("jax_poseidon").poseidonL(rng=rng, weight_path=weight_path, verbose=verbose)


t = T
b = B
l = L


__all__ = ["T", "B", "L", "t", "b", "l"]
