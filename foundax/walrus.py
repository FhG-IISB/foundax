"""Walrus -- 1.29 B Parameter Foundation Model (PolymathicAI).

**Paper:** Bodner et al., *"Walrus: A 1.29B Parameter Foundation Model
for Multi-Physics"* (2024)
https://arxiv.org/abs/2511.15684

Architecture: Isotropic encoder-processor-decoder with grouped convolutions,
windowed multi-head attention, and SpaceBag augmented input.

Usage::

    model = foundax.walrus.base(hidden_dim=512)
    model = foundax.walrus(processor_blocks=24)
"""

import importlib

from ._vendors import ensure_repo_on_path
from . import _callable_module


def _build(
    hidden_dim=768,
    intermediate_dim=192,
    n_states=4,
    processor_blocks=12,
    groups=16,
    num_heads=12,
    mlp_dim=0,
    max_d=3,
    causal_in_time=False,
    drop_path=0.05,
    bias_type="rel",
    base_kernel_size=((8, 4), (8, 4), (8, 4)),
    use_spacebag=True,
    use_silu=True,
    include_d=(2, 3),
    encoder_groups=16,
    learned_pad=True,
    *,
    key=None,
):
    import jax

    ensure_repo_on_path("jax_walrus")
    mod = importlib.import_module("jax_walrus.model_eqx")
    if key is None:
        key = jax.random.PRNGKey(0)
    kw = {k: v for k, v in locals().items() if k not in ("mod", "jax")}
    return mod.IsotropicModel(**kw)


def base(
    hidden_dim=768,
    intermediate_dim=192,
    n_states=4,
    processor_blocks=12,
    groups=16,
    num_heads=12,
    mlp_dim=0,
    max_d=3,
    causal_in_time=False,
    drop_path=0.05,
    bias_type="rel",
    base_kernel_size=((8, 4), (8, 4), (8, 4)),
    use_spacebag=True,
    use_silu=True,
    include_d=(2, 3),
    encoder_groups=16,
    learned_pad=True,
    *,
    key=None,
):
    """Walrus base ~1.29 B params.

    Isotropic encoder-processor-decoder with grouped convolutions,
    windowed multi-head attention, and SpaceBag augmented input for
    unified 1-D / 2-D / 3-D operator learning (PolymathicAI).

    Reference:
        Bodner et al., *Walrus: A 1.29B Parameter Foundation Model
        for Multi-Physics* (2024).
        https://arxiv.org/abs/2511.15684

    Example::

        model = foundax.walrus.base(n_states=4, hidden_dim=512)

    Shape:
        - Input: ``(B, T, H, W, C)`` where
          ``C = n_states + 3`` (state channels + coordinates),
          + state_labels ``(C,)``.
        - Output: ``(B, 1, H, W, n_states)``

    Args:
        hidden_dim: Hidden feature dimension throughout the processor.
        intermediate_dim: Intermediate dimension in encoder / decoder.
        n_states: Number of active physical state variables (channels).
        processor_blocks: Number of space-time processor blocks.
        groups: Number of groups for grouped convolutions.
        num_heads: Number of attention heads.
        mlp_dim: MLP hidden dimension (``0`` = disable MLP branch).
        max_d: Maximum spatial dimensionality (1\u20133).
        causal_in_time: Apply causal masking along the time axis.
        drop_path: Stochastic depth rate.
        bias_type: Attention bias type (``"rel"`` = relative position).
        base_kernel_size: Base convolution kernel sizes per dimension,
            given as ``((enc, dec), ...)`` for each spatial dim.
        use_spacebag: Use the SpaceBag encoder variant.
        use_silu: Use SiLU activation in encoder / decoder.
        include_d: Which spatial dimensionalities to include.
        encoder_groups: Number of groups for encoder convolutions.
        learned_pad: Learn padding values instead of zeros.
        key: JAX PRNG key (``None`` \u2192 ``PRNGKey(0)``).

    Returns:
        An ``equinox.Module`` (IsotropicModel).
    """
    return _build(**{k: v for k, v in locals().items()})


v1 = base
default = base


__all__ = ["base", "v1", "default"]

_callable_module.install(__name__, _build)
