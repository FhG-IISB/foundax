"""Walrus -- 1.29 B Parameter Foundation Model (PolymathicAI).

**Paper:** Bodner et al., *"Aurora: A Foundation Model of the Atmosphere"* (2024)
https://github.com/nubskr/walrus

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
    """Walrus base ~1.29 B params. IsotropicModel with default config."""
    return _build(**{k: v for k, v in locals().items()})


v1 = base
default = base


__all__ = ["base", "v1", "default"]

_callable_module.install(__name__, _build)
