"""Walrus — 1.29 B Parameter Foundation Model (PolymathicAI).

Lazy-loading factory for the vendored ``jax_walrus`` package.

**Related:** Bodner et al., *"Aurora: A Foundation Model of the Atmosphere"* (2024)
https://github.com/nubskr/walrus

Architecture: Isotropic encoder-processor-decoder with grouped convolutions,
windowed multi-head attention, and SpaceBag augmented input.
Weights format: ``.msgpack``.

The ``IsotropicModel`` Flax module accepts channels-last input
``(B, T, H, [W, [D]], C)`` and returns ``(B, T_out, H, [W, [D]], C_out)``.

Default model attributes::

    hidden_dim         = 768
    intermediate_dim   = 192
    n_states           = 4
    processor_blocks   = 12
    groups             = 16
    num_heads          = 12
    max_d              = 3
    causal_in_time     = False
    drop_path          = 0.05
    input_field_drop   = 0.1
    bias_type          = "rel"
    base_kernel_size   = ((8, 4), (8, 4), (8, 4))
    use_spacebag       = True
    use_silu           = True
    include_d          = (2, 3)
    encoder_groups     = 16
    jitter_patches     = True
    learned_pad        = True
    remat              = True
"""

import importlib
from typing import Any, Optional, Tuple

from ._vendors import ensure_repo_on_path


def base(**kwargs: Any) -> Any:
    """Create the default Walrus ``IsotropicModel`` (~1.29 B params).

    Delegates to ``jax_walrus.IsotropicModel``.

    Parameters
    ----------
    **kwargs
        Override any ``IsotropicModel`` attribute listed below.

        hidden_dim : int, default 768
            Latent embedding dimension.
        intermediate_dim : int, default 192
            Intermediate projection dim inside processor blocks.
        n_states : int, default 4
            Number of time-state channels.
        processor_blocks : int, default 12
            Number of processor transformer blocks.
        groups : int, default 16
            Group count for grouped convolutions.
        num_heads : int, default 12
            Number of attention heads.
        max_d : int, default 3
            Maximum spatial dimensionality (2-D / 3-D).
        causal_in_time : bool, default False
            Enable causal masking along the time axis.
        drop_path : float, default 0.05
            Stochastic depth rate.
        input_field_drop : float, default 0.1
            Input field dropout rate.
        bias_type : str, default ``"rel"``
            Positional bias type.
        base_kernel_size : tuple, default ``((8, 4), (8, 4), (8, 4))``
            Kernel sizes for base convolutions per spatial dim.
        use_spacebag : bool, default True
            Use SpaceBag channel augmentation.
        use_silu : bool, default True
            Use SiLU activation.
        include_d : tuple[int, ...], default ``(2, 3)``
            Spatial dims included in the model.
        encoder_groups : int, default 16
            Group count for encoder convolutions.
        jitter_patches : bool, default True
            Apply patch jittering during training.
        learned_pad : bool, default True
            Use learned padding.
        remat : bool, default True
            Enable gradient checkpointing (rematerialisation).

    Returns
    -------
    IsotropicModel
        Uninitialised Flax ``nn.Module``.

        Forward call signature::

            model.__call__(
                x,                              # (B, T, H, [W, [D]], C)
                state_labels,                   # (C_out,)
                bcs,                            # boundary conditions list
                stride1=None, stride2=None,
                field_indices=None, dim_key=None,
                deterministic=True,
            ) -> jnp.ndarray                    # (B, T_out, ..., C_out)

    References
    ----------
    https://github.com/nubskr/walrus
    """
    ensure_repo_on_path("jax_walrus")
    return importlib.import_module("jax_walrus").IsotropicModel(**kwargs)


def v1(**kwargs: Any) -> Any:
    """Create Walrus v1 — alias of :func:`base`."""
    return base(**kwargs)


default = base


__all__ = ["base", "v1", "default"]
