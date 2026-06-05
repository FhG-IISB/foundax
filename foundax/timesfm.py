"""TimesFM -- Time-series Foundation Model (Google Research).

**Paper:** Das et al. (2024). *A decoder-only foundation model for time-series
forecasting.* https://arxiv.org/abs/2310.10688
**Upstream:** https://github.com/google-research/timesfm
**Weights:** ``google/timesfm-2.5-200m-flax`` (and ``-torch``) on Hugging Face

This is a **wrap**, not a port — TimesFM 2.5 is a Flax NNX model maintained
by Google. We expose it through a thin ``eqx.Module`` shell so it
composes with the rest of foundax (``fx.block``, ``fx.timesfm(...)`` etc.).
The wrapping pattern is appropriate here because:

- TimesFM is an inference model with pretrained weights; users want the
  weights, not architecture surgery.
- Google maintains the Flax implementation in lockstep with the published
  checkpoints.
- Re-porting 200M parameters of NNX code into Equinox is high-effort and
  high-maintenance with no functional gain.

Trade-off: the wrapped instance is stored as a static field, so the params
are not exposed as Equinox pytree leaves. ``eqx.filter_grad`` on this
module will not find trainable arrays — by design, since TimesFM 2.5 is
not meant to be fine-tuned through this surface. If you need a gradient-
trainable surface, drop down to the upstream ``nnx.split`` / ``nnx.merge``
machinery directly.

Usage::

    import foundax as fx
    import numpy as np

    model = fx.timesfm.flax_200m(max_context=2048, max_horizon=256)
    point, quantile = model(
        horizon=24,
        inputs=[np.sin(np.linspace(0, 50, 512))],
    )

    # default callable shortcut == flax_200m()
    model = fx.timesfm()
"""

from __future__ import annotations

from typing import Any

import equinox as eqx

from . import _callable_module


class _TimesFMWrap(eqx.Module):
    """Thin Equinox shell around a compiled ``TimesFM_2p5_200M_*`` instance."""

    _instance: Any = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    max_context: int = eqx.field(static=True)
    max_horizon: int = eqx.field(static=True)

    def __init__(
        self,
        instance,
        backend: str,
        max_context: int,
        max_horizon: int,
    ):
        self._instance = instance
        self.backend = backend
        self.max_context = max_context
        self.max_horizon = max_horizon

    def __call__(self, horizon: int, inputs):
        """Forecast ``horizon`` steps ahead given ``inputs`` (list of 1-D arrays).

        Returns ``(point_forecast, quantile_forecast)``:
          - ``point_forecast``: ``(B, horizon)`` median point forecast
          - ``quantile_forecast``: ``(B, horizon, num_quantiles)``
        """
        return self._instance.forecast(horizon, inputs)


def _build_flax(
    model_id: str = "google/timesfm-2.5-200m-flax",
    max_context: int = 2048,
    max_horizon: int = 256,
    per_core_batch_size: int = 1,
    normalize_inputs: bool = True,
    **hf_kwargs,
) -> _TimesFMWrap:
    from timesfm import ForecastConfig, TimesFM_2p5_200M_flax

    instance = TimesFM_2p5_200M_flax.from_pretrained(model_id, **hf_kwargs)
    fc = ForecastConfig(
        max_context=max_context,
        max_horizon=max_horizon,
        per_core_batch_size=per_core_batch_size,
        normalize_inputs=normalize_inputs,
    )
    instance.compile(forecast_config=fc, dryrun=False)
    return _TimesFMWrap(
        instance=instance,
        backend="flax",
        max_context=fc.max_context,
        max_horizon=fc.max_horizon,
    )


def _build_torch(
    model_id: str = "google/timesfm-2.5-200m-pytorch",
    max_context: int = 2048,
    max_horizon: int = 256,
    per_core_batch_size: int = 1,
    normalize_inputs: bool = True,
    **hf_kwargs,
) -> _TimesFMWrap:
    from timesfm import ForecastConfig, TimesFM_2p5_200M_torch

    instance = TimesFM_2p5_200M_torch.from_pretrained(model_id, **hf_kwargs)
    fc = ForecastConfig(
        max_context=max_context,
        max_horizon=max_horizon,
        per_core_batch_size=per_core_batch_size,
        normalize_inputs=normalize_inputs,
    )
    instance.compile(forecast_config=fc)
    return _TimesFMWrap(
        instance=instance,
        backend="torch",
        max_context=fc.max_context,
        max_horizon=fc.max_horizon,
    )


def flax_200m(
    max_context: int = 2048,
    max_horizon: int = 256,
    per_core_batch_size: int = 1,
    normalize_inputs: bool = True,
    model_id: str = "google/timesfm-2.5-200m-flax",
    **hf_kwargs,
) -> _TimesFMWrap:
    """TimesFM 2.5, 200M parameters, Flax NNX backend.

    Downloads the pretrained checkpoint from Hugging Face on first call
    (cached afterwards). Compiles the model for the given ``max_context``
    and ``max_horizon``.

    Args:
        max_context: Maximum input series length (will be rounded up to a
            multiple of ``32``, the input patch size).
        max_horizon: Maximum forecast horizon (rounded up to a multiple of
            ``128``, the output patch size).
        per_core_batch_size: Batch size per device for compiled inference.
        normalize_inputs: Whether to normalise inputs before forecasting
            (recommended when raw series magnitudes are extreme).
        model_id: Hugging Face repo id of the checkpoint.
        **hf_kwargs: Forwarded to ``huggingface_hub.snapshot_download``
            (``revision``, ``cache_dir``, ``token``, etc.).

    Returns:
        An ``eqx.Module`` shell wrapping the compiled Flax instance.
    """
    return _build_flax(
        model_id=model_id,
        max_context=max_context,
        max_horizon=max_horizon,
        per_core_batch_size=per_core_batch_size,
        normalize_inputs=normalize_inputs,
        **hf_kwargs,
    )


def torch_200m(
    max_context: int = 2048,
    max_horizon: int = 256,
    per_core_batch_size: int = 1,
    normalize_inputs: bool = True,
    model_id: str = "google/timesfm-2.5-200m-pytorch",
    **hf_kwargs,
) -> _TimesFMWrap:
    """TimesFM 2.5, 200M parameters, PyTorch backend.

    Same API as :func:`flax_200m`, useful for cross-implementation parity
    checks (see ``scripts/compare_timesfm.py``).
    """
    return _build_torch(
        model_id=model_id,
        max_context=max_context,
        max_horizon=max_horizon,
        per_core_batch_size=per_core_batch_size,
        normalize_inputs=normalize_inputs,
        **hf_kwargs,
    )


default = flax_200m


__all__ = ["flax_200m", "torch_200m", "default"]

_callable_module.install(__name__, flax_200m)
