"""TimesFM -- Time-series Foundation Model (Google Research).

**Paper:** Das et al. (2024). *A decoder-only foundation model for time-series
forecasting.* https://arxiv.org/abs/2310.10688
**Upstream:** https://github.com/google-research/timesfm
**Weights:** ``google/timesfm-2.5-200m-flax`` on Hugging Face

Thin ``eqx.Module`` wrap around Google's Flax NNX implementation. Splits
the upstream NNX module via ``nnx.split`` into a static ``GraphDef`` and
a JAX-array ``State`` pytree, so the wrapper:

- composes with ``eqx.filter_jit`` for compilation,
- **supports gradient training** via ``eqx.filter_grad`` (the entire
  ~200M-param ``state`` tree is a JAX pytree of leaves that participate
  in autodiff), and
- takes only ``jnp.ndarray`` inputs.

The horizon (forecast length) is fixed at model construction so it can
be static for JIT, and the model can specialise its decode-scan
configuration. A model instance forecasts one horizon; build multiple
instances if you need multiple.

API follows the foundax core-architecture convention: **channel-last,
unbatched** as the canonical single-series shape, with explicit
``(B, ...)`` batched shape also accepted for batched inference / training.

    input  : ``(context, 1)`` or ``(B, context, 1)`` float32
              ``context`` must be a multiple of 32 (the input patch size).
    output : ``(horizon, 1)`` or ``(B, horizon, 1)``
              median point forecast.
    mask   : optional, same shape as input, bool — True where the input
              is padding to ignore.

**Batching via ``jax.vmap`` is NOT supported** (NNX's internal
``nnx.scan`` conflicts with the outer vmap trace level). Use the
explicit ``(B, context, 1)`` form instead — JIT specialises per batch
shape, equally efficient.

Usage (inference)::

    import foundax as fx
    import jax.numpy as jnp, equinox as eqx

    model = fx.timesfm.small(horizon=24)     # = fx.timesfm(horizon=24)
    x = jnp.zeros((512, 1), dtype=jnp.float32)
    y = model(x)                              # (24, 1)
    fast = eqx.filter_jit(model)              # compile once per shape

Usage (fine-tuning)::

    import optax
    model = fx.timesfm.small(horizon=24, normalize_inputs=True)
    optimizer = optax.adamw(1e-5)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, x_batch, y_batch):
        def loss_fn(m):
            return jnp.mean((m(x_batch) - y_batch) ** 2)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        return eqx.apply_updates(model, updates), opt_state, loss
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp

from . import _callable_module


class _TimesFM(eqx.Module):
    """Equinox shell around an ``nnx.split`` view of the upstream Flax model."""

    _graphdef: Any = eqx.field(static=True)
    state: Any
    horizon: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)  # 32
    output_patch_size: int = eqx.field(static=True)  # 128
    decode_index: int = eqx.field(static=True)  # 5 (median quantile index)
    normalize_inputs: bool = eqx.field(static=True)

    def __init__(
        self,
        model_id: str,
        horizon: int,
        normalize_inputs: bool,
        **hf_kwargs,
    ):
        from flax import nnx
        from timesfm import TimesFM_2p5_200M_flax

        instance = TimesFM_2p5_200M_flax.from_pretrained(model_id, **hf_kwargs)
        nnx_model = instance.model
        graphdef, state = nnx.split(nnx_model)

        self._graphdef = graphdef
        self.state = state
        self.horizon = int(horizon)
        self.patch_size = int(nnx_model.p)
        self.output_patch_size = int(nnx_model.o)
        self.decode_index = int(nnx_model.decode_index)
        self.normalize_inputs = normalize_inputs

    def __call__(
        self,
        inputs: jnp.ndarray,
        mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Point forecast for the horizon set at model construction.

        Args:
            inputs: ``(context, 1)`` for a single series (foundax channel-last
                convention) or ``(B, context, 1)`` for batched inputs.
                ``context`` must be a multiple of ``patch_size`` (32).
            mask: optional, same shape as ``inputs``, ``True`` where input
                is padding to ignore.

        Returns:
            ``(horizon, 1)`` or ``(B, horizon, 1)`` float32 — median point
            forecast. Shape matches the input rank.
        """
        if inputs.ndim == 2:
            out = self._forward_batched(
                inputs[None], None if mask is None else mask[None]
            )
            return out[0]
        if inputs.ndim == 3:
            return self._forward_batched(inputs, mask)
        raise ValueError(
            f"expected inputs of shape (context, 1) or (B, context, 1), got {inputs.shape}"
        )

    def _forward_batched(
        self,
        inputs: jnp.ndarray,
        mask: jnp.ndarray | None,
    ) -> jnp.ndarray:
        from flax import nnx

        B, context, channels = inputs.shape
        assert channels == 1, (
            f"TimesFM is univariate; expected channels=1, got {channels}"
        )
        assert context % self.patch_size == 0, (
            f"context ({context}) must be a multiple of patch_size ({self.patch_size})"
        )
        x = inputs[..., 0]  # (B, context)
        if mask is None:
            mask = jnp.zeros_like(x, dtype=jnp.bool_)
        else:
            mask = mask[..., 0]

        if self.normalize_inputs:
            mu = jnp.mean(x, axis=-1, keepdims=True)
            sigma = jnp.std(x, axis=-1, keepdims=True)
            x = (x - mu) / (sigma + 1e-8)

        model = nnx.merge(self._graphdef, self.state)
        pf, _, ar = model.decode(self.horizon, x, mask)

        # Stitch raw decode output into (B, total) point forecast:
        #   pf: (B, num_input_patches, output_patch_size, num_quantiles+1)
        #   ar: (B, num_decode_steps, output_patch_size, num_quantiles+1) or None
        pieces = [pf[:, -1, :, self.decode_index]]  # last prefill: (B, o)
        if ar is not None:
            ar_point = ar[..., self.decode_index]  # (B, n_steps, o)
            pieces.append(ar_point.reshape(B, -1))
        forecast = jnp.concatenate(pieces, axis=1)[:, : self.horizon]  # (B, horizon)

        if self.normalize_inputs:
            forecast = forecast * (sigma + 1e-8) + mu

        return forecast[..., None]  # (B, horizon, 1) channel-last


# ── variants (poseidon-style naming) ────────────────────────────────────────


def small(
    horizon: int = 64,
    model_id: str = "google/timesfm-2.5-200m-flax",
    normalize_inputs: bool = True,
    **hf_kwargs,
) -> _TimesFM:
    """TimesFM 2.5, 200M params. Currently the only published size.

    Downloads ``google/timesfm-2.5-200m-flax`` from Hugging Face on first
    call (cached afterwards).

    Args:
        horizon: forecast horizon — number of future time steps the model
            will predict. Fixed at construction so JIT can specialise on
            it (build separate instances for different horizons). Need not
            be a multiple of any particular size; internally TimesFM
            produces ``ceil(horizon/128)*128`` predictions and the wrapper
            slices to your requested length. Practical upper bound is
            ~1024 before quality degrades.
        model_id: HF repo id. Override to load a fork or a local checkpoint
            directory (passed straight through to upstream
            ``from_pretrained``).
        normalize_inputs: Whether to revIN-normalise each series before the
            forward pass (and un-normalise the output). Recommended for
            inputs with extreme magnitudes; matches the upstream
            ``forecast()`` default.
        **hf_kwargs: Forwarded to ``huggingface_hub.snapshot_download``
            (``revision``, ``cache_dir``, ``token``, etc.).

    Returns:
        An ``eqx.Module`` wrapping the upstream NNX model. Callable with
        ``model(inputs, mask=None)``; see :class:`_TimesFM`. Supports both
        inference and gradient-based fine-tuning.
    """
    return _TimesFM(
        model_id=model_id,
        horizon=horizon,
        normalize_inputs=normalize_inputs,
        **hf_kwargs,
    )


default = small


__all__ = ["small", "default"]

_callable_module.install(__name__, small)
