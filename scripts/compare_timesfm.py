#!/usr/bin/env python3
"""
TimesFM: forward-pass parity + training-readiness checks for the
foundax Flax wrapper.

Four checks (all pure JAX, no PyTorch dep):

  1. **Upstream-vs-foundax-wrapper parity** — call the upstream high-level
     ``TimesFM_2p5_200M_flax.forecast()`` and compare against our
     wrapper's ``model(inputs)`` on the same input.
  2. **Unbatched vs batched shape equivalence** — single-series
     ``(context, 1)`` call vs stacking + batched ``(B, context, 1)`` call.
  3. **JIT compatibility** — eager call equals ``eqx.filter_jit(model)``.
  4. **Training readiness** — gradients flow through ``eqx.filter_grad``,
     and one optimizer step decreases a synthetic loss.

Checks 1-4 need the HF checkpoint (~800MB). On first run, pass
``--download`` to allow the fetch. After that the cache makes subsequent
runs fast.

Usage::

    python scripts/compare_timesfm.py            # structural-only
    python scripts/compare_timesfm.py --download # full parity + jit + train
"""

from __future__ import annotations

import argparse
import os
from importlib.util import find_spec
from pathlib import Path

import numpy as np


def _checkpoint_exists_locally(model_id: str) -> bool:
    try:
        hf_home = Path(
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub")
        )
        org, name = model_id.split("/")
        candidates = list(
            (hf_home / "hub").glob(f"models--{org}--{name}/snapshots/*")
        ) + list(hf_home.glob(f"models--{org}--{name}/snapshots/*"))
        return any(c.is_dir() and any(c.iterdir()) for c in candidates)
    except Exception:
        return False


def _generate_inputs(seed: int, batch: int = 3, context: int = 512):
    """Return ``(B, context, 1)`` jnp array — context % patch_size=32 == 0.

    Foundax channel-last convention: 1-D univariate series have channels=1.
    """
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 50.0, context)
    series = np.stack(
        [
            np.sin(t) + 0.3 * np.cos(2.3 * t) + 0.05 * rng.standard_normal(t.shape),
            0.5 * t / 50.0 + 0.4 * np.sin(0.7 * t) + 0.1 * rng.standard_normal(t.shape),
            np.cumsum(rng.standard_normal(t.shape)) * 0.05 + 0.3 * np.sin(0.5 * t),
        ],
        axis=0,
    )[:batch]
    return jnp.asarray(series, dtype=jnp.float32)[..., None]  # (B, context, 1)


def _compare(name, ref, ours, atol=1e-4, rtol=1e-4):
    ref = np.asarray(ref)
    ours = np.asarray(ours)
    assert ref.shape == ours.shape, f"{name}: shape {ref.shape} vs {ours.shape}"
    diff = np.abs(ref - ours)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    rel = float(np.linalg.norm(diff) / (np.linalg.norm(ref) + 1e-12))
    threshold = atol + rtol * float(np.max(np.abs(ref)))
    status = "PASS" if max_abs < threshold else "FAIL"
    print(
        f"  [{status}] {name:<40} shape={tuple(ours.shape)} "
        f"max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} rel_l2={rel:.3e}"
    )
    return status == "PASS"


def upstream_vs_wrapper(seed: int) -> bool:
    """Foundax wrapper (batched form) == upstream forecast() output."""
    import foundax as fx
    from timesfm import ForecastConfig, TimesFM_2p5_200M_flax

    horizon, context = 64, 512

    print("Loading upstream Flax model + compiling (~30s)...")
    upstream = TimesFM_2p5_200M_flax.from_pretrained("google/timesfm-2.5-200m-flax")
    upstream.compile(
        forecast_config=ForecastConfig(
            max_context=context,
            max_horizon=horizon,
            normalize_inputs=False,
            force_flip_invariance=False,
            infer_is_positive=False,
        ),
        dryrun=False,
    )

    print("Loading foundax wrapper...")
    ours = fx.timesfm.small(horizon=horizon, normalize_inputs=False)

    batch = _generate_inputs(seed, batch=3, context=context)  # (3, context, 1)
    inputs_list = [np.asarray(batch[i, :, 0]) for i in range(batch.shape[0])]

    upstream_point, _ = upstream.forecast(horizon=horizon, inputs=inputs_list)
    ours_point = ours(batch)[..., 0]  # strip channel for comparison

    return _compare(
        "upstream forecast vs wrapper (batched)", upstream_point, ours_point
    )


def single_vs_batched(seed: int) -> bool:
    """Single-series (context, 1) call == stacking + batched call."""
    import jax.numpy as jnp
    import foundax as fx

    horizon, context = 64, 512
    model = fx.timesfm.small(horizon=horizon, normalize_inputs=True)
    batch = _generate_inputs(seed, batch=3, context=context)

    print("Per-series (channel-last unbatched) loop...")
    single_outs = jnp.stack([model(batch[i]) for i in range(batch.shape[0])])
    print("Batched (B, context, 1) call...")
    batched_out = model(batch)

    return _compare("single (context,1) vs batched", batched_out, single_outs)


def jit_equivalence(seed: int) -> bool:
    """Eager == eqx.filter_jit on the unbatched (context, 1) form."""
    import equinox as eqx
    import foundax as fx

    horizon, context = 64, 512
    model = fx.timesfm.small(horizon=horizon, normalize_inputs=True)
    series = _generate_inputs(seed, batch=1, context=context)[0]

    print("Running eager forward...")
    eager = model(series)
    print("Running jit'd forward...")
    jitted = eqx.filter_jit(model)(series)

    return _compare("eager vs eqx.filter_jit", eager, jitted, atol=1e-5, rtol=1e-5)


def train_step_decreases_loss(seed: int) -> bool:
    """Fine-tuning loop: ``eqx.filter_grad`` finds trainable arrays + one
    optax step strictly decreases an MSE loss on synthetic data."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax
    import foundax as fx

    horizon, context = 32, 128  # small for CPU speed
    model = fx.timesfm.small(horizon=horizon, normalize_inputs=True)

    # Synthetic supervised data: model trained to predict a step-shifted sine.
    rng = jax.random.PRNGKey(seed)
    x_rng, y_rng = jax.random.split(rng)
    x = (
        jnp.sin(jnp.linspace(0, 20, context))[None, :, None]
        + 0.05 * jax.random.normal(x_rng, (2, context, 1))
    ).astype(jnp.float32)
    y = (
        jnp.sin(jnp.linspace(20, 20 + 10 * horizon / context, horizon))[None, :, None]
        + 0.05 * jax.random.normal(y_rng, (2, horizon, 1))
    ).astype(jnp.float32)

    def loss_fn(m):
        return jnp.mean((m(x) - y) ** 2)

    optimizer = optax.adamw(learning_rate=1e-5)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        return eqx.apply_updates(model, updates), opt_state, loss

    print("Running 3 fine-tuning steps...")
    losses = []
    for _ in range(3):
        model, opt_state, loss = step(model, opt_state)
        losses.append(float(loss))
    print(f"  losses: {losses[0]:.6f} → {losses[-1]:.6f}")
    ok = losses[-1] < losses[0]
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] training step decreases loss     "
        f"loss[0]={losses[0]:.3e}  loss[-1]={losses[-1]:.3e}  "
        f"Δ={losses[0] - losses[-1]:+.3e}"
    )
    return ok


def run_structural_check() -> int:
    import foundax as fx

    print("[TimesFM] Structural check (no checkpoint download)")
    print(f"  fx.timesfm is callable:      {callable(fx.timesfm)}")
    print(f"  fx.timesfm.small exists:     {hasattr(fx.timesfm, 'small')}")
    try:
        from timesfm import TimesFM_2p5_200M_flax  # noqa: F401

        print("  upstream timesfm class imports: ok")
    except Exception as e:
        print(f"  upstream import failed: {e}")
        return 1
    print("  PASS")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--download",
        action="store_true",
        help="Allow downloading the HF checkpoint (~800MB) for full forward parity.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if find_spec("timesfm") is None:
        print("timesfm package not installed.")
        return 1

    flax_cached = _checkpoint_exists_locally("google/timesfm-2.5-200m-flax")
    if not (args.download or flax_cached):
        print("=" * 70)
        print("TimesFM: structural check only (run with --download for full parity)")
        print("=" * 70)
        return run_structural_check()

    print("=" * 70)
    print("TimesFM: foundax wrapper forward + JIT parity")
    print(f"  seed: {args.seed}")
    print("=" * 70)
    ok = [
        upstream_vs_wrapper(args.seed),
        single_vs_batched(args.seed),
        jit_equivalence(args.seed),
        train_step_decreases_loss(args.seed),
    ]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
