# jax_dpot

JAX/Flax translation of DPOT (2D) with checkpoint conversion and PyTorch equivalence checks.

## Status

Phase 1 currently targets:
- DPOTNet (2D) architecture
- Ti checkpoint conversion
- PyTorch-vs-JAX compare script

## Quick start

```bash
uv venv
uv pip install -e .
uv pip install -e ".[convert]"
```

### Run smoke test

```bash
uv run pytest -q tests/test_model.py
```

### Convert Ti weights

```bash
uv run python scripts/convert.py \
  --input ogrepo/model_Ti.pth \
  --output model_Ti.msgpack
```

### Compare against PyTorch

```bash
export DPOT_ROOT=/path/to/jax_dpot/ogrepo/DPOT
uv run python scripts/compare.py \
  --checkpoint ogrepo/model_Ti.pth \
  --threshold 1e-4
```
