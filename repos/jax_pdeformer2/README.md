# jax_pdeformer2

> **Note:** This package is designed to be used with [jNO](https://github.com/armbrusl/jNO).

> **Warning:** This is a research-level repository. It may contain bugs and is subject to continuous change without notice.

JAX/Flax translation of **PDEformer-2**, a neural operator for solving partial differential equations, maintaining exact 1-to-1 weight compatibility with the original MindSpore implementation for pretrained checkpoint conversion.

## Overview

PDEformer-2 represents PDEs as directed acyclic graphs (DAGs) and processes them with a Graphormer encoder paired with an implicit neural representation (PolyINR) decoder. This repository provides a pure JAX/Flax reimplementation supporting all three pretrained variants (Small, Base, Fast) and custom configurations.

### Architecture

```
     Scalar coefficients          Function fields (IC, BC, ...)
            │                              │
            ▼                              ▼
  ┌───────────────────┐        ┌───────────────────────────┐
  │  ScalarEncoder    │        │  Conv2dFuncEncoderV3      │
  │  (3-layer MLP)    │        │  4-branch CNN per field   │
  └─────────┬─────────┘        └─────────────┬─────────────┘
            │ (B, S, D)                       │ (B, F×4, D)
            └──────────────┬──────────────────┘
                           │ Concatenate → (B, N, D)
                           ▼
              ┌────────────────────────┐
              │  GraphormerEncoder     │   Graph-aware transformer:
              │  L × EncoderLayer      │   - MultiheadAttention + attn_bias
              │                        │   - GraphNodeFeature (degree + spatial)
              │                        │   - Feed-forward network
              └────────────┬───────────┘
                           │ (B, N, D)
                           ▼
              ┌────────────────────────┐
              │  PolyINR + Hypernet    │   - Hypernet maps encoder output → INR weights
              │                        │   - PolyINR evaluates at query coordinates
              │                        │   - Polynomial modulation with sin activation
              └────────────┬───────────┘
                           │
                           ▼
                Output (B, Q, 1) — solution at query points
```

### Model Variants

| Config | Encoder Layers | Embed Dim | FFN Dim | Heads | INR Hidden | INR Layers | Parameters |
|--------|---------------|-----------|---------|-------|------------|------------|------------|
| Small  | 9             | 512       | 1024    | 32    | 128        | 12         | ~27.7M     |
| Base   | 12            | 768       | 1536    | 32    | 768        | 12         |            |
| Fast   | 12            | 768       | 1536    | 32    | 256        | 12         |            |

All share: `num_node_type=128`, `num_spatial=16`, scalar encoder 3×256, CNN function encoder with 4 branches at 128² resolution.

## Reference

Shi et al., "PDEformer-2: A Foundation Model for Two-Dimensional PDEs" (2025)
- Paper: <https://arxiv.org/abs/2502.14844>
- Code: <https://github.com/functoreality/pdeformer-2>

## Installation

```bash
# Using uv (recommended)
uv venv
uv pip install -e .

# For development/testing
uv pip install -e ".[dev]"

# For MindSpore checkpoint conversion
uv pip install -e ".[convert]"
```

## Usage

### Loading a Pretrained Model

```python
from jax_pdeformer2 import create_pdeformer_from_config, PDEFORMER_SMALL_CONFIG
from jax_pdeformer2.utils import load_pdeformer_weights

# Load model with pretrained weights (from converted .npz file)
model, params = load_pdeformer_weights("pdeformer2-small.npz")

# Or from msgpack format
from jax_pdeformer2.utils import load_pdeformer_from_msgpack
model, params = load_pdeformer_from_msgpack("pdeformer2-small.msgpack")

# Run inference
output = model.apply(params,
    node_type=node_type,
    node_scalar=node_scalar,
    node_function=node_function,
    in_degree=in_degree,
    out_degree=out_degree,
    attn_bias=attn_bias,
    spatial_pos=spatial_pos,
    coordinate=coordinate,
)
```

### Creating a Model from Scratch

```python
from jax_pdeformer2 import create_pdeformer_from_config, PDEFORMER_SMALL_CONFIG
import jax

model = create_pdeformer_from_config({"model": PDEFORMER_SMALL_CONFIG})

key = jax.random.PRNGKey(42)
params = model.init(key, **dummy_inputs)
output = model.apply(params, **inputs)
```

### Converting MindSpore Checkpoints

Two-stage conversion: MindSpore `.ckpt` → `.npz` → Flax `.msgpack`:

```bash
# Stage 1: MindSpore ckpt → numpy npz (requires MindSpore)
python -m jax_pdeformer2.convert_checkpoint input.ckpt output.npz

# Stage 2: npz → msgpack (JAX only)
python scripts/convert.py
```

## Input Format

| Input | Shape | Description |
|-------|-------|-------------|
| `node_type` | `(B, N, 1)` | Node type IDs |
| `node_scalar` | `(B, S, 1)` | Scalar coefficient values |
| `node_function` | `(B, F, P, 5)` | Function values `(t, x, y, z, value)` |
| `in_degree` | `(B, N)` | Node in-degrees |
| `out_degree` | `(B, N)` | Node out-degrees |
| `attn_bias` | `(B, N, N)` | Graph attention bias |
| `spatial_pos` | `(B, N, N)` | Shortest path distances |
| `coordinate` | `(B, Q, 4)` | Query points `(t, x, y, z)` |

Where `B` = batch, `N` = total nodes (`S + F × num_branches`), `S` = scalar nodes, `F` = function nodes, `P` = points per function (128²), `Q` = query points.

## Equivalence Testing

```bash
# Requires MindSpore installation + original PDEformer-2 repo
python scripts/compare.py
```

The comparison script tests advection-Burgers and heat equation samples, reporting per-sample L2 differences.

## Project Structure

```
jax_pdeformer2/
├── jax_pdeformer2/             # Core library
│   ├── __init__.py             # Public API + version
│   ├── pdeformer.py            # PDEformer, PDEEncoder, model configs
│   ├── graphormer.py           # GraphormerEncoder, attention, graph features
│   ├── inr_with_hypernet.py    # PolyINR + hypernet decoder
│   ├── function_encoder.py     # Conv2dFuncEncoderV3 (CNN branch encoder)
│   ├── basic_block.py          # MLP, Sine, Scale, Clamp primitives
│   ├── utils.py                # Weight loading + dummy input creation
│   └── convert_checkpoint.py   # MindSpore → npz conversion
├── scripts/
│   ├── convert.py              # npz → msgpack batch conversion
│   └── compare.py              # MindSpore vs JAX equivalence tests
├── tests/
│   ├── test_components.py      # Unit tests for individual modules
│   ├── test_equivalence.py     # Numerical equivalence tests
│   ├── test_all_checkpoints.py # Test all 3 converted checkpoints
│   └── inspect_fast_checkpoint.py
├── pyproject.toml
├── LICENSE                     # Apache 2.0
├── .gitignore
└── README.md
```

## Module Details

### PDEformer (`pdeformer.py`)

- **`PDEEncoder`** — Combines scalar encoder, function encoder, and Graphormer into a single encoder pipeline
- **`PDEformer`** — Full model: PDEEncoder + PolyINR hypernet decoder
- **`create_pdeformer_from_config()`** — Factory function building a PDEformer from a nested config dict
- **`PDEFORMER_SMALL_CONFIG`** / **`PDEFORMER_BASE_CONFIG`** / **`PDEFORMER_FAST_CONFIG`** — Pretrained model configurations

### Graphormer (`graphormer.py`)

Graph-structure-aware transformer encoder:

- **`GraphormerEncoder`** — Stack of `GraphormerEncoderLayer` with graph-specific input embeddings
- **`GraphormerEncoderLayer`** — Self-attention + FFN with pre-LayerNorm
- **`MultiheadAttention`** — Standard multi-head attention with additive attention bias for graph structure
- **`GraphNodeFeature`** — Combines node type embedding + in/out degree embeddings
- **`GraphAttnBias`** — Computes attention bias from spatial (shortest-path) distances

### PolyINR with Hypernet (`inr_with_hypernet.py`)

Implicit neural representation decoder:

- **`PolyINR`** — Polynomial-modulated INR with sinusoidal activation, optional affine/shift/scale modulation per layer
- **`PolyINRWithHypernet`** — Wraps a PolyINR whose weights are generated by a hypernetwork from the encoder output
- **`get_inr_with_hypernet()`** — Factory function from config

### Function Encoder (`function_encoder.py`)

- **`Conv2dFuncEncoderV3`** — Multi-branch 2D CNN that encodes spatiotemporal function fields (IC, BC, forcing) into fixed-dimensional tokens
- **`get_function_encoder()`** — Factory function from config

### Basic Blocks (`basic_block.py`)

- **`MLP`** — Standard feedforward network with configurable layers, activations, and dropout
- **`Sine`** — `sin(ω₀ · x)` activation (SIREN-style)
- **`Scale`** — Learnable scalar multiplier
- **`Clamp`** — Gradient-safe value clamping

## Implementation Notes

### Key Differences from MindSpore

1. **Parameter naming**: MindSpore's `Dense` → Flax `Dense` with `.kernel` / `.bias` convention; weight shapes are transposed during conversion.

2. **Convolutions**: MindSpore uses NCHW by default; the JAX version uses NHWC throughout with `cnn_keep_nchw=True` in config handled via appropriate transposes during weight conversion.

3. **Graph bias broadcasting**: Attention bias computation matches MindSpore's `(B, H, N, N)` layout via explicit reshape rather than implicit broadcasting.

4. **PolyINR modulation**: The polynomial modulation (`shift`, `scale`, `affine`) is applied identically — the hypernet generates per-layer modulation parameters from the encoder's node embeddings.

5. **Numerical precision**: Verified max absolute difference < 0.03% against MindSpore on advection–Burgers and heat equation test cases across all three model sizes.

## License

Apache 2.0 — see [LICENSE](LICENSE).
