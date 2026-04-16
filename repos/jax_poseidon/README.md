# jax_poseidon

> **Note:** This package is designed to be used with [jNO](https://github.com/FhG-IISB/jNO).

> **Warning:** This is a research-level repository. It may contain bugs and is subject to continuous change without notice.

JAX/Flax translation of the **Poseidon** PDE foundation model, maintaining exact 1-to-1 weight compatibility with the original PyTorch implementation for pretrained checkpoint conversion.

## Overview

Poseidon is a family of efficient foundation models for partial differential equations (PDEs) based on a Swin Transformer V2 encoder-decoder architecture (ScOT). This repository provides a pure JAX/Flax reimplementation that supports the three pretrained variants (T, B, L) and custom configurations.

### Architecture

```
Input (B, H, W, C_in)
        │
        ▼
┌───────────────────────────┐
│  ScOTPatchEmbeddings      │  Conv2D patch projection + optional absolute pos embedding
│  + ScOTEmbeddings         │  Input: (B,H,W,C) → Patches: (B, H/p, W/p, embed_dim)
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│  Encoder (4 stages)       │  Each stage:
│  ScOTEncodeStage ×4       │    N × ScOTLayer (shifted window attention + MLP)
│                           │    + ScOTPatchMerging (2× spatial downsample)
└───────────┬───────────────┘
            │  Skip connections
            ▼
┌───────────────────────────┐
│  Decoder (4 stages)       │  Each stage:
│  ScOTDecodeStage ×4       │    ScOTPatchUnmerging (2× spatial upsample)
│                           │    + N × ScOTLayer + skip connection fusion
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│  ScOTPatchRecovery        │  Transpose conv to recover original resolution
│  + ResidualBlockWrapper   │  Optional ConvNeXt/ResNet residual blocks
└───────────┬───────────────┘
            │
            ▼
Output (B, H, W, C_out)
```

### Model Variants

| Variant | Embed Dim | Depths | Heads | Window | Image Size | Channels | Parameters |
|---------|-----------|--------|-------|--------|------------|----------|------------|
| T (Tiny)  | 48  | (4,4,4,4) | (3,6,12,24) | 16 | 128×128 | 4 in / 4 out | ~20.8M  |
| B (Base)  | 96  | (8,8,8,8) | (3,6,12,24) | 16 | 128×128 | 4 in / 4 out | ~157.7M |
| L (Large) | 192 | (8,8,8,8) | (3,6,12,24) | 16 | 128×128 | 4 in / 4 out | ~628.6M |

All use `patch_size=4`, `mlp_ratio=4.0`, `residual_model="convnext"`, `use_conditioning=True`.

## Reference

Herde et al., "Poseidon: Efficient Foundation Models for PDEs" (2024)
- Paper: <https://arxiv.org/abs/2405.19101>
- Weights: Available via the jax_poseidon loader functions (`.msgpack` format)
- Code: <https://github.com/camlab-ethz/poseidon>

## Installation

```bash
# Using uv (recommended)
uv venv
uv pip install -e .

# For GPU support (CUDA 12)
uv pip install -e ".[gpu]"

# For weight conversion from PyTorch
uv pip install -e ".[convert]"
```

## Usage

### Quick Start with Pretrained Weights

```python
import jax
from jax_poseidon import poseidonT, poseidonB, poseidonL

rng = jax.random.PRNGKey(42)

# Load a pretrained model (downloads weights from msgpack)
model, params = poseidonT(rng, "./poseidonT.msgpack")
model, params = poseidonB(rng, "./poseidonB.msgpack")
model, params = poseidonL(rng, "./poseidonL.msgpack")
```

### Separate Initialisation and Weight Loading

```python
from jax_poseidon import poseidonT, init_poseidon_with_weights

# Create model without weights (e.g. for training from scratch)
model = poseidonT()

# Load weights later
rng = jax.random.PRNGKey(0)
model, params = init_poseidon_with_weights(model, rng, "./poseidonT.msgpack")
```

### Custom Configuration with Partial Weights

```python
from jax_poseidon import ScOT, ScOTConfig, merge_pretrained_params
import jax
import jax.numpy as jnp
from flax.serialization import from_bytes

config = ScOTConfig(
    name="poseidonT",
    image_size=128,
    patch_size=4,
    num_channels=1,           # Changed from 4
    num_out_channels=1,       # Changed from 4
    embed_dim=48,
    depths=(4, 4, 4, 4),
    num_heads=(3, 6, 12, 24),
    skip_connections=(2, 2, 2, 0),
    window_size=16,
    mlp_ratio=4.0,
    qkv_bias=True,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    drop_path_rate=0.0,
    hidden_act="gelu",
    use_absolute_embeddings=False,
    initializer_range=0.02,
    layer_norm_eps=1e-5,
    p=1,
    channel_slice_list_normalized_loss=[0, 1, 3, 4],
    residual_model="convnext",
    use_conditioning=True,
    learn_residual=False,
    pretrained_window_sizes=(0, 0, 0, 0),
)

model = ScOT(config, False, False)

rng = jax.random.PRNGKey(42)
dummy = jnp.ones((1, 128, 128, 1))
dummy_time = jnp.zeros((1,))

fresh_params = model.init(
    {"params": rng, "dropout": rng},
    pixel_values=dummy,
    time=dummy_time,
    deterministic=False,
)

# Load pretrained weights and merge (overlapping params copied, rest kept fresh)
with open("./poseidonT.msgpack", "rb") as f:
    pretrained_bytes = f.read()

layer_params = from_bytes(fresh_params, pretrained_bytes)
params = merge_pretrained_params(layer_params, fresh_params, verbose=True)
```

### Running Inference

```python
import jax.numpy as jnp

# x: (B, H, W, C) input field
# time: (B,) conditioning timestep
output = model.apply(
    params,
    pixel_values=x,
    time=time,
    deterministic=True,
    rngs={"dropout": jax.random.PRNGKey(0)},
)
prediction = output.output  # (B, H, W, C_out)
```

## Weight Conversion

Convert PyTorch Poseidon weights to JAX msgpack format:

```bash
python scripts/convert.py --input poseidon_t.pth --output poseidonT.msgpack --model T
```

### Weight Mapping Rules

| PyTorch | Flax | Transformation |
|---------|------|----------------|
| `nn.Linear.weight` (O,I) | `.kernel` (I,O) | Transpose |
| `nn.Linear.bias` | `.bias` | As-is |
| `nn.LayerNorm.weight` | `.scale` | As-is |
| `nn.Conv2d.weight` (O,I,H,W) | `.kernel` (H,W,I,O) | Transpose (2,3,1,0) |
| `nn.ConvTranspose2d.weight` | `.kernel` | Transpose (2,3,1,0) |
| `layers.{i}.blocks.{j}.*` | `layer_{i}/block_{j}/*` | ModuleList → named params |
| `relative_position_bias_table` | `.relative_position_bias_table` | As-is |
| `logit_scale` | `.logit_scale` | As-is |
| `residual_blocks.{i}.*` | `residual_block_{i}/*` | ModuleList → named params |

## Equivalence Testing

```bash
# Requires cloned Poseidon repo
export POSEIDON_ROOT=/path/to/poseidon
python scripts/compare.py
```

The comparison script generates smooth sinusoidal inputs, runs both models, and reports per-sample L2 differences.

## Project Structure

```
jax_poseidon/
├── jax_poseidon/           # Core library
│   ├── __init__.py         # Public API exports
│   ├── scot.py             # Full ScOT architecture (24 classes)
│   └── load.py             # Pretrained model loaders + weight utilities
├── scripts/
│   ├── convert.py          # CLI: PyTorch → msgpack weight conversion
│   └── compare.py          # CLI: PyTorch vs JAX equivalence test
├── pyproject.toml
├── LICENSE
├── .gitignore
└── README.md
```

## Module Details

### ScOT Architecture (`scot.py`)

The full Swin Transformer V2 encoder-decoder in a single file (~1800 lines):

- **`ScOTConfig`** — Dataclass holding all model hyperparameters
- **`ScOTPatchEmbeddings`** — Conv2D patch projection with LayerNorm
- **`ScOTEmbeddings`** — Combines patch embedding + optional absolute position embedding
- **`Swinv2RelativePositionBias`** — Log-CPB relative position bias (continuous)
- **`Swinv2Attention`** — Windowed self-attention with cosine similarity and learned scale
- **`ScOTLayer`** — Single Swin V2 block (attention + MLP + DropPath + optional shift)
- **`ScOTEncodeStage`** — N layers + patch merging (spatial downsample)
- **`ScOTDecodeStage`** — Patch unmerging (spatial upsample) + N layers + skip connections
- **`ScOTEncoder`** / **`ScOTDecoder`** — Stacks of encode/decode stages
- **`ScOTPatchRecovery`** — Transpose conv to recover original resolution
- **`ConvNeXtBlock`** / **`ResNetBlock`** — Residual blocks for output refinement
- **`ScOT`** — Top-level model combining all components

### Model Loading (`load.py`)

- **`poseidonT()`** / **`poseidonB()`** / **`poseidonL()`** — Convenience constructors for pretrained variants
- **`scot()`** — Generic constructor from `ScOTConfig`
- **`init_poseidon_with_weights()`** — Initialise model + load weights from msgpack
- **`merge_pretrained_params()`** — Merge pretrained weights into a fresh param tree (handles shape mismatches for transfer learning)

## Implementation Notes

### Key Differences from PyTorch

1. **Window partitioning**: Uses `jnp.reshape` + `jnp.transpose` instead of PyTorch's `view` + `permute`, matching the same memory layout.

2. **Shifted window attention**: Implements cyclic shift via `jnp.roll` with masking, identical to the PyTorch Swin V2 approach.

3. **DropPath**: Stochastic depth implemented using `jax.random` for per-sample path dropping during training.

4. **Relative position bias**: Log-CPB (continuous position bias) using a small MLP on relative coordinates, matching the Swin V2 paper exactly.

5. **Weight merging**: `merge_pretrained_params` performs shape-aware merging — only copies weights where shapes match, allowing architecture modifications while preserving pretrained features.

## License

MIT — see [LICENSE](LICENSE).
