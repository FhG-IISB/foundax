# Getting Started

`foundax` exposes a unified Equinox-facing package with two model groups:

- core architectures implemented directly in `foundax/architectures`
- foundation-model wrappers backed by vendored repositories in `repos`

## Installation

### From PyPI

```bash
uv pip install foundax
```

### From source

```bash
git clone https://github.com/FhG-IISB/foundax.git
cd foundax
uv pip install -e .
```

The source installation includes the vendored JAX packages referenced in `pyproject.toml`, so wrapper modules resolve without extra path setup.

## First Import

```python
import foundax as fx

# Core model
core = fx.fno2d(in_features=1, hidden_channels=32, n_modes=16)

# Foundation wrapper
fm = fx.poseidon.T()
```

## Recommended API Style

Use namespace style for foundation wrappers:

```python
import foundax as fx

poseidon = fx.poseidon.T()
morph = fx.morph.S()
mpp = fx.mpp.B(n_states=12)
walrus = fx.walrus.base()
```

Top-level aliases such as `fx.poseidonT()` remain available, but namespace calls are preferred because they map directly to the repository structure.

## Package Layout

| Path / Module | Purpose |
|---|---|
| `foundax.nn` | Factory functions for core and convenience wrapper aliases |
| `foundax.poseidon`, `foundax.morph`, `foundax.mpp`, `foundax.walrus`, `foundax.bcat`, `foundax.pdeformer2`, `foundax.dpot`, `foundax.prose` | Namespace wrapper entry points |
| `foundax/architectures` | Direct Equinox implementations of core model families |
| `repos` | Vendored upstream JAX reimplementations used by wrappers |

## What Factories Return

Most constructors documented here return an `equinox.Module` for direct JAX/Equinox use.

```python
import jax.numpy as jnp
import foundax as fx

model = fx.mlp(in_features=2, output_dim=1, hidden_dims=64, num_layers=3)
x = jnp.ones((16, 2))
y = model(x)
```

Exact forward signatures vary by family. Core models usually consume dense tensors, while some foundation wrappers require structured inputs (for example graph metadata or boundary-condition descriptors).

## Choosing A Model

- Use FNO or UNet for compact structured-grid operator baselines.
- Use DeepONet for explicit branch/trunk operator evaluation.
- Use GeoFNO, PiT, PCNO, or PointNet when geometry or point-cloud structure matters.
- Use GNOT variants for transformer-style operator learning on irregular domains.
- Use foundation wrappers when reproducing or adapting large published architectures.

## Next Steps

| Topic | Page |
|---|---|
| Direct Equinox architecture families | [Core Models](core-models.md) |
| Wrapper families, variants, and references | [Foundation Models](equinox-architectures.md) |
| Minimal runnable constructor snippets | [Model Examples](model-examples.md) |
| Docs build and deployment | [GitHub Pages](github-pages.md) |