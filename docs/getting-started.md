# Getting Started

`foundax` exposes a single package with two main kinds of models:

- **Core Equinox architectures** implemented directly in `foundax/architectures/`
- **Equinox wrappers** around vendored model implementations in `repos/`

This documentation focuses on the **Equinox-facing API** and intentionally does not document the Flax-specific foundation-model entry points.

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

The source install includes the vendored JAX packages listed in `pyproject.toml`, so the foundation-model wrappers can resolve their implementation modules without extra manual path setup.

## Package layout

- `foundax.nn`: top-level factory functions for the main model families
- `foundax.poseidon`, `foundax.morph`, `foundax.mpp`, `foundax.walrus`, `foundax.bcat`, `foundax.pdeformer2`, `foundax.dpot`, `foundax.prose`: namespace-style access to foundation-model wrappers
- `foundax/architectures/`: direct Equinox implementations of the non-foundation models
- `repos/`: vendored upstream JAX reimplementations used by the foundation-model wrappers

## Recommended usage style

For the foundation-model families, prefer the namespace-style calls:

```python
import foundax as fx

poseidon = fx.poseidon.T()
morph = fx.morph.S()
mpp = fx.mpp.B(n_states=12)
walrus = fx.walrus.base()
```

The package also exposes top-level convenience aliases such as `fx.poseidonT()` and `fx.morph_Ti()`, but the namespace form is easier to read and maps more directly onto the repository structure.

For the core architectures, use the factory functions in `foundax.nn`:

```python
import foundax as fx

mlp = fx.mlp(in_features=2, output_dim=1, hidden_dims=64, num_layers=3)
fno = fx.fno2d(in_features=1, hidden_channels=32, n_modes=16)
unet = fx.unet2d(in_channels=1, out_channels=1)
deeponet = fx.deeponet(branch_type="mlp", trunk_type="mlp")
```

## What each factory returns

- Most entries documented here return an `equinox.Module`
- The intended workflow is direct JAX/Equinox use, or wrapping inside downstream code such as `jNO`

Typical usage is:

```python
import jax
import jax.numpy as jnp
import foundax as fx

model = fx.fno2d(in_features=1, hidden_channels=32, n_modes=16)
x = jnp.ones((64, 64, 1))
y = model(x)
```

The exact call signature depends on the model family. Simpler models such as MLPs or FNOs accept direct tensors. Some foundation-model wrappers expect structured inputs such as graph tensors, symbolic tokens, or boundary-condition metadata.

## Model groups

### Core models

These are local Equinox implementations intended as reusable building blocks and operator-learning baselines:

- `linear`, `mlp`
- `fno1d`, `fno2d`, `fno3d`
- `unet1d`, `unet2d`, `unet3d`
- `transformer`, `deeponet`, `cno2d`
- `mgno1d`, `mgno2d`, `geofno`, `pcno`
- `cgptno`, `gnot`, `moegptno`, `pit`, `pointnet`

### Foundation-model wrappers

These expose larger pretrained or pretrained-style model families through Equinox constructors:

- `poseidon`
- `morph`
- `mpp`
- `walrus`
- `bcat`
- `pdeformer2`
- `dpot`
- `prose`

## Choosing a model

- Use **FNO** or **UNet** when you want a compact baseline for grid-based PDE surrogates.
- Use **DeepONet** when your problem is naturally expressed as branch/trunk operator evaluation.
- Use **GeoFNO**, **PiT**, or **PointNet** when geometry or point-cloud structure matters.
- Use **GNOT** variants when you need transformer-style operator learning on irregular domains or multiple inputs.
- Use the **foundation-model wrappers** when you want to reproduce or adapt a specific large published architecture.

The next pages split these two groups into more detailed catalogs.