# Core Models

This page covers the **direct Equinox architectures** implemented inside `foundax/architectures/` and exposed through `foundax.nn`.

These models are the lighter-weight part of the repository: they are intended for direct experimentation, baseline comparisons, and downstream integration without depending on the vendored foundation-model packages.

## Summary

| Family | Constructors | Typical use |
| --- | --- | --- |
| Linear / MLP | `linear`, `mlp` | Simple regression and coordinate networks |
| Fourier Neural Operators | `fno1d`, `fno2d`, `fno3d` | Structured-grid operator learning |
| UNet | `unet1d`, `unet2d`, `unet3d` | Encoder-decoder baselines on regular grids |
| Generic Transformer | `transformer` | Sequence-to-sequence baselines |
| DeepONet | `deeponet` | Operator learning with branch/trunk factorization |
| CNO | `cno2d` | Continuous neural operator on image-like fields |
| MgNO | `mgno1d`, `mgno2d` | Multigrid-inspired operator learning |
| Geometry-aware operators | `geofno`, `pcno`, `pit`, `pointnet` | Irregular meshes, coordinates, point clouds |
| GNOT family | `cgptno`, `gnot`, `moegptno` | Transformer-based operator learning on irregular domains |

## Linear And MLP

### `linear`

Thin wrapper around a batched linear layer. Useful for simple heads, projections, and small regression models.

### `mlp`

Standard multilayer perceptron with configurable depth, hidden width, activation, normalization, and dropout.

Use it when:

- your input is low-dimensional coordinates or features
- you want a simple baseline before moving to neural operators
- you need a small shared subnetwork inside a larger pipeline

## Fourier Neural Operators

Constructors:

- `fno1d`
- `fno2d`
- `fno3d`

These are spectral neural operators that alternate learned Fourier-domain mixing with local projections. They are the most direct choice in `foundax` for regular-grid PDE surrogate learning.

Implementation notes:

- `fno1d` uses stacked spectral convolution layers for 1D signals
- `fno2d` and `fno3d` extend the same idea to 2D and 3D fields
- the implementations support configurable mode truncation, hidden width, normalization, and repeated spectral blocks

Use them when:

- your inputs live on fixed Cartesian grids
- you want strong operator-learning baselines with moderate implementation complexity
- Fourier mixing is a better fit than a pure convolutional encoder-decoder

## UNet

Constructors:

- `unet1d`
- `unet2d`
- `unet3d`

The UNet family in `foundax` provides standard encoder-decoder architectures with skip connections for structured inputs. These are useful as robust baselines for dense prediction over 1D, 2D, or 3D fields.

Use them when:

- locality matters more than global spectral mixing
- you want an interpretable baseline for image-like or volume-like PDE states
- you need a familiar encoder-decoder architecture that is easy to adapt

## Transformer

### `transformer`

General encoder-decoder transformer factory. This is a generic sequence model rather than a PDE-specific architecture.

Use it when:

- you want a standard attention-based baseline
- your data is already tokenized or sequence-structured
- you need a reusable transformer backbone inside another experimental setup

## DeepONet

### `deeponet`

Implements a flexible Deep Operator Network with configurable branch, trunk, and combination strategies.

Supported branch/trunk choices include MLP-style, residual, convolutional, and transformer-style components. This makes it one of the most configurable operator-learning models in the repository.

Use it when:

- your task is naturally described as evaluating an operator at query coordinates
- you want explicit branch/trunk decomposition
- you need a strong operator-learning baseline that is less tied to a single grid resolution

## CNO

### `cno2d`

Continuous Neural Operator for 2D fields. This model is an alternative to FNO and UNet for image-like PDE data and uses a hierarchical convolutional design.

Use it when:

- you want a convolution-heavy operator model rather than spectral mixing
- you are working with 2D grid data
- you want a stronger learned multiscale image-to-image operator baseline

## MgNO

Constructors:

- `mgno1d`
- `mgno2d`

Multigrid Neural Operator models use restriction, prolongation, and iterative correction ideas inspired by multigrid solvers.

Use them when:

- you want solver-inspired inductive bias
- you care about hierarchical scale interactions
- you want an alternative to FNO on structured grids

## Geometry-Aware Models

### `geofno`

Geometry-aware FNO variant for non-uniform spatial layouts.

### `pcno`

Point-cloud neural operator variant for irregular coordinate sets.

### `pit`

Position-induced transformer-style operator model for coordinate-aware learning.

### `pointnet`

PointNet-style model for unordered point sets.

Use this group when:

- your domain is not a simple fixed Cartesian grid
- point coordinates or geometry carry important information
- mesh or point-cloud structure is central to the task

## GNOT Family

Constructors:

- `cgptno`
- `gnot`
- `moegptno`

These models implement the General Neural Operator Transformer family. They are intended for operator learning on arbitrary geometries and irregular sampling patterns, with transformer-style cross-attention and optional mixture-of-experts routing.

Use them when:

- you need attention-based operator learning on irregular domains
- you have multiple input branches or multiple coupled fields
- you want a more expressive transformer-based architecture than DeepONet or FNO

Reference:

- GNOT paper: https://arxiv.org/abs/2302.14376

## Factory conventions

All core factories are exposed through `foundax.nn` and re-exported at package level:

```python
import foundax as fx

model = fx.fno2d(in_features=1, hidden_channels=32, n_modes=16)
model = fx.deeponet(branch_type="mlp", trunk_type="mlp")
model = fx.gnot(branch_sizes=[64], trunk_size=2)
```

The exact forward signature depends on the model family, so it is best to inspect the constructor in `foundax/nn.py` together with the implementation module in `foundax/architectures/` when integrating a new model.