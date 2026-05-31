# Core Models

This page covers the **direct Equinox architectures** implemented inside `foundax/architectures/` and exposed through `foundax.nn`.

These models are the lighter-weight part of the repository: they are intended for direct experimentation, baseline comparisons, and downstream integration without depending on the vendored foundation-model packages.

## Summary

For the full list with paper references — including the foundation-model wrappers — see [Architectures Overview](architectures.md).

| Family | Constructors | Reference | Typical use |
| --- | --- | --- | --- |
| Linear / MLP | `linear`, `mlp` | — | Simple regression and coordinate networks |
| Fourier Neural Operators | `fno1d`, `fno2d`, `fno3d` | Li et al. 2020 — [arXiv:2010.08895](https://arxiv.org/abs/2010.08895) | Structured-grid operator learning |
| UNet | `unet1d`, `unet2d`, `unet3d` | Ronneberger et al. 2015 — [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) | Encoder-decoder baselines on regular grids |
| Generic Transformer | `transformer` | Vaswani et al. 2017 — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | Sequence-to-sequence baselines |
| DeepONet | `deeponet` | Lu et al. 2019 — [arXiv:1910.03193](https://arxiv.org/abs/1910.03193) | Operator learning with branch/trunk factorization |
| CNO | `cno2d` | Raonić et al. 2023 — [arXiv:2302.01178](https://arxiv.org/abs/2302.01178) | Continuous neural operator on image-like fields |
| MgNO | `mgno1d`, `mgno2d` | He et al. 2023 — [arXiv:2310.19809](https://arxiv.org/abs/2310.19809) | Multigrid-inspired operator learning |
| Geometry-aware operators | `geofno`, `pcno`, `pit`, `pointnet` | GeoFNO: [arXiv:2207.05209](https://arxiv.org/abs/2207.05209); PiT: [arXiv:2405.09285](https://arxiv.org/abs/2405.09285); PointNet: [arXiv:1612.00593](https://arxiv.org/abs/1612.00593); PCNO: [github](https://github.com/PKU-CMEGroup/NeuralOperator) | Irregular meshes, coordinates, point clouds |
| GNOT family | `cgptno`, `gnot`, `moegptno` | Hao et al., ICML 2023 — [arXiv:2302.14376](https://arxiv.org/abs/2302.14376) | Transformer-based operator learning on irregular domains |
| Diffusion / flow backbones | `dit2d/3d`, `ffno2d/3d`, `wno1d/2d/3d` | DiT: [arXiv:2212.09748](https://arxiv.org/abs/2212.09748); F-FNO: [arXiv:2111.13802](https://arxiv.org/abs/2111.13802); WNO: [arXiv:2205.02191](https://arxiv.org/abs/2205.02191) | Time-conditioned backbones for flow-matching and diffusion training |
| Kolmogorov–Arnold Networks | `kan`, `fastkan`, `chebyshev_kan`, `fourier_kan`, … (17 variants) | Liu et al. 2024 — [arXiv:2404.19756](https://arxiv.org/abs/2404.19756) (and others) | MLP alternative with learnable univariate edges — see the [KAN page](kan.md) |

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

## Diffusion And Flow-Matching Backbones

These models are channel-last, time-conditioned backbones designed to drop into a diffusion or flow-matching training loop. They share the foundax pipe API and the standard `SinusoidalTimeEmbedding` / `FiLMLayer` / `AdaLayerNorm` / `AdaLayerNormZero` primitives exposed at package level.

### `dit2d`, `dit3d` — Diffusion Transformer

ViT-style backbone with patch embedding and fixed sin/cos positional encoding. Time conditioning is injected via `AdaLayerNormZero`, following the original DiT formulation.

- Paper: Peebles & Xie, *Scalable Diffusion Models with Transformers* — [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)
- PyTorch reference: [facebookresearch/DiT](https://github.com/facebookresearch/DiT)

Use them when you want a transformer backbone for image-like or volumetric flow-matching targets at moderate resolution.

### `ffno2d`, `ffno3d` — Factorized Fourier Neural Operator

Replaces the O(m^d · C²) full d-dimensional spectral convolution with `d` independent 1-D spectral convolutions summed together. Reduces cost to O(d · m · C²) while preserving most of the expressivity of FNO.

- Paper: Tran et al., *Factorized Fourier Neural Operators* — [arXiv:2111.13802](https://arxiv.org/abs/2111.13802)
- PyTorch reference: [alasdairtran/fourierflow](https://github.com/alasdairtran/fourierflow)

Use them when you want FNO's spectral mixing at higher channel counts or higher dimensions, where the full d-D variant becomes prohibitive.

### `wno1d`, `wno2d`, `wno3d` — Wavelet Neural Operator

Multi-scale discrete wavelet transform with a hardcoded Daubechies-8 (16-tap) low-pass filter; the high-pass is derived via the quadrature mirror. Learned linear mixing in the wavelet domain; `jax.image.resize` restores spatial resolution. Spatial dimensions must be divisible by `2^n_scales`.

- Paper: Tripura & Chakraborty, *Wavelet Neural Operator for solving parametric PDEs* — [arXiv:2205.02191](https://arxiv.org/abs/2205.02191)
- PyTorch reference: [tapas-tripura/Wavelet-Neural-Operator](https://github.com/tapas-tripura/Wavelet-Neural-Operator)

Use them when the target signal has localised or multi-scale structure that a wavelet decomposition captures better than Fourier mixing.

## Kolmogorov–Arnold Networks

Constructors (subset):

- `kan`, `efficient_kan`, `fastkan`
- `fourier_kan`, `chebyshev_kan`, `jacobi_kan`, `legendre_kan`, `gram_kan`
- `wavelet_kan`, `taylor_kan`, `hermite_kan`, `laguerre_kan`, `bernstein_kan`
- `relu_kan`, `rational_kan`, `sinc_kan`, `bsrbf_kan`
- `kan_conv1d`, `kan_conv2d`, `kan_conv3d`
- `kan_spectral_block1d/2d/3d`, `kan_res_block`, `kan_attention_block`

KANs replace the scalar weights of an MLP with learnable univariate functions on each edge, parameterised by a basis (B-spline, RBF, Fourier, orthogonal polynomial, wavelet, …). foundax ships 17 KAN variants plus matching convolutional / spectral / residual / attention blocks, all integrated with the `|` pipe API.

Use them when:

- you want an expressive MLP alternative with strong inductive bias
- the target function has known structure (smooth, periodic, multi-scale, polynomial) that a particular basis encodes well
- you want a drop-in pointwise mixer for FNO/UNet/transformer pipelines (`kan_spectral_block2d`, `kan_attention_block`)

See the [dedicated KAN page](kan.md) for the full variant list, basis details, and usage examples.

## Factory conventions

All core factories are exposed through `foundax.nn` and re-exported at package level:

```python
import foundax as fx

model = fx.fno2d(in_features=1, hidden_channels=32, n_modes=16)
model = fx.deeponet(branch_type="mlp", trunk_type="mlp")
model = fx.gnot(branch_sizes=[64], trunk_size=2)
```

The exact forward signature depends on the model family, so it is best to inspect the constructor in `foundax/nn.py` together with the implementation module in `foundax/architectures/` when integrating a new model.