# Changelog

All notable changes to foundax are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-11-28

### Added
- `fx.geofno`: Geo-FNO of Li et al. 2022 (arXiv:2207.05209) ported from neuraloperator/Geo-FNO: learned deformation (`IPHI`, optional design code), non-uniform Fourier transform onto a latent grid and back, FFT Fourier layers on the grid. Unit tests in `tests/test_geofno.py`, parity script `scripts/compare_geofno.py` (`pixi run -e dev verify-geofno`).
- `fx.describe(model)`: pure introspection of any `eqx.Module`. It returns the module tree (paths, shapes, dtypes, parameter counts, bytes), a summary, and, for a model built from `fx.block`, `|` pipes and `fx.dot` / `fx.add` / `fx.cat`, its dataflow graph (blocks with in/out channels, chains, combinator nodes with their branches, one input per `dot` branch). A plain module has a tree and no graph.
- `CITATION.cff`, `CONTRIBUTING.md`, `SECURITY.md`, and `CODE_OF_CONDUCT.md`.
- `docs/architectures.md` — single-page overview of every supported architecture with paper references, split into Core / KAN / Foundation tables.
- README badge row, status note, citation block, and a condensed architectures summary linking to the full docs page.
- `.pre-commit-config.yaml` (ruff + standard hygiene hooks) and `foundax/py.typed` marker for PEP 561 typing.
- GitHub issue templates (`bug_report`, `feature_request`) and a pull-request template with a pre-merge checklist.
- `pixi run fmt` and `pixi run all` tasks; format-check step in `.github/workflows/ci.yml`.
- `fx.layers.make_norm(kind, channels)` / `fx.layers.Norm`: `"layer"`, `"rms"`, `"instance"` and `"group"` normalisation of one channel-last sample of any spatial rank.
- `fx.unet3d(groups=...)`; UNets accept `norm` `"rms"` and `"instance"`.
- `t_scale` on `fx.SinusoidalTimeEmbedding`, `fx.dit2d` / `dit3d` and the Transolver factories: the time features are taken of `t_scale * t` (default 1; 1000 for t in [0, 1]).

### Changed
- **Breaking:** the previous `fx.geofno` (node-basis Fourier operator from PKU-CMEGroup/NeuralOperator, which has no learned deformation) is renamed `fx.geofno_nodal` / `GeoFNONodal`; same arguments as before.
- `docs/core-models.md` summary table now includes a Reference column linking to the canonical paper for each family.
- `pyproject.toml` adds `authors`, `classifiers`, `[project.urls]`, and a `py.typed` package-data entry.
- **Breaking:** `norm="batch"` raises `ValueError` in FNO1D/2D/3D and `fx.layers.SpectralBlock1d/2d/3d`: a model sees one unbatched sample, so there is no batch to normalise over. Use `"instance"` (per channel over the grid), `"layer"`, `"rms"` or `"group"`.
- **Breaking (checkpoints):** fixed arrays are no longer pytree leaves: the FNO coordinate grid, KAN knots, centres and scale factors, the SFNO Legendre table and quadrature weights, PiT distance matrices, GAOT latent positions, PCNO and `geofno_nodal` modes, the Transformer's sinusoidal encoding and DeepONet's fixed Fourier features are static `Constant` fields. Checkpoints of these models saved with 0.2 load with `scripts/migrate_checkpoint.py`.
- **Breaking:** UNet `norm="batch"` and `fx.mlp(batch_norm=True)` raise `ValueError` (no batch inside a model). The UNet default `norm="batch"` becomes `"instance"`, which computes the same on one sample and loads the same checkpoints.
- `fx.mgno1d` outputs change: its prolongation broadcast the width-1 grid to n×n, so every model ran on an n×n field internally; it now stays on the n×1 grid. Parameters are unchanged.
- A new DiT outputs zero: its final adaLN projection and output projection are zero-initialised, as in the paper (the code said so and did not). Trained checkpoints are unaffected.

### Fixed
- The foundation-model shortcuts `fx.poseidonT/B/L`, `fx.morph_*`, `fx.mpp_*`, `fx.pdeformer2_*`, `fx.dpot_*` (and `foundax.nn.walrus` / `bcat`) raised `TypeError`: they were copies passing arguments the namespace factories do not take. They are now the namespace factories themselves (`fx.poseidonT is fx.poseidon.T`), with their signatures and defaults.
- FNO `norm="batch"` was vmapped over the grid points, so it normalised over no axes and every layer returned its bias; FNO1D `norm="instance"` built a LayerNorm. FNO and the spectral blocks now use `make_norm`; `"instance"` is an instance norm.
- Fixed arrays stored as float leaves were trained and weight-decayed by any optimiser that takes the float leaves (`eqx.partition(model, eqx.is_inexact_array)`, as jNO does); `stop_gradient` in SFNO and DeepONet did not stop weight decay.
- DeepONet `combination_type="attention"` returned the same output at every query point (one key token, so the softmax was 1, and no residual). The branch output is now split into `n_tokens` tokens that each query point attends over, with a residual from the trunk; the combiner's checkpoint layout changes.
- Crashes on valid input: UNet `norm="group"` on channel-last data; UNet3D on odd sizes and with `groups > 1` (grouped convolution weights had the full input width, also in 1-D and 2-D); `fx.unet3d` with its default activation `"celu"`; MgNO on odd grids; `fx.mlp(layer_norm=True)` on `(N, D)` points; `fx.fno2d` / `fno3d` with `use_positions=True` (the lift did not count the coordinate channels).
- `fx.cno2d` returns one sample for one sample (it added a batch axis), and `fx.geofno_nodal` keeps the channel axis for `out_dim=1` (it squeezed the output to `(N,)`, and to a scalar for one node).
- Dropout keys: the GNOT family reused one key across blocks and sublayers, DeepONet used one mask for all query points and all samples of `evaluate_batch`, and DeepONet's self-attention ignored its `dropout_rate`. Every layer, point and sample now draws its own mask.
- Options that were accepted and ignored: GNOT/CGPTNO/MoEGPTNO `attn_type` now selects the linear-attention kernel (`"linear"` = `"l1"`, `"galerkin"`, `"l2"`; others raise); MgNO `padding_mode` (`circular`, `zeros`, `reflect`, `replicate`) reaches its convolutions; DeepONet `norm="rms"` is an RMS norm (it was a LayerNorm without affine) and DeepONet norms go through `make_norm` (`"batch"` raises).
- The GNOT MLP with `n_layers=1` built two layers from one key; it is one layer. FFNO kept its spectrum in `complex64` under x64. PiT computed a percentile for locality 100, where the mask keeps every key. `KANConv2d` uses the skip policy of `KANConv1d` / `KANConv3d` (a skip for `bspline`, `bsrbf` and `relu`) and takes one sample.
- `SinusoidalTimeEmbedding(2)` returned NaN (it divided by `half - 1`), and its frequencies were float32 under x64.
- DPOT's AFNO2D gave a different result on the GPU at power-of-2 sizes (1e-3 at 128): after mode mixing its spectrum was not Hermitian, so `irfft2` was undefined. AFNO2D and Geo-FNO now project the zero and Nyquist columns onto their Hermitian part first (`irfftn_hermitian`), which is the CPU result on every backend.
- The package metadata (`pyproject.toml`) declared the MIT license; foundax is EPL-2.0, as `LICENSE` and the README state (also in `CITATION.cff`).

## [0.2.0] - 2026-05-31

### Added
- **Kolmogorov–Arnold Networks**: 17 KAN variants (B-spline, RBF, Fourier, Chebyshev, Jacobi, Legendre, wavelet, Taylor, Hermite, Laguerre, Bernstein, ReLU, rational, sinc, Gram, BSRBF) plus 8 structural blocks (`kan_conv1d/2d/3d`, `kan_spectral_block1d/2d/3d`, `kan_res_block`, `kan_attention_block`). Full integration with the pipe API.
- **Flow-matching backbones**: `dit2d`, `dit3d` (Diffusion Transformer), `ffno2d`, `ffno3d` (Factorized FNO), `wno1d`, `wno2d`, `wno3d` (Wavelet Neural Operator) — all channel-last, pipe-compatible.
- **Verify pipeline**: Hydra-driven `scripts/verify.py` with per-model `pixi run -e dev verify-<name>` shortcuts. PyTorch → Equinox L2 comparison wired up for all 8 foundation models.

### Fixed
- PyTorch → Equinox weight transfer for Poseidon, MORPH, MPP, BCAT, DPOT, PROSE — random-weight equivalency at machine precision; DPOT down to L2 ≈ 1.13e-5 against the PyTorch reference.

## [0.1.x]

### Added
- Equinox foundation-model wrappers for Poseidon, MORPH, MPP, Walrus, BCAT, PDEformer-2, DPOT, PROSE.
- Pipe composition API: `fx.block`, `|`, `fx.dot`, `fx.add`, `fx.cat`, plus `SpectralBlock` layers.
- Core Equinox architectures: FNO, UNet, transformer, DeepONet, CNO, MgNO, GeoFNO, PCNO, GNOT family, PiT, PointNet.
- pixi-based development environment, ruff linting, pytest with `heavy` / `train` markers.
- mkdocs Material documentation site with per-family guides.
- Hugging Face integration for foundation-model weights (`FhG-IISB/foundax`).

[Unreleased]: https://github.com/FhG-IISB/foundax/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/FhG-IISB/foundax/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/FhG-IISB/foundax/releases/tag/v0.2.0
