# Architectures

Every model foundax ships, grouped into three tables: core architectures, KAN variants, and foundation-model wrappers. Constructors are all in `foundax/__init__.py`; usage examples live in the per-family pages linked below.

## 1. Core architectures

Direct Equinox implementations in `foundax/architectures/`, exposed via `foundax.nn`.

| Constructor(s) | Family | Reference | Notes |
|---|---|---|---|
| `fx.linear`, `fx.mlp` | Linear / MLP | — | Pointwise heads, simple regression, small shared subnetworks |
| `fx.fno1d`, `fx.fno2d`, `fx.fno3d` | Fourier Neural Operator | Li et al. 2020 — [arXiv:2010.08895](https://arxiv.org/abs/2010.08895) | Spectral mixing on structured grids |
| `fx.unet1d`, `fx.unet2d`, `fx.unet3d` | U-Net | Ronneberger et al. 2015 — [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) | Encoder-decoder with skip connections |
| `fx.transformer` | Generic transformer | Vaswani et al. 2017 — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762); JAX port from [voyager-jhk/JaxTransformer](https://github.com/voyager-jhk/JaxTransformer) | Sequence-to-sequence baseline |
| `fx.deeponet` | Deep Operator Network | Lu et al. 2019 — [arXiv:1910.03193](https://arxiv.org/abs/1910.03193) | Branch / trunk factorisation; configurable sub-networks |
| `fx.cno2d` | Continuous Neural Operator | Raonić et al. 2023 — [arXiv:2302.01178](https://arxiv.org/abs/2302.01178); code from [bogdanraonic3/AI_Science_Engineering](https://github.com/bogdanraonic3/AI_Science_Engineering) | Hierarchical convolutional operator on 2D fields |
| `fx.mgno1d`, `fx.mgno2d` | Multigrid Neural Operator | He et al. 2023 — [arXiv:2310.19809](https://arxiv.org/abs/2310.19809) | Restriction / prolongation inspired by multigrid solvers |
| `fx.geofno` | Geometry-aware FNO | Li et al. 2022 — [arXiv:2207.05209](https://arxiv.org/abs/2207.05209) | FNO with learned deformations for non-uniform layouts |
| `fx.pcno` | Point-Cloud Neural Operator | [PKU-CMEGroup/NeuralOperator](https://github.com/PKU-CMEGroup/NeuralOperator) | Operator learning on point clouds |
| `fx.pit` | Position-induced Transformer | Chen & Wu 2024 — [arXiv:2405.09285](https://arxiv.org/abs/2405.09285) | Coordinate-aware attention with distance-based weights |
| `fx.pointnet` | PointNet | Qi et al. 2017 — [arXiv:1612.00593](https://arxiv.org/abs/1612.00593) | Unordered point-set encoder–decoder |
| `fx.gnot`, `fx.cgptno`, `fx.moegptno` | GNOT family | Hao et al., ICML 2023 — [arXiv:2302.14376](https://arxiv.org/abs/2302.14376) | Transformer-style operator learning on irregular domains, with optional mixture-of-experts routing |
| `fx.dit2d`, `fx.dit3d` | Diffusion Transformer (DiT) | Peebles & Xie 2022 — [arXiv:2212.09748](https://arxiv.org/abs/2212.09748) | Patch + sinusoidal positional embedding; flow-matching backbone |
| `fx.ffno2d`, `fx.ffno3d` | Factorized FNO | Tran et al. 2023 — [arXiv:2111.13802](https://arxiv.org/abs/2111.13802) | `d` independent 1-D spectral convs; reduces O(m^d·C²) to O(d·m·C²) |
| `fx.wno1d`, `fx.wno2d`, `fx.wno3d` | Wavelet Neural Operator | Tripura & Chakraborty 2022 — [arXiv:2205.02191](https://arxiv.org/abs/2205.02191) | Multi-scale DWT decomposition with Daubechies-8 wavelets |
| `fx.transolver`, `fx.transolver2d`, `fx.transolver3d` | Transolver (Physics-Attention) | Wu et al., ICML 2024 — [arXiv:2402.02366](https://arxiv.org/abs/2402.02366); code from [thuml/Transolver](https://github.com/thuml/Transolver) | Slice-based linear attention over learnable physical groups; unstructured + structured 2D/3D variants |
| `fx.sfno2d` | Spherical Fourier Neural Operator | Bonev et al., ICML 2023 — [arXiv:2306.03838](https://arxiv.org/abs/2306.03838); reference code [NVIDIA/torch-harmonics](https://github.com/NVIDIA/torch-harmonics) | FFT replaced by a real-valued spherical harmonic transform (pure-JAX, no exotic deps); Gauss–Legendre or equiangular grid |

See [Core Models](core-models.md) for detailed usage notes per family.

---

## 2. Kolmogorov–Arnold Networks

All 17 variants share the same constructor surface (`in_features`, `output_dim`, `hidden_dims`, `num_layers`, `key=...`) plus basis-specific hyperparameters.

| Factory | Basis | Key hyperparameters | Reference |
|---|---|---|---|
| `fx.kan` | B-spline + SiLU residual | `grid_size`, `spline_order` | Liu et al. 2024 — [arXiv:2404.19756](https://arxiv.org/abs/2404.19756) |
| `fx.kan.efficient` | B-spline (memory-optimised) | `grid_size`, `spline_order` | Blealtan 2024 — [github.com/Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan) |
| `fx.kan.fast` | Gaussian RBF | `grid_size`, `grid_range` | Li 2024 — [arXiv:2405.06721](https://arxiv.org/abs/2405.06721) |
| `fx.kan.fourier` | sin/cos series | `num_frequencies` | GistNoesis 2024 — [github.com/GistNoesis/FourierKAN](https://github.com/GistNoesis/FourierKAN) |
| `fx.kan.chebyshev` | Chebyshev T_n | `degree` | SS 2024 — [arXiv:2405.07200](https://arxiv.org/abs/2405.07200) |
| `fx.kan.jacobi` | Jacobi P_n^(α,β) | `degree`, `alpha`, `beta` | Aghaei 2024 (*fKAN*) — [arXiv:2406.07456](https://arxiv.org/abs/2406.07456) |
| `fx.kan.legendre` | Legendre P_n | `degree` | Seydi 2024 — [arXiv:2406.02583](https://arxiv.org/abs/2406.02583) |
| `fx.kan.wavelet` | Mexican hat / Morlet / Shannon / DoG | `num_scales`, `wavelet_type` | Bozorgasl & Chen 2024 (*Wav-KAN*) — [arXiv:2405.12832](https://arxiv.org/abs/2405.12832) |
| `fx.kan.taylor` | Truncated power series | `degree` | Muyuzhierchengse 2024 — [github.com/Muyuzhierchengse/TaylorKAN](https://github.com/Muyuzhierchengse/TaylorKAN) |
| `fx.kan.hermite` | Hermite He_n | `degree` | Seydi 2024 — [arXiv:2406.02583](https://arxiv.org/abs/2406.02583) |
| `fx.kan.laguerre` | Laguerre L_n | `degree` | Seydi 2024 — [arXiv:2406.02583](https://arxiv.org/abs/2406.02583) |
| `fx.kan.bernstein` | Bernstein polynomials | `degree` | Seydi 2024 — [arXiv:2406.02583](https://arxiv.org/abs/2406.02583) |
| `fx.kan.relu` | (ReLU·ReLU)^order on a grid | `grid_size`, `order` | Qiu et al. 2024 — [arXiv:2406.02075](https://arxiv.org/abs/2406.02075) |
| `fx.kan.rational` | Padé-style rational Chebyshev | `degree` | Aghaei 2024 (*rKAN*) — [arXiv:2406.14495](https://arxiv.org/abs/2406.14495) |
| `fx.kan.sinc` | sinc basis on a grid | `grid_size`, `grid_range` | Yu et al. 2024 (*SincKAN*) — [arXiv:2410.04096](https://arxiv.org/abs/2410.04096) |
| `fx.kan.gram` | Orthonormal Legendre (Gram limit) | `degree` | Seydi 2024 — [arXiv:2406.02583](https://arxiv.org/abs/2406.02583) |
| `fx.kan.bsrbf` | B-spline + RBF concatenation | `grid_size`, `rbf_grid_size` | Ta 2024 (*BSRBF-KAN*) — [arXiv:2406.11173](https://arxiv.org/abs/2406.11173) |

### Structural blocks

| Factory | Description | Reference |
|---|---|---|
| `fx.kan.conv1d`, `fx.kan.conv2d`, `fx.kan.conv3d` | KAN convolution (any basis) | Bodner et al. 2024 — [arXiv:2406.13155](https://arxiv.org/abs/2406.13155) |
| `fx.kan.spectral_block1d/2d/3d` | FNO spectral block + KAN channel mixer | FNO: Li et al. 2020 — [arXiv:2010.08895](https://arxiv.org/abs/2010.08895); KAN: Liu et al. 2024 — [arXiv:2404.19756](https://arxiv.org/abs/2404.19756) |
| `fx.kan.res_block` | Residual KAN block | ResNet pattern: He et al. 2015 — [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
| `fx.kan.attention_block` | Pre-norm transformer block with KAN feed-forward | Yang & Wang 2024 (*KAT*) — [arXiv:2409.10594](https://arxiv.org/abs/2409.10594); attention: Vaswani et al. 2017 — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |

See the dedicated [KAN page](kan.md) for basis details, choice guidance, and runnable pipe examples.

---

## 3. Foundation-model wrappers

Each namespace wraps a vendored JAX implementation in `repos/jax_*`. Pretrained weights are downloaded separately, mostly from Hugging Face.

| Namespace | Variants | Backbone | Reference | Weights license |
|---|---|---|---|---|
| `fx.poseidon` | `T`, `B`, `L` | ScOT (Swin-style hierarchical operator transformer) | Herde et al. 2024 — [arXiv:2405.19101](https://arxiv.org/abs/2405.19101) | CC-BY-NC-4.0 |
| `fx.morph` | `Ti`, `S`, `M`, `L` | ViT3D regression | Rautela et al. 2025 — [arXiv:2509.21670](https://arxiv.org/abs/2509.21670) | MIT |
| `fx.mpp` | `Ti`, `S`, `B`, `L` | AViT (axial vision transformer) | McCabe et al., NeurIPS 2024 — [openreview/DKSI3bULiZ](https://openreview.net/forum?id=DKSI3bULiZ) | MIT |
| `fx.walrus` | `base` | Isotropic encoder–processor–decoder (1.29B params) | McCabe et al. 2025 — [arXiv:2511.15684](https://arxiv.org/abs/2511.15684) | MIT |
| `fx.bcat` | `base` | Block-causal transformer (patched spatio-temporal) | Liu et al. 2025 — [arXiv:2501.18972](https://arxiv.org/abs/2501.18972) | MIT |
| `fx.pdeformer2` | `small`, `base`, `fast` | Graphormer encoder + INR decoder with hypernetwork | Ye et al. 2025 — [arXiv:2507.15409](https://arxiv.org/abs/2507.15409) | Apache-2.0 |
| `fx.dpot` | `Ti`, `S`, `M`, `L`, `H` | DPOTNet (AFNO / Fourier-style mixing) | Hao et al., ICML 2024 — [arXiv:2403.03542](https://arxiv.org/abs/2403.03542) | Apache-2.0 |
| `fx.prose` | `fd_1to1`, `fd_2to1`, `ode_2to1`, `pde_2to1` | Transformer sequence-to-sequence (FD / ODE / PDE tasks) | Liu et al. 2023 — [arXiv:2309.16816](https://arxiv.org/abs/2309.16816); follow-up Sun et al. 2024 — [arXiv:2404.12355](https://arxiv.org/abs/2404.12355) | MIT |

> Weights keep their upstream licenses — see [THIRD_PARTY_LICENSES](https://github.com/FhG-IISB/foundax/blob/main/THIRD_PARTY_LICENSES); Poseidon weights are non-commercial.

See [Foundation Models](equinox-architectures.md) for per-namespace usage.

For the pipe API (`fx.block`, `|`, `fx.dot`, `fx.add`, `fx.cat`) and time-conditioning primitives, see [Getting Started](getting-started.md).

---

## 4. Parity verification against PyTorch upstreams

For every wired-up architecture we run a numerical-parity test that **instantiates the actual upstream PyTorch class, copies its weights tensor-by-tensor into the foundax Equinox port, runs both forwards on the same input, and compares element-wise**. Numbers below are generated by `scripts/parity_table.py` (which calls every compare script and parses its output), and confirmed to match end-to-end via `pixi run verify 'models=[...]'`.

| Architecture | Test | Input → Output | Max abs diff | Rel L2 | PyTorch reference |
|---|---|---|---|---|---|
| Transolver | Transolver Irregular | (64, 2) coords + (64, 1) func → (64, 1) | 3.066e-06 | 3.514e-05 | [thuml/Transolver](https://github.com/thuml/Transolver) |
|  | Transolver Structured2D | (16, 16, 2) coords + (16, 16, 1) func → (256, 1)¹ | 8.401e-07 | 3.529e-06 |  |
| SFNO | SHT forward | (32, 64) lat-lon → (8, 8) | 3.332e-08 | 1.610e-07 | [NVIDIA/torch-harmonics](https://github.com/NVIDIA/torch-harmonics) |
|  | SHT inverse | (8, 8) spectral → (32, 64) | 1.192e-07 | 1.105e-07 |  |
|  | SphericalConv2d | (32, 64, 2) → (32, 64, 3) | 3.353e-08 | 2.249e-07 |  |
|  | SFNO2d full | (16, 32, 3) → (16, 32, 2) | 4.780e-05 | 4.182e-05 |  |
| FFNO | FactorizedSpectralConv2d | (16, 16, 8) → (16, 16, 8) | 2.384e-07 | 1.159e-07 | [alasdairtran/fourierflow](https://github.com/alasdairtran/fourierflow) |
|  | FactorizedSpectralConv3d | (8, 10, 12, 6) → (8, 10, 12, 6) | 5.364e-07 | 1.340e-07 |  |
| FNO (+ Geo-FNO) | SpectralConv1d | (32, 4) → (32, 6) | 3.576e-07 | 1.797e-07 | [neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator) |
|  | SpectralConv2d | (16, 20, 3) → (16, 20, 5) | 3.576e-07 | 1.921e-07 |  |
|  | SpectralConv3d | (12, 14, 16, 3) → (12, 14, 16, 4) | 3.576e-07 | 2.049e-07 |  |
| WNO² | WNO1d structural | (32, 2) → (32, 2) | — | — | [TapasTripura/WNO](https://github.com/TapasTripura/WNO) |
|  | WNO2d structural | (32, 32, 2) → (32, 32, 2) | — | — |  |
|  | WNO3d gradient-flow | (16, 16, 16, 1) → (16, 16, 16, 1) | — | — |  |
| DiT | DiTBlock | (16, 32) tokens + (32,) cond → (16, 32) | 3.576e-07 | 5.117e-08 | [facebookresearch/DiT](https://github.com/facebookresearch/DiT) |
| GNOT | LinearAttention | (1, 12, 32) → (1, 12, 32) | 5.960e-08 | 1.269e-07 | [HaoZhongkai/GNOT](https://github.com/HaoZhongkai/GNOT) |
|  | LinearCrossAttention | (1, 10, 32) query + 2×(1, 16, 32) branches → (1, 10, 32) | 1.192e-07 | 1.179e-07 |  |
|  | CrossAttentionBlock | (1, 10, 32) query + 2×(1, 16, 32) branches → (1, 10, 32) | 1.725e-04 | 5.043e-05 |  |

¹ foundax's structured-2D input is `(16, 16, …)` channel-last; the parity output shape is flattened to `(256, 1)` only to match upstream's `(B, N, C)` layout for the diff. The foundax model returns `(16, 16, 1)` natively.

² WNO is a structural-only check rather than a numerical parity test because upstream uses Daubechies-6 with symmetric extension via `pytorch_wavelets`, while foundax uses Daubechies-8 with zero-boundary in pure JAX. No shared input + shared weights configuration produces matching output.

**Notes on what's compared.**
- *Transolver*: full upstream `Model` class, both Irregular and Structured 2D variants. The most complete comparison in the table.
- *SFNO*: upstream `RealSHT` / `InverseRealSHT` primitives are real; the wrapping SFNO recipe is built in PT to match foundax's own (no canonical SFNO class exists upstream).
- *FFNO*: upstream `SpectralConv2d` / 3D primitives directly. The full upstream `FNOFactorized2DBlock` has per-block FeedForward MLPs not present in our cleaner wrapper.
- *FNO*: upstream `SpectralConv` (legacy module) with `factorization=None`, `fft_norm='ortho'`, `bias=False` to match foundax conventions.
- *DiT*: upstream `DiTBlock` against an Equinox port that mirrors upstream's design choices (SiLU + GELU-tanh + no-affine LN). foundax's user-facing `dit2d` uses different conventions by design (no class labels, exact GELU, no `learn_sigma`).
- *GNOT*: upstream `LinearAttention`, `LinearCrossAttention`, and `CrossAttentionBlock` primitives. The full `CGPTNO.forward` needs `dgl` for graph batching, which the parity test bypasses via sys.modules stub.

**Metric.** "Max abs diff" is element-wise `max(|pt − jax|)` on a forward pass with identical inputs and transferred weights. "Rel L2" is `‖pt − jax‖₂ / ‖pt‖₂` — the closest analog to a relative RMSE. All numerical values are at float32 noise floor.

**Reproduce.** Install dev deps (`pixi install -e dev`), then either:
- Generate the table from scratch: `pixi run --environment dev python scripts/parity_table.py`
- Run the orchestrated Hydra pipeline: `pixi run verify 'models=[transolver,sfno,ffno,fno,wno,dit,gnot]'`
- Per-model: `pixi run verify-transolver`, `verify-sfno`, `verify-ffno`, `verify-fno`, `verify-wno`, `verify-dit`, `verify-gnot`

Compare-script source lives in `scripts/compare_<name>.py`; the shared PT→EQX weight-transfer helpers are in `scripts/_pt2eqx.py`.
