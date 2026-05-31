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
| `fx.deeponet` | Deep Operator Network | Lu et al. 2021 — [arXiv:1910.03193](https://arxiv.org/abs/1910.03193) | Branch / trunk factorisation; configurable sub-networks |
| `fx.cno2d` | Continuous Neural Operator | Raonić et al. 2023 — [arXiv:2302.01178](https://arxiv.org/abs/2302.01178); code from [bogdanraonic3/AI_Science_Engineering](https://github.com/bogdanraonic3/AI_Science_Engineering) | Hierarchical convolutional operator on 2D fields |
| `fx.mgno1d`, `fx.mgno2d` | Multigrid Neural Operator | He et al. 2024 — [arXiv:2310.19809](https://arxiv.org/abs/2310.19809) | Restriction / prolongation inspired by multigrid solvers |
| `fx.geofno` | Geometry-aware FNO | Li et al. 2022 — [arXiv:2207.05209](https://arxiv.org/abs/2207.05209) | FNO with learned deformations for non-uniform layouts |
| `fx.pcno` | Point-Cloud Neural Operator | [PKU-CMEGroup/NeuralOperator](https://github.com/PKU-CMEGroup/NeuralOperator) | Operator learning on point clouds |
| `fx.pit` | Position-induced Transformer | Chen et al. 2024 — [arXiv:2405.09285](https://arxiv.org/abs/2405.09285) | Coordinate-aware attention with distance-based weights |
| `fx.pointnet` | PointNet | Qi et al. 2017 — [arXiv:1612.00593](https://arxiv.org/abs/1612.00593) | Unordered point-set encoder–decoder |
| `fx.gnot`, `fx.cgptno`, `fx.moegptno` | GNOT family | Hao et al., ICML 2023 — [arXiv:2302.14376](https://arxiv.org/abs/2302.14376) | Transformer-style operator learning on irregular domains, with optional mixture-of-experts routing |
| `fx.dit2d`, `fx.dit3d` | Diffusion Transformer (DiT) | Peebles & Xie 2022 — [arXiv:2212.09748](https://arxiv.org/abs/2212.09748) | Patch + sinusoidal positional embedding; flow-matching backbone |
| `fx.ffno2d`, `fx.ffno3d` | Factorized FNO | Tran et al. 2023 — [arXiv:2111.13802](https://arxiv.org/abs/2111.13802) | `d` independent 1-D spectral convs; reduces O(m^d·C²) to O(d·m·C²) |
| `fx.wno1d`, `fx.wno2d`, `fx.wno3d` | Wavelet Neural Operator | Tripura & Chakraborty 2022 — [arXiv:2205.02191](https://arxiv.org/abs/2205.02191) | Multi-scale DWT decomposition with Daubechies-8 wavelets |

See [Core Models](core-models.md) for detailed usage notes per family.

---

## 2. Kolmogorov–Arnold Networks

All 17 variants share the same constructor surface (`in_features`, `output_dim`, `hidden_dims`, `num_layers`, `key=...`) plus basis-specific hyperparameters.

| Factory | Basis | Key hyperparameters | Reference |
|---|---|---|---|
| `fx.kan` | B-spline + SiLU residual | `grid_size`, `spline_order` | Liu et al. 2024 — [arXiv:2404.19756](https://arxiv.org/abs/2404.19756) |
| `fx.efficient_kan` | B-spline (memory-optimised) | `grid_size`, `spline_order` | Blealtan 2024 — [github.com/Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan) |
| `fx.fastkan` | Gaussian RBF | `grid_size`, `grid_range` | Li 2024 — [arXiv:2405.06721](https://arxiv.org/abs/2405.06721) |
| `fx.fourier_kan` | sin/cos series | `num_frequencies` | GistNoesis 2024 — [github.com/GistNoesis/FourierKAN](https://github.com/GistNoesis/FourierKAN) |
| `fx.chebyshev_kan` | Chebyshev T_n | `degree` | SS 2024 — [arXiv:2405.07200](https://arxiv.org/abs/2405.07200) |
| `fx.jacobi_kan` | Jacobi P_n^(α,β) | `degree`, `alpha`, `beta` | Aghaei 2024 (*fKAN*) — [arXiv:2406.07456](https://arxiv.org/abs/2406.07456) |
| `fx.legendre_kan` | Legendre P_n | `degree` | Seydi 2024 — [arXiv:2406.02583](https://arxiv.org/abs/2406.02583) |
| `fx.wavelet_kan` | Mexican hat / Morlet / Shannon / DoG | `num_scales`, `wavelet_type` | Bozorgasl & Chen 2024 (*Wav-KAN*) — [arXiv:2405.12832](https://arxiv.org/abs/2405.12832) |
| `fx.taylor_kan` | Truncated power series | `degree` | Muyuzhierchengse 2024 — [github.com/Muyuzhierchengse/TaylorKAN](https://github.com/Muyuzhierchengse/TaylorKAN) |
| `fx.hermite_kan` | Hermite He_n | `degree` | Seydi 2024 — [arXiv:2406.02583](https://arxiv.org/abs/2406.02583) |
| `fx.laguerre_kan` | Laguerre L_n | `degree` | Seydi 2024 — [arXiv:2406.02583](https://arxiv.org/abs/2406.02583) |
| `fx.bernstein_kan` | Bernstein polynomials | `degree` | Seydi 2024 — [arXiv:2406.02583](https://arxiv.org/abs/2406.02583) |
| `fx.relu_kan` | (ReLU·ReLU)^order on a grid | `grid_size`, `order` | Qiu et al. 2024 — [arXiv:2406.02075](https://arxiv.org/abs/2406.02075) |
| `fx.rational_kan` | Padé-style rational Chebyshev | `degree` | Aghaei 2024 (*rKAN*) — [arXiv:2406.14495](https://arxiv.org/abs/2406.14495) |
| `fx.sinc_kan` | sinc basis on a grid | `grid_size`, `grid_range` | Yu et al. 2024 (*SincKAN*) — [arXiv:2410.04096](https://arxiv.org/abs/2410.04096) |
| `fx.gram_kan` | Orthonormal Legendre (Gram limit) | `degree` | Seydi 2024 — [arXiv:2406.02583](https://arxiv.org/abs/2406.02583) |
| `fx.bsrbf_kan` | B-spline + RBF concatenation | `grid_size`, `rbf_grid_size` | Ta 2024 (*BSRBF-KAN*) — [arXiv:2406.11173](https://arxiv.org/abs/2406.11173) |

### Structural blocks

| Factory | Description | Reference |
|---|---|---|
| `fx.kan_conv1d`, `fx.kan_conv2d`, `fx.kan_conv3d` | KAN convolution (any basis) | Bodner et al. 2024 — [arXiv:2406.13155](https://arxiv.org/abs/2406.13155) |
| `fx.kan_spectral_block1d/2d/3d` | FNO spectral block + KAN channel mixer | FNO: Li et al. 2020 — [arXiv:2010.08895](https://arxiv.org/abs/2010.08895); KAN: Liu et al. 2024 — [arXiv:2404.19756](https://arxiv.org/abs/2404.19756) |
| `fx.kan_res_block` | Residual KAN block | ResNet pattern: He et al. 2015 — [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
| `fx.kan_attention_block` | Pre-norm transformer block with KAN feed-forward | Yang & Wang 2024 (*KAT*) — [arXiv:2409.10594](https://arxiv.org/abs/2409.10594); attention: Vaswani et al. 2017 — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |

See the dedicated [KAN page](kan.md) for basis details, choice guidance, and runnable pipe examples.

---

## 3. Foundation-model wrappers

Each namespace wraps a vendored JAX implementation in `repos/jax_*`. Pretrained weights are downloaded separately, mostly from Hugging Face.

| Namespace | Variants | Backbone | Reference | Weights license |
|---|---|---|---|---|
| `fx.poseidon` | `T`, `B`, `L` | ScOT (Swin-style hierarchical operator transformer) | Herde et al. 2024 — [arXiv:2405.19101](https://arxiv.org/abs/2405.19101) | CC-BY-NC-4.0 |
| `fx.morph` | `Ti`, `S`, `M`, `L` | ViT3D regression | Rautela et al. 2025 — [arXiv:2509.21670](https://arxiv.org/abs/2509.21670) | MIT |
| `fx.mpp` | `Ti`, `S`, `B`, `L` | AViT (axial vision transformer) | McCabe et al., NeurIPS 2024 — [openreview/DKSI3bULiZ](https://openreview.net/forum?id=DKSI3bULiZ) | MIT |
| `fx.walrus` | `base` | Isotropic encoder–processor–decoder (1.29B params) | Bodner et al. 2024 — [arXiv:2511.15684](https://arxiv.org/abs/2511.15684) | MIT |
| `fx.bcat` | `base` | Block-causal transformer (patched spatio-temporal) | Liu et al. 2025 — [arXiv:2501.18972](https://arxiv.org/abs/2501.18972) | MIT |
| `fx.pdeformer2` | `small`, `base`, `fast` | Graphormer encoder + INR decoder with hypernetwork | Shi et al. 2025 — [arXiv:2502.14844](https://arxiv.org/abs/2502.14844) | Apache-2.0 |
| `fx.dpot` | `Ti`, `S`, `M`, `L`, `H` | DPOTNet (AFNO / Fourier-style mixing) | Hao et al., ICML 2024 — [arXiv:2403.03542](https://arxiv.org/abs/2403.03542) | Apache-2.0 |
| `fx.prose` | `fd_1to1`, `fd_2to1`, `ode_2to1`, `pde_2to1` | Transformer sequence-to-sequence (FD / ODE / PDE tasks) | Sun et al. 2024 — [arXiv:2404.12355](https://arxiv.org/abs/2404.12355) | MIT |

> Weights keep their upstream licenses — see [THIRD_PARTY_LICENSES](https://github.com/FhG-IISB/foundax/blob/main/THIRD_PARTY_LICENSES); Poseidon weights are non-commercial.

See [Foundation Models](equinox-architectures.md) for per-namespace usage.

For the pipe API (`fx.block`, `|`, `fx.dot`, `fx.add`, `fx.cat`) and time-conditioning primitives, see [Getting Started](getting-started.md).
