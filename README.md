<p align="center">
  <img src="assets/logo.png" alt="foundax logo" width="400">
</p>

<p align="center">
    <a href="https://github.com/FhG-IISB/foundax/actions/workflows/ci.yml">
        <img src="https://img.shields.io/github/actions/workflow/status/FhG-IISB/foundax/ci.yml?branch=main&label=tests" alt="Tests"/>
    </a>
    <a href="LICENSE">
        <img src="https://img.shields.io/badge/license-EPL_2.0-2ea44f" alt="License"/>
    </a>
    <a href="https://huggingface.co/FhG-IISB/foundax">
        <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FhG--IISB%2Ffoundax-ff9d00" alt="Hugging Face"/>
    </a>
</p>

A small Equinox-based collection of JAX models for operator learning and PDE surrogates: a handful of core architectures, the KAN family, and wrappers around eight vendored foundation models. Plays nicely with [jNO](https://github.com/FhG-IISB/jNO).

> Early days — APIs may shift between minor versions.

## Install

```bash
pip install foundax
```

Development setup uses pixi — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Supported architectures

Full list with paper references: [`docs/architectures.md`](docs/architectures.md).

### Core architectures

| Family | Constructors | Reference |
|---|---|---|
| Linear / MLP | `fx.linear`, `fx.mlp` | — |
| Fourier Neural Operator | `fx.fno1d/2d/3d` | Li et al. 2020 — [arXiv:2010.08895](https://arxiv.org/abs/2010.08895) |
| U-Net | `fx.unet1d/2d/3d` | Ronneberger et al. 2015 — [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) |
| Generic transformer | `fx.transformer` | Vaswani et al. 2017 — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| DeepONet | `fx.deeponet` | Lu et al. 2019 — [arXiv:1910.03193](https://arxiv.org/abs/1910.03193) |
| Continuous Neural Operator | `fx.cno2d` | Raonić et al. 2023 — [arXiv:2302.01178](https://arxiv.org/abs/2302.01178) |
| Multigrid Neural Operator | `fx.mgno1d/2d` | He et al. 2023 — [arXiv:2310.19809](https://arxiv.org/abs/2310.19809) |
| Geometry-aware FNO | `fx.geofno` | Li et al. 2022 — [arXiv:2207.05209](https://arxiv.org/abs/2207.05209) |
| Point-Cloud Neural Operator | `fx.pcno` | [PKU-CMEGroup/NeuralOperator](https://github.com/PKU-CMEGroup/NeuralOperator) |
| Position-induced Transformer | `fx.pit` | Chen & Wu 2024 — [arXiv:2405.09285](https://arxiv.org/abs/2405.09285) |
| PointNet | `fx.pointnet` | Qi et al. 2017 — [arXiv:1612.00593](https://arxiv.org/abs/1612.00593) |
| GNOT family | `fx.gnot`, `fx.cgptno`, `fx.moegptno` | Hao et al., ICML 2023 — [arXiv:2302.14376](https://arxiv.org/abs/2302.14376) |
| Diffusion Transformer (DiT) | `fx.dit2d/3d` | Peebles & Xie 2022 — [arXiv:2212.09748](https://arxiv.org/abs/2212.09748) |
| Factorized FNO | `fx.ffno2d/3d` | Tran et al. 2023 — [arXiv:2111.13802](https://arxiv.org/abs/2111.13802) |
| Wavelet Neural Operator | `fx.wno1d/2d/3d` | Tripura & Chakraborty 2022 — [arXiv:2205.02191](https://arxiv.org/abs/2205.02191) |

### Kolmogorov–Arnold Networks

| Factory                   | Basis                                | Reference |
|---------------------------|--------------------------------------|-----------|
| `fx.kan`                  | B-spline + SiLU residual             | Liu et al. 2024 — [arXiv:2404.19756](https://arxiv.org/abs/2404.19756) |
| `fx.efficient_kan`        | B-spline (memory-optimised)          | Blealtan 2024 — [github.com/Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan) |
| `fx.fastkan`              | Gaussian RBF                         | Li 2024 — [arXiv:2405.06721](https://arxiv.org/abs/2405.06721) |
| `fx.fourier_kan`          | sin/cos series                       | GistNoesis 2024 — [github.com/GistNoesis/FourierKAN](https://github.com/GistNoesis/FourierKAN) |
| `fx.chebyshev_kan`        | Chebyshev T_n                        | SS 2024 — [arXiv:2405.07200](https://arxiv.org/abs/2405.07200) |
| `fx.jacobi_kan`           | Jacobi P_n^(α,β)                     | Aghaei 2024 (*fKAN*) — [arXiv:2406.07456](https://arxiv.org/abs/2406.07456) |
| `fx.legendre_kan`         | Legendre P_n                         | Seydi 2024 — [arXiv:2406.02583](https://arxiv.org/abs/2406.02583) |
| `fx.wavelet_kan`          | Mexican hat / Morlet / Shannon / DoG | Bozorgasl & Chen 2024 (*Wav-KAN*) — [arXiv:2405.12832](https://arxiv.org/abs/2405.12832) |
| `fx.taylor_kan`           | Truncated power series               | [github.com/Muyuzhierchengse/TaylorKAN](https://github.com/Muyuzhierchengse/TaylorKAN) |
| `fx.hermite_kan`          | Hermite He_n                         | Seydi 2024 — [arXiv:2406.02583](https://arxiv.org/abs/2406.02583) |
| `fx.laguerre_kan`         | Laguerre L_n                         | Seydi 2024 — [arXiv:2406.02583](https://arxiv.org/abs/2406.02583) |
| `fx.bernstein_kan`        | Bernstein polynomials                | Seydi 2024 — [arXiv:2406.02583](https://arxiv.org/abs/2406.02583) |
| `fx.relu_kan`             | (ReLU·ReLU)^order on a grid          | Qiu et al. 2024 — [arXiv:2406.02075](https://arxiv.org/abs/2406.02075) |
| `fx.rational_kan`         | Padé-style rational Chebyshev        | Aghaei 2024 (*rKAN*) — [arXiv:2406.14495](https://arxiv.org/abs/2406.14495) |
| `fx.sinc_kan`             | sinc basis on a grid                 | Yu et al. 2024 (*SincKAN*) — [arXiv:2410.04096](https://arxiv.org/abs/2410.04096) |
| `fx.gram_kan`             | Orthonormal Legendre (Gram limit)    | Seydi 2024 — [arXiv:2406.02583](https://arxiv.org/abs/2406.02583) |
| `fx.bsrbf_kan`            | B-spline + RBF concatenation         | Ta 2024 (*BSRBF-KAN*) — [arXiv:2406.11173](https://arxiv.org/abs/2406.11173) |
### Foundation-model wrappers

| Namespace | Variants | Backbone | Reference |
|---|---|---|---|
| `fx.poseidon` | T, B, L | ScOT (Swin operator transformer) | Herde et al. 2024 — [arXiv:2405.19101](https://arxiv.org/abs/2405.19101) |
| `fx.morph` | Ti, S, M, L | ViT3D regression | Rautela et al. 2025 — [arXiv:2509.21670](https://arxiv.org/abs/2509.21670) |
| `fx.mpp` | Ti, S, B, L | AViT (axial ViT) | McCabe et al., NeurIPS 2024 — [openreview/DKSI3bULiZ](https://openreview.net/forum?id=DKSI3bULiZ) |
| `fx.walrus` | base | Encoder-processor-decoder (1.29B) | McCabe et al. 2025 — [arXiv:2511.15684](https://arxiv.org/abs/2511.15684) |
| `fx.bcat` | base | Block-causal transformer | Liu et al. 2025 — [arXiv:2501.18972](https://arxiv.org/abs/2501.18972) |
| `fx.pdeformer2` | small, base, fast | Graphormer + INR | Ye et al. 2025 — [arXiv:2507.15409](https://arxiv.org/abs/2507.15409) |
| `fx.dpot` | Ti, S, M, L, H | DPOTNet (AFNO) | Hao et al., ICML 2024 — [arXiv:2403.03542](https://arxiv.org/abs/2403.03542) |
| `fx.prose` | fd_1to1, fd_2to1, ode_2to1, pde_2to1 | Seq-to-seq transformer | Liu et al. 2023 — [arXiv:2309.16816](https://arxiv.org/abs/2309.16816); follow-up Sun et al. 2024 — [arXiv:2404.12355](https://arxiv.org/abs/2404.12355) |

Pretrained weights keep their upstream licenses — see [`THIRD_PARTY_LICENSES`](THIRD_PARTY_LICENSES).

## Quick Start

```python
import foundax as fx

# Core models
model = fx.mlp(in_features=2, output_dim=1, hidden_dims=64, num_layers=3)
model = fx.fno2d(in_features=1, hidden_channels=32, n_modes=16)
model = fx.unet2d(in_channels=1, out_channels=1)
model = fx.deeponet(branch_type="mlp", trunk_type="mlp")

# KAN family (one of 17 variants)
model = fx.fastkan(in_features=2, output_dim=1, hidden_dims=64, num_layers=3)

# Foundation wrappers (namespace style)
model = fx.poseidon.T()           # T/B/L
model = fx.morph.S()              # Ti/S/M/L
model = fx.mpp.B(n_states=12)     # Ti/S/B/L
model = fx.walrus.base()
model = fx.bcat.base()
model = fx.pdeformer2.small()     # small/base/fast
model = fx.dpot.Ti()              # Ti/S/M/L/H
model, variables = fx.prose.fd_1to1()
```

## Composable Pipe API

Wrap any model or layer with `fx.block()` and chain them with `|`.
Channel mismatches are caught at construction time with a clear error message.

```python
import jax
import foundax as fx

ks = jax.random.split(jax.random.PRNGKey(0), 8)

# ── Build a 2-D FNO-style pipeline from individual spectral layers ──────────
lift    = fx.block(fx.layers.SpectralBlock2d(1,  32, n_modes=16, key=ks[0]), name="lift")
s1      = fx.block(fx.layers.SpectralBlock2d(32, 32, n_modes=16, key=ks[1]))
s2      = fx.block(fx.layers.SpectralBlock2d(32, 32, n_modes=16, key=ks[2]))
s3      = fx.block(fx.layers.SpectralBlock2d(32, 32, n_modes=16, key=ks[3]))
project = fx.block(fx.layers.SpectralBlock2d(32,  1, n_modes=16, key=ks[4]), name="project")

model = lift | s1 | s2 | s3 | project   # Pipe of 5 blocks

# ── Existing full models work as blocks too ──────────────────────────────────
encoder = fx.block(fx.fno2d(in_features=3, hidden_channels=32, n_modes=16, key=ks[5]))
decoder = fx.block(fx.layers.SpectralBlock2d(32, 1, n_modes=16, key=ks[6]))

model = encoder | decoder

# ── Multi-input combinators (DeepONet-style) ─────────────────────────────────
branch = (
    fx.block(fx.layers.SpectralBlock1d(1, 32, n_modes=16, key=ks[0]))
    | fx.block(fx.mlp(in_features=32, output_dim=64, hidden_dims=64, key=ks[1]))
)
trunk = fx.block(fx.mlp(in_features=2, output_dim=64, hidden_dims=64, key=ks[2]))

model = fx.dot(branch, trunk)   # branch(u) · trunk(y)  →  (N_pts,)

# Also available: fx.add(a, b)  — elementwise sum of two branches
#                 fx.cat(a, b)  — concatenate outputs along the channel axis

# ── All pipe models are plain Equinox modules ────────────────────────────────
import equinox as eqx, optax, jax.numpy as jnp

opt   = optax.adam(1e-3)
state = opt.init(eqx.filter(model, eqx.is_array))

@eqx.filter_jit
def step(model, state, u, y, target):
    loss, grads = eqx.filter_value_and_grad(
        lambda m: jnp.mean((m(u, y) - target) ** 2)
    )(model)
    updates, state = opt.update(grads, state, eqx.filter(model, eqx.is_array))
    return eqx.apply_updates(model, updates), state, loss
```

## Integration With jNO

```python
import foundax as fx
import jno
import optax

net = jno.nn.wrap(fx.poseidon.T(num_channels=5, num_out_channels=1))
net.optimizer(
    optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=optax.schedules.warmup_cosine_decay_schedule(
                init_value=1e-7,
                peak_value=1e-3,
                warmup_steps=500,
                decay_steps=10000,
                end_value=1e-6,
            ),
            weight_decay=1e-4,
        ),
    )
)
net.initialize('./poseidonT.eqx')
net.mask(param_mask).lora(rank=4)
```

## Citation

If you use foundax in academic work, the accompanying paper is the [jNO preprint](https://arxiv.org/abs/2605.10159) (`arXiv:2605.10159`). A machine-readable [`CITATION.cff`](CITATION.cff) is provided.

## License

EPL-2.0 — see [LICENSE](LICENSE). Vendored foundation-model code and pretrained weights keep their original licenses (see [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES)); Poseidon weights are non-commercial.
