# Kolmogorov–Arnold Networks (KAN)

Kolmogorov–Arnold Networks replace the scalar weights of an MLP with **learnable univariate functions on each edge**. Different variants choose different bases for those univariate functions — B-splines, RBFs, Fourier series, orthogonal polynomials, wavelets, and many more.

`foundax` ships **17 KAN variants**, **8 structural building blocks**, and standalone + pipe-API integration for all of them. Every layer exposes the same `in_features`/`out_features` static fields as `fx.mlp`, so it slots into a `|` pipeline like any other foundax module.

## Quick start

```python
import jax
import foundax as fx

key = jax.random.PRNGKey(0)

# Standalone factory — same surface as fx.mlp(...)
model = fx.kan.fast(in_features=2, output_dim=1, hidden_dims=64, num_layers=3, key=key)
y = model(jnp.ones((128, 2)))  # (128, 1)

# Pipe API — chain KAN layers with `|`, mix bases freely
ks = jax.random.split(key, 3)
pipe = (
    fx.block(fx.layers.FastKANLayer(2, 32, grid_size=8, key=ks[0]))
    | fx.block(fx.layers.ChebyshevKANLayer(32, 32, degree=5, key=ks[1]))
    | fx.block(fx.layers.FastKANLayer(32, 1, grid_size=8, key=ks[2]))
)

# Combinators work too — DeepONet-style with a KAN branch + KAN trunk
branch = fx.block(fx.kan.fast(in_features=3, output_dim=16, key=ks[0]))
trunk  = fx.block(fx.kan.chebyshev(in_features=2, output_dim=16, key=ks[1]))
op = fx.dot(branch, trunk)
```

## Variant overview

All 17 variants share the same constructor surface (`in_features`, `output_dim`, `hidden_dims`, `num_layers`, `key=...`) plus basis-specific hyperparameters. Each row links to the primary publication.

| Factory                   | Basis                                | Key hyperparameters              | Reference |
|---------------------------|--------------------------------------|----------------------------------|-----------|
| `fx.kan`                  | B-spline + SiLU residual             | `grid_size`, `spline_order`      | Liu et al. 2024 — [arXiv:2404.19756](https://arxiv.org/abs/2404.19756) |
| `fx.kan.efficient`        | B-spline (memory-optimised)          | `grid_size`, `spline_order`      | Blealtan 2024 — [github.com/Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan) |
| `fx.kan.fast`              | Gaussian RBF                         | `grid_size`, `grid_range`        | Li 2024 — [arXiv:2405.06721](https://arxiv.org/abs/2405.06721) |
| `fx.kan.fourier`          | sin/cos series                       | `num_frequencies`                | GistNoesis 2024 — [github.com/GistNoesis/FourierKAN](https://github.com/GistNoesis/FourierKAN) |
| `fx.kan.chebyshev`        | Chebyshev T_n                        | `degree`                         | SS 2024 — [arXiv:2405.07200](https://arxiv.org/abs/2405.07200) |
| `fx.kan.jacobi`           | Jacobi P_n^(α,β)                     | `degree`, `alpha`, `beta`        | Aghaei 2024 (*fKAN*) — [arXiv:2406.07456](https://arxiv.org/abs/2406.07456) |
| `fx.kan.legendre`         | Legendre P_n                         | `degree`                         | Seydi 2024 — [arXiv:2406.02583](https://arxiv.org/abs/2406.02583) |
| `fx.kan.wavelet`          | Mexican hat / Morlet / Shannon / DoG | `num_scales`, `wavelet_type`     | Bozorgasl & Chen 2024 (*Wav-KAN*) — [arXiv:2405.12832](https://arxiv.org/abs/2405.12832) |
| `fx.kan.taylor`           | Truncated power series               | `degree`                         | [github.com/Muyuzhierchengse/TaylorKAN](https://github.com/Muyuzhierchengse/TaylorKAN) |
| `fx.kan.hermite`          | Hermite He_n                         | `degree`                         | Seydi 2024 — [arXiv:2406.02583](https://arxiv.org/abs/2406.02583) |
| `fx.kan.laguerre`         | Laguerre L_n                         | `degree`                         | Seydi 2024 — [arXiv:2406.02583](https://arxiv.org/abs/2406.02583) |
| `fx.kan.bernstein`        | Bernstein polynomials                | `degree`                         | Seydi 2024 — [arXiv:2406.02583](https://arxiv.org/abs/2406.02583) |
| `fx.kan.relu`             | (ReLU·ReLU)^order on a grid          | `grid_size`, `order`             | Qiu et al. 2024 — [arXiv:2406.02075](https://arxiv.org/abs/2406.02075) |
| `fx.kan.rational`         | Padé-style rational Chebyshev        | `degree`                         | Aghaei 2024 (*rKAN*) — [arXiv:2406.14495](https://arxiv.org/abs/2406.14495) |
| `fx.kan.sinc`             | sinc basis on a grid                 | `grid_size`, `grid_range`        | Yu et al. 2024 (*SincKAN*) — [arXiv:2410.04096](https://arxiv.org/abs/2410.04096) |
| `fx.kan.gram`             | Orthonormal Legendre (Gram limit)    | `degree`                         | Seydi 2024 — [arXiv:2406.02583](https://arxiv.org/abs/2406.02583) |
| `fx.kan.bsrbf`            | B-spline + RBF concatenation         | `grid_size`, `rbf_grid_size`     | Ta 2024 (*BSRBF-KAN*) — [arXiv:2406.11173](https://arxiv.org/abs/2406.11173) |

### Structural-block references

| Factory                            | Description                                | Reference |
|------------------------------------|--------------------------------------------|-----------|
| `fx.kan.conv1d/2d/3d`              | KAN convolution                            | Bodner et al. 2024 — [arXiv:2406.13155](https://arxiv.org/abs/2406.13155) |
| `fx.kan.spectral_block1d/2d/3d`    | FNO spectral block + KAN channel mixer     | FNO: Li et al. 2020 — [arXiv:2010.08895](https://arxiv.org/abs/2010.08895); KAN: Liu et al. 2024 |
| `fx.kan.res_block`                 | Residual KAN block                         | ResNet pattern: He et al. 2015 — [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
| `fx.kan.attention_block`           | Transformer block with KAN feed-forward    | Yang & Wang 2024 (*KAT*) — [arXiv:2409.10594](https://arxiv.org/abs/2409.10594); attention: Vaswani et al. 2017 — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |

Every variant has a matching **single layer** exposed under `foundax.layers` (`fx.layers.KANLayer`, `fx.layers.FastKANLayer`, …), and a matching network class (`foundax.architectures.KAN`, `FastKAN`, …) for direct use.

## Choosing a variant

- **Default / general-purpose**: `fastkan` — Gaussian RBF basis is fast, well-conditioned, and almost always competitive with B-splines without the recursion cost.
- **Smooth, polynomial structure**: `chebyshev_kan`, `legendre_kan`, `gram_kan` — orthogonal-polynomial bases with stable recurrences.
- **Periodic targets**: `fourier_kan` — natural choice when the signal has wave-like structure.
- **Localised / multi-scale signals**: `wavelet_kan` (`morlet` for oscillatory features, `mexican_hat` for bumps), `relu_kan` (compact support).
- **Bounded / shape-preserving**: `bernstein_kan` (partition of unity), `rational_kan` (Padé, bounded on ℝ).
- **Original paper fidelity**: `kan` or `efficient_kan` — both implement the B-spline + SiLU-residual formulation.

## Bases

Every layer is built on top of a small basis module that maps `(..., I) → (..., I, G)` where `G` is the basis size. The KAN layer then contracts the `(I, G)` axes with an `(O, I, G)` weight tensor.

Bases are exposed for use in custom layers:

```python
from foundax.architectures import BSplineBasis, RBFBasis, FourierBasis, ChebyshevBasis, \
    JacobiBasis, LegendreBasis, WaveletBasis, TaylorBasis, HermiteBasis, LaguerreBasis, \
    BernsteinBasis, ReLUKANBasis, RationalBasis, SincBasis, GramBasis, BSRBFBasis
```

For example, you can mix and match bases inside a custom `eqx.Module`, or use them directly to study the basis functions themselves.

## Structural blocks

Beyond stacked KAN layers, foundax ships building blocks designed to drop into larger architectures.

### KAN-Convolutional layers (1D / 2D / 3D)

`KANConv1d`, `KANConv2d`, `KANConv3d` extract sliding windows from a feature map and pass each window through a KAN with the basis of your choice. Channel-last conventions match the rest of foundax:

```python
fx.kan.conv1d(in_channels=3, out_channels=8, kernel_size=3, basis="rbf", key=key)   # (W, C)
fx.kan.conv2d(in_channels=3, out_channels=8, kernel_size=3, basis="chebyshev", key=key)  # (H, W, C)
fx.kan.conv3d(in_channels=2, out_channels=4, kernel_size=3, basis="bspline", key=key)    # (D, H, W, C)
```

Any registered basis name (`bspline`, `rbf`, `fourier`, `chebyshev`, `jacobi`, `legendre`, `wavelet`, `taylor`, `hermite`, `laguerre`, `bernstein`, `relu`, `rational`, `sinc`, `gram`, `bsrbf`) can be passed.

### KAN-Spectral block (FNO-style)

`KANSpectralBlock1d/2d/3d` is a drop-in upgrade for `fx.layers.SpectralBlock*d` — the linear skip / pointwise mixer is replaced by a KAN layer:

```python
blk = fx.kan.spectral_block2d(
    in_channels=4, out_channels=8, n_modes=16,
    basis="rbf", key=key,
)
# (H, W, 4) -> (H, W, 8)
```

Use these to combine FNO's global spectral mixing with KAN's expressive channel mixing.

### Residual KAN block

`KANResBlock` chains two KAN layers with an additive skip:

```python
blk = fx.kan.res_block(features=64, basis="rbf", use_layer_norm=True, key=key)
```

`in_features == out_features` is required.

### KAN-Attention block

`KANAttentionBlock` is a pre-norm transformer block whose feed-forward sub-layer is a KAN (the basis for "Kolmogorov–Arnold Transformer" / KAT experiments):

```python
blk = fx.kan.attention_block(features=128, num_heads=4, basis="chebyshev", key=key)
# (N_tokens, 128) -> (N_tokens, 128)
```

## Pipe integration

Because every KAN layer and block sets `in_features`/`out_features` (or `in_channels`/`out_channels` for conv variants) as `eqx.field(static=True)`, the pipe operator detects channel mismatches at construction time:

```python
encoder = fx.block(fx.layers.FastKANLayer(2, 32, key=ks[0]))
hidden  = fx.block(fx.layers.ChebyshevKANLayer(32, 32, degree=4, key=ks[1]))
decoder = fx.block(fx.layers.FastKANLayer(33, 1, key=ks[2]))  # ← mismatched in_features
pipe = encoder | hidden | decoder
# foundax.pipe.ShapeMismatchError:
#   Channel mismatch: 'ChebyshevKANLayer' outputs 32 channels but 'FastKANLayer' expects 33.
```

Mixed-basis pipelines, KAN→MLP hybrids, and KAN branches inside `fx.dot` / `fx.add` / `fx.cat` combinators all work the same way as any other foundax module.

## Direct class access

For maximum control, import the underlying classes:

```python
from foundax.architectures.kan import (
    # Bases
    BSplineBasis, RBFBasis, FourierBasis, ChebyshevBasis, JacobiBasis, LegendreBasis,
    WaveletBasis, TaylorBasis, HermiteBasis, LaguerreBasis, BernsteinBasis,
    ReLUKANBasis, RationalBasis, SincBasis, GramBasis, BSRBFBasis,
    # Layers
    KANLayer, EfficientKANLayer, FastKANLayer, FourierKANLayer, ChebyshevKANLayer,
    JacobiKANLayer, LegendreKANLayer, WaveletKANLayer, TaylorKANLayer,
    HermiteKANLayer, LaguerreKANLayer, BernsteinKANLayer, ReLUKANLayer,
    RationalKANLayer, SincKANLayer, GramKANLayer, BSRBFKANLayer,
    # Networks
    KAN, EfficientKAN, FastKAN, FourierKAN, ChebyshevKAN, JacobiKAN, LegendreKAN,
    WaveletKAN, TaylorKAN, HermiteKAN, LaguerreKAN, BernsteinKAN, ReLUKAN,
    RationalKAN, SincKAN, GramKAN, BSRBFKAN,
    # Blocks
    KANConv1d, KANConv2d, KANConv3d,
    KANResBlock, KANSpectralBlock1d, KANSpectralBlock2d, KANSpectralBlock3d,
    KANAttentionBlock,
)
```

## References

The full bibliography of papers and reference implementations that inform foundax's KAN module:

### Foundational
- Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y., & Tegmark, M. (2024). **KAN: Kolmogorov-Arnold Networks**. *arXiv:2404.19756*. Code: [github.com/KindXiaoming/pykan](https://github.com/KindXiaoming/pykan).
- Blealtan (2024). **efficient-kan** — a memory- and speed-optimised reformulation of the original KAN. [github.com/Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan).

### Basis variants

- **FastKAN** — Li, Z. (2024). *Kolmogorov-Arnold Networks are Radial Basis Function Networks*. [arXiv:2405.06721](https://arxiv.org/abs/2405.06721). Code: [github.com/ZiyaoLi/fast-kan](https://github.com/ZiyaoLi/fast-kan).
- **FourierKAN** — GistNoesis (2024). *FourierKAN*. [github.com/GistNoesis/FourierKAN](https://github.com/GistNoesis/FourierKAN).
- **ChebyshevKAN** — SS, S. S. (2024). *Chebyshev Polynomial-Based Kolmogorov-Arnold Networks: An Efficient Architecture for Nonlinear Function Approximation*. [arXiv:2405.07200](https://arxiv.org/abs/2405.07200). Code: [github.com/SynodicMonth/ChebyKAN](https://github.com/SynodicMonth/ChebyKAN).
- **JacobiKAN / fKAN** — Aghaei, A. A. (2024). *fKAN: Fractional Kolmogorov-Arnold Networks with trainable Jacobi basis functions*. [arXiv:2406.07456](https://arxiv.org/abs/2406.07456). Code: [github.com/alirezaafzalaghaei/fKAN](https://github.com/alirezaafzalaghaei/fKAN).
- **LegendreKAN, HermiteKAN, LaguerreKAN, BernsteinKAN, GramKAN** — Seydi, S. T. (2024). *Exploring the Potential of Polynomial Basis Functions in Kolmogorov-Arnold Networks: A Comparative Study of Different Groups of Polynomials*. [arXiv:2406.02583](https://arxiv.org/abs/2406.02583). Code: [github.com/seydi1370/Basis_Functions](https://github.com/seydi1370/Basis_Functions).
- **WaveletKAN / Wav-KAN** — Bozorgasl, Z. & Chen, H. (2024). *Wav-KAN: Wavelet Kolmogorov-Arnold Networks*. [arXiv:2405.12832](https://arxiv.org/abs/2405.12832). Code: [github.com/zavareh1/Wav-KAN](https://github.com/zavareh1/Wav-KAN).
- **TaylorKAN** — Muyuzhierchengse (2024). *TaylorKAN*. [github.com/Muyuzhierchengse/TaylorKAN](https://github.com/Muyuzhierchengse/TaylorKAN).
- **ReLU-KAN / FasterKAN** — Qiu, Q., Zhu, T., Gong, H., Chen, L., & Ning, H. (2024). *ReLU-KAN: New Kolmogorov-Arnold Networks that Only Need Matrix Addition, Dot Multiplication, and ReLU*. [arXiv:2406.02075](https://arxiv.org/abs/2406.02075). Related: Delis, A. (2024). *FasterKAN*. [github.com/AthanasiosDelis/faster-kan](https://github.com/AthanasiosDelis/faster-kan).
- **RationalKAN / rKAN** — Aghaei, A. A. (2024). *rKAN: Rational Kolmogorov-Arnold Networks*. [arXiv:2406.14495](https://arxiv.org/abs/2406.14495). Code: [github.com/alirezaafzalaghaei/rKAN](https://github.com/alirezaafzalaghaei/rKAN).
- **SincKAN** — Yu, R., Yu, W., & Wang, X. (2024). *SincKAN: Function Approximation with Sinc Interpolation Inside Kolmogorov-Arnold Networks*. [arXiv:2410.04096](https://arxiv.org/abs/2410.04096).
- **BSRBF-KAN** — Ta, H. T. (2024). *BSRBF-KAN: A combination of B-splines and Radial Basis Functions in Kolmogorov-Arnold Networks*. [arXiv:2406.11173](https://arxiv.org/abs/2406.11173). Code: [github.com/hoangthangta/BSRBF_KAN](https://github.com/hoangthangta/BSRBF_KAN).

### Structural blocks

- **Convolutional KAN** — Bodner, A. D., Tepsich, A. S., Spolski, J. N., & Pourteau, S. (2024). *Convolutional Kolmogorov-Arnold Networks*. [arXiv:2406.13155](https://arxiv.org/abs/2406.13155). Code: [github.com/AntonioTepsich/Convolutional-KANs](https://github.com/AntonioTepsich/Convolutional-KANs).
- **Kolmogorov-Arnold Transformer (KAT)** — Yang, X. & Wang, X. (2024). *Kolmogorov-Arnold Transformer*. [arXiv:2409.10594](https://arxiv.org/abs/2409.10594). Code: [github.com/Adamdad/kat](https://github.com/Adamdad/kat).
- **Fourier Neural Operator (FNO)** — Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020). *Fourier Neural Operator for Parametric Partial Differential Equations*. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895). Foundation for `KANSpectralBlock*d`.
- **ResNet** — He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385). Foundation for `KANResBlock`.
- **Transformer / Attention** — Vaswani, A. et al. (2017). *Attention Is All You Need*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762). Foundation for `KANAttentionBlock`.

### Theoretical background

- Kolmogorov, A. N. (1957). *On the representation of continuous functions of several variables by superpositions of continuous functions of a smaller number of variables*. Dokl. Akad. Nauk SSSR 108, 179–182.
- Arnold, V. I. (1959). *On the representation of continuous functions of three variables by superpositions of continuous functions of two variables*. Mat. Sb. (N.S.) 48(90), 3–74.
