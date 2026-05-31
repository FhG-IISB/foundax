# Foundation Models

This page documents the **Equinox foundation-model wrappers** exposed by `foundax`.

These wrappers are the large-model side of the repository. Each namespace provides a small public API over a vendored JAX implementation under `repos/`, while keeping the package-level import surface consistent.

## Summary

For the full architectures matrix (including core models and KAN variants), see [Architectures Overview](architectures.md).

| Namespace | Variants | Backbone | Reference | Weights license |
|---|---|---|---|---|
| `fx.poseidon` | T, B, L | ScOT (Swin-style hierarchical operator transformer) | Herde et al. 2024 — [arXiv:2405.19101](https://arxiv.org/abs/2405.19101) | CC-BY-NC-4.0 |
| `fx.morph` | Ti, S, M, L | ViT3D regression | Rautela et al. 2025 — [arXiv:2509.21670](https://arxiv.org/abs/2509.21670) | MIT |
| `fx.mpp` | Ti, S, B, L | AViT (axial vision transformer) | McCabe et al., NeurIPS 2024 — [openreview/DKSI3bULiZ](https://openreview.net/forum?id=DKSI3bULiZ) | MIT |
| `fx.walrus` | base | Isotropic encoder–processor–decoder (1.29B params) | Bodner et al. 2024 — [arXiv:2511.15684](https://arxiv.org/abs/2511.15684) | MIT |
| `fx.bcat` | base | Block-causal transformer | Liu et al. 2025 — [arXiv:2501.18972](https://arxiv.org/abs/2501.18972) | MIT |
| `fx.pdeformer2` | small, base, fast | Graphormer encoder + INR decoder | Shi et al. 2025 — [arXiv:2502.14844](https://arxiv.org/abs/2502.14844) | Apache-2.0 |
| `fx.dpot` | Ti, S, M, L, H | DPOTNet (AFNO-style mixing) | Hao et al., ICML 2024 — [arXiv:2403.03542](https://arxiv.org/abs/2403.03542) | Apache-2.0 |
| `fx.prose` | fd_1to1, fd_2to1, ode_2to1, pde_2to1 | Transformer sequence-to-sequence | Sun et al. 2024 — [arXiv:2404.12355](https://arxiv.org/abs/2404.12355) | MIT |

Weights remain subject to their upstream licenses — see [THIRD_PARTY_LICENSES](https://github.com/FhG-IISB/foundax/blob/main/THIRD_PARTY_LICENSES).

## API pattern

The preferred entry points are the namespace modules:

```python
import foundax as fx

poseidon = fx.poseidon.T()
morph = fx.morph.S()
mpp = fx.mpp.B(n_states=12)
walrus = fx.walrus.base()
```

The package also exposes top-level aliases such as `fx.poseidonT()` or `fx.prose_fd_1to1()`. Those are convenience shortcuts, but the namespace style is what this documentation uses.

## Poseidon

Namespace: `fx.poseidon`

Available variants:

- `T`
- `B`
- `L`

Architecture summary:

- Poseidon wraps **ScOT**, a Swin-style hierarchical operator transformer
- the encoder-decoder structure uses multiscale windowed attention and skip connections
- the implementation is designed for PDE operator learning on image-like grids

What to use it for:

- 2D grid-based PDE surrogates
- large operator models where multiscale transformer processing is preferable to a smaller FNO or UNet baseline

Repository mapping:

- Public wrapper: `foundax/poseidon.py`
- Vendored implementation: `repos/jax_poseidon`

References:

- Paper: https://arxiv.org/abs/2405.19101
- Original repository: https://github.com/camlab-ethz/poseidon

## MORPH

Namespace: `fx.morph`

Available variants:

- `Ti`
- `S`
- `M`
- `L`

Architecture summary:

- MORPH uses a **ViT3D-style regression architecture** over PDE data
- the model is intended for arbitrary modality and multi-physics settings
- the wrapper constructs the Equinox `ViT3DRegression` path from the vendored package

What to use it for:

- volumetric or time-aware PDE data
- settings where a large transformer over patchified 3D structure is a better fit than 2D operator models

Repository mapping:

- Public wrapper: `foundax/morph.py`
- Vendored implementation: `repos/jax_morph`

References:

- Paper: https://arxiv.org/abs/2509.21670
- Original repository: https://github.com/lanl/MORPH

## MPP

Namespace: `fx.mpp`

Available variants:

- `Ti`
- `S`
- `B`
- `L`

Architecture summary:

- MPP wraps **AViT**, an axial vision transformer used for multiple-physics pretraining
- the model mixes temporal and spatial attention blocks over structured simulation data
- it is built for multi-physics surrogate learning rather than a single PDE family

What to use it for:

- structured spatio-temporal simulation data
- experiments that need a large transformer-based surrogate model with pretraining-oriented design

Repository mapping:

- Public wrapper: `foundax/mpp.py`
- Vendored implementation: `repos/jax_mpp`

References:

- Paper: https://openreview.net/forum?id=DKSI3bULiZ
- Original repository: https://github.com/PolymathicAI/multiple_physics_pretraining

## Walrus

Namespace: `fx.walrus`

Available variants:

- `base`

Architecture summary:

- Walrus uses an **isotropic encoder-processor-decoder** layout with large space-time attention blocks
- the wrapped model is designed for large-scale PDE or atmosphere-style state evolution tasks
- the implementation includes structured handling of variable dimensionality and boundary conditions

What to use it for:

- large spatio-temporal surrogate models
- experiments where the model scale and processor depth are more important than using a smaller baseline

Repository mapping:

- Public wrapper: `foundax/walrus.py`
- Vendored implementation: `repos/jax_walrus`

References:

- Original Walrus repository: https://github.com/nubskr/walrus
- Aurora paper cited in the vendored README: https://arxiv.org/abs/2405.13063

## BCAT

Namespace: `fx.bcat`

Available variants:

- `base`

Architecture summary:

- BCAT is a **block-causal transformer** for autoregressive spatio-temporal PDE prediction
- the model uses patched inputs, causal structure in time, and transformer blocks designed for rollout-style forecasting

What to use it for:

- temporal forecasting on regular grids
- fluid or PDE sequence modeling where causal rollout is central

Repository mapping:

- Public wrapper: `foundax/bcat.py`
- Vendored implementation: `repos/jax_bcat`

References:

- Paper: https://arxiv.org/abs/2501.18972

## PDEformer-2

Namespace: `fx.pdeformer2`

Available variants:

- `small`
- `base`
- `fast`

Architecture summary:

- PDEformer-2 combines a **Graphormer encoder** with an **implicit neural representation** decoder driven by a hypernetwork
- the model represents PDE problems as graph-structured inputs and predicts solution values at query coordinates
- this is the most graph-centric model family in the repository

What to use it for:

- PDE problems described through graph or DAG structure
- query-based solution evaluation rather than only dense image-to-image prediction

Repository mapping:

- Public wrapper: `foundax/pdeformer2.py`
- Vendored implementation: `repos/jax_pdeformer2`

References:

- Paper: https://arxiv.org/abs/2502.14844
- Original repository: https://github.com/functoreality/pdeformer-2

## DPOT

Namespace: `fx.dpot`

Available variants:

- `Ti`
- `S`
- `M`
- `L`
- `H`

Architecture summary:

- DPOT wraps **DPOTNet**, a transformer-style operator model with AFNO or Fourier-style mixing
- the model is built for autoregressive PDE pretraining and large-scale surrogate modeling on regular grids

What to use it for:

- large regular-grid PDE sequence modeling
- experiments where AFNO-style token mixing is a better fit than a pure CNN or standard transformer baseline

Repository mapping:

- Public wrapper: `foundax/dpot.py`
- Vendored implementation: `repos/jax_dpot`

References:

- Paper: https://arxiv.org/abs/2403.03542
- Original repository: https://github.com/hzk17/DPOT

## PROSE

Namespace: `fx.prose`

Available variants:

- `fd_1to1`
- `fd_2to1`
- `ode_2to1`
- `pde_2to1`

Architecture summary:

- PROSE is a **transformer-based sequence model family** for finite-difference, ODE, and PDE tasks
- unlike the other wrappers, this family spans multiple task formulations rather than just scaled variants of one architecture
- the Equinox implementations in the vendored package cover both pure data-to-data and text/data fusion settings

What to use it for:

- sequence-to-sequence operator learning
- problems with symbolic or text-conditioned inputs
- ODE and PDE tasks that are more naturally expressed as token or sequence prediction than as a single image-to-image mapping

Repository mapping:

- Public wrapper: `foundax/prose.py`
- Vendored implementation: `repos/jax_prose`

References:

- Paper: Sun et al. 2024 — [arXiv:2404.12355](https://arxiv.org/abs/2404.12355)
- Vendored JAX repository: `repos/jax_prose`

## Notes On Package Surface

The top-level `foundax.nn` module also exposes convenience aliases for these wrappers, for example:

- `poseidonT`, `poseidonB`, `poseidonL`
- `morph_Ti`, `morph_S`, `morph_M`, `morph_L`
- `mpp_Ti`, `mpp_S`, `mpp_B`, `mpp_L`
- `pdeformer2_small`, `pdeformer2_base`, `pdeformer2_fast`
- `dpot_Ti`, `dpot_S`, `dpot_M`, `dpot_L`, `dpot_H`
- `prose_fd_1to1`, `prose_fd_2to1`, `prose_ode_2to1`, `prose_pde_2to1`

Those aliases simply forward to the namespace modules documented above.
