# foundax

`foundax` is a JAX model repository that combines:

- **direct Equinox implementations** of neural-operator and PDE-learning architectures
- **Equinox wrappers** for larger vendored model families such as Poseidon, MORPH, MPP, Walrus, BCAT, PDEformer-2, DPOT, and PROSE

This site is written as repository documentation rather than only API notes. It explains package layout, model families, and practical Equinox usage.

## Scope

The docs here cover:

- the Equinox model factories exposed from `foundax`
- the repository’s core model categories
- the foundation-model wrappers and their upstream references
- practical examples for both core and foundation models
- GitHub Pages hosting for this repository

The docs intentionally do **not** focus on the Flax-specific foundation-model paths.

## Quick start

```python
import foundax as fx

# Core models
mlp = fx.mlp(in_features=2, output_dim=1, hidden_dims=64, num_layers=3)
fno = fx.fno2d(in_features=1, hidden_channels=32, n_modes=16)
unet = fx.unet2d(in_channels=1, out_channels=1)

# Foundation-model namespaces
poseidon = fx.poseidon.T()
morph = fx.morph.Ti()
mpp = fx.mpp.Ti(n_states=12)
```

## Model layout

### Core Equinox architectures

These live in `foundax/architectures/` and are exposed through `foundax.nn`:

- MLP and linear models
- Fourier Neural Operators
- UNets
- DeepONet
- CNO and MgNO
- geometry-aware and point-based models
- GNOT-family transformer operators

### Equinox foundation-model wrappers

These live behind namespace modules such as:

- `foundax.poseidon`
- `foundax.morph`
- `foundax.mpp`
- `foundax.walrus`
- `foundax.bcat`
- `foundax.pdeformer2`
- `foundax.dpot`
- `foundax.prose`

The wrapper modules are thin public entry points over vendored JAX implementations in `repos/`.

## Where to read next

- **Getting Started** for installation and package structure
- **Core Models** for the direct Equinox architectures
- **Foundation Models** for the larger wrapper families, variants, and upstream links
- **Model Examples** for concise end-to-end constructor and forward-call snippets
