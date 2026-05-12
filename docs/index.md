# foundax Documentation

Build and reuse neural-operator models in JAX with an Equinox-first API surface, from lightweight core architectures to large foundation-model wrappers.

<div class="hero-actions" markdown>
[Getting Started](getting-started.md){ .md-button .md-button--primary }
[Core Models](core-models.md){ .md-button }
[Foundation Models](equinox-architectures.md){ .md-button }
[Model Examples](model-examples.md){ .md-button }
</div>

<div class="grid cards" markdown>

- __New to foundax?__

	Start with installation, package layout, and usage conventions.

	[Open getting started](getting-started.md)

- __Need architecture guidance?__

	Browse the direct Equinox model families and where they fit.

	[Open core models](core-models.md)

- __Using large wrappers?__

	See foundation-model namespaces, variants, and upstream references.

	[Open foundation models](equinox-architectures.md)

</div>

## Start Here

If you are new to foundax, follow this path:

1. [Getting Started](getting-started.md): install and understand the package surface.
2. [Core Models](core-models.md): choose a direct Equinox architecture baseline.
3. [Foundation Models](equinox-architectures.md): choose a large wrapper family and variant.
4. [Model Examples](model-examples.md): copy minimal runnable constructor and forward-pass snippets.

## Model Surface

| Guide | Focus |
|---|---|
| [Core Models](core-models.md) | Direct Equinox architectures in `foundax/architectures` and exposed via `foundax.nn` |
| [Foundation Models](equinox-architectures.md) | Namespace wrappers for Poseidon, MORPH, MPP, Walrus, BCAT, PDEformer-2, DPOT, and PROSE |
| [Model Examples](model-examples.md) | Minimal end-to-end examples for both core and foundation-model constructors |
| [GitHub Pages](github-pages.md) | Local preview, build, and deployment setup |

## Quick Usage

```python
import foundax as fx

# Core architectures
mlp = fx.mlp(in_features=2, output_dim=1, hidden_dims=64, num_layers=3)
fno = fx.fno2d(in_features=1, hidden_channels=32, n_modes=16)

# Foundation wrappers (preferred namespace style)
poseidon = fx.poseidon.T()
morph = fx.morph.S()
```

## Integration With jNO

```python
import foundax as fx
import jno
import optax

net = jno.nn.wrap(fx.fno2d(in_features=1, hidden_channels=32, n_modes=16))
net.optimizer(optax.adam, lr=1e-3)
```

!!! warning
		The documentation intentionally focuses on the Equinox-facing model surface.
		Flax-specific paths from vendored repositories are not covered here.
