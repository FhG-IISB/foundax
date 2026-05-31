# Contributing

Thanks for your interest in foundax — PRs and issues are welcome.

## Setup

foundax uses [pixi](https://pixi.sh):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
git clone https://github.com/FhG-IISB/foundax.git
cd foundax
pixi install
```

Optionally enable pre-commit hooks (ruff + a few hygiene checks):

```bash
pixi run pre-commit install
```

## Before you push

Run the three checks the CI runs:

```bash
pixi run fmt && pixi run lint && pixi run test
# or, equivalently:
pixi run all
```

## Pixi tasks

| Task | What it does |
|---|---|
| `pixi run fmt` | `ruff format .` |
| `pixi run lint` | `ruff check . --fix` |
| `pixi run test` | Fast test suite |
| `pixi run all` | format + lint + test |
| `pixi run ci-fmt` / `ci-lint` / `ci-test` | Read-only CI variants |
| `pixi run -e dev test-train` | Tests that actually train (needs `dev` env) |
| `pixi run -e dev verify-<model>` | Verify a foundation-model wrapper (poseidon, walrus, morph, mpp, bcat, dpot, pdeformer2, prose) |

## Notes on the vendored foundation models

The eight foundation-model wrappers re-export Equinox modules from packages vendored under `repos/jax_*`. When changing a wrapper, only edit `foundax/<name>.py` and re-run `pixi run -e dev verify-<name>`. Leave `repos/jax_<name>/` untouched unless you're syncing from upstream.
