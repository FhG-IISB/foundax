# Changelog

All notable changes to foundax are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `CITATION.cff`, `CONTRIBUTING.md`, `SECURITY.md`, and `CODE_OF_CONDUCT.md`.
- `docs/architectures.md` — single-page overview of every supported architecture with paper references, split into Core / KAN / Foundation tables.
- README badge row, status note, citation block, and a condensed architectures summary linking to the full docs page.
- `.pre-commit-config.yaml` (ruff + standard hygiene hooks) and `foundax/py.typed` marker for PEP 561 typing.
- GitHub issue templates (`bug_report`, `feature_request`) and a pull-request template with a pre-merge checklist.
- `pixi run fmt` and `pixi run all` tasks; format-check step in `.github/workflows/ci.yml`.

### Changed
- `docs/core-models.md` summary table now includes a Reference column linking to the canonical paper for each family.
- `pyproject.toml` adds `authors`, `classifiers`, `[project.urls]`, and a `py.typed` package-data entry.

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

[Unreleased]: https://github.com/FhG-IISB/foundax/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/FhG-IISB/foundax/releases/tag/v0.2.0
