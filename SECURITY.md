# Security

If you spot something that looks like a vulnerability, please don't open a public issue — email [leon.armbruster@iisb.fraunhofer.de](mailto:leon.armbruster@iisb.fraunhofer.de) (subject prefix `[foundax security]`) or use GitHub's [private vulnerability reporting](https://github.com/FhG-IISB/foundax/security/advisories/new).

A short note on what to keep in mind when using foundax:

- Equinox checkpoints and Hugging Face weights load via pickle and execute arbitrary Python. Only load files from sources you trust.
- `foundax-convert` calls `torch.load` on PyTorch checkpoints — same caveat.
- Vendored code under `repos/jax_*` is third-party. For bugs there, please report upstream first.
