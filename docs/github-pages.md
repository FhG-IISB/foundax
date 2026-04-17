# GitHub Pages Hosting

This repository publishes docs from `docs/` using MkDocs and GitHub Actions.

## Local preview

```bash
uvx --from mkdocs mkdocs serve
```

Open `http://127.0.0.1:8000` to preview the site.

## Build locally

```bash
uvx --from mkdocs mkdocs build
```

This writes static output to `site/`.

## Automatic deploy

A workflow is included at `.github/workflows/docs.yml`.
It builds with MkDocs and deploys to GitHub Pages on pushes to `main` when docs files or MkDocs config change.

## GitHub settings

In the repository settings:

1. Open **Settings -> Pages**.
2. Set **Source** to **GitHub Actions**.
3. Push to `main` to trigger deployment.

After the first successful deployment, GitHub Pages provides the site URL in the Pages settings panel.
