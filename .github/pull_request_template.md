## Summary

<!-- One or two sentences: what does this PR change, and why? -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation only
- [ ] Test / tooling / CI

## Linked issues / PRs

<!-- e.g. Closes #123, Related to #456 -->

## Pre-merge checklist

- [ ] `pixi run ci-fmt` passes
- [ ] `pixi run ci-lint` passes
- [ ] `pixi run ci-test` passes
- [ ] New / changed public APIs are documented (docstrings + relevant `docs/` page)
- [ ] `docs/architectures.md` is updated if a new model family was added
- [ ] `CHANGELOG.md` entry added under `[Unreleased]`
- [ ] No debug prints, commented-out code, or unused imports left behind
- [ ] Pretrained-weight verification (`pixi run -e dev verify-<model>`) re-run if a foundation-model wrapper changed

## Testing notes

<!-- How did you verify this works? Which tests cover it? -->

## Reviewer focus (optional)

<!-- Anything you specifically want reviewers to look at? -->
