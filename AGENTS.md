# AGENTS.md

This file contains instructions for all Copilot agents working on this repository.

## Code Quality Before Every Commit

Before making any commit, always ensure the following checks pass:

1. **Tests** — all tests must pass:
   ```bash
   uv run pytest tests -v
   ```

2. **Type checks** — run `ty` against the source:
   ```bash
   uv run ty check fourdvarjax
   ```

3. **Lint** — ruff lint must report no errors:
   ```bash
   uv run ruff check .
   ```

4. **Format** — ruff format must produce no diffs:
   ```bash
   uv run ruff format --check .
   ```

If any check fails, fix the issue before committing.

## Conventional Commits

All PR titles and commit messages **must** follow the
[Conventional Commits](https://www.conventionalcommits.org/) specification.

The PR title and every commit message must start with one of these types:

- `feat:` — a new feature
- `fix:` — a bug fix
- `docs:` — documentation-only changes
- `style:` — formatting, missing semi-colons, etc. (no production code change)
- `refactor:` — code refactoring
- `perf:` — performance improvements
- `test:` — adding or updating tests
- `build:` — build-system or dependency changes
- `ci:` — CI/CD configuration changes
- `chore:` — maintenance tasks, updating lockfiles, etc.
- `revert:` — reverts a previous commit

Examples:
```
feat: add fixed-point solver and variational cost utilities
fix: correct time dimension selection in obs_interpolation_init
docs: update README with new API examples
test: add tests for IdentityPrior and decomposed_loss
```

The subject (text after the `:`) must start with a **lowercase letter**.

## PR Title and Description

- The PR title must follow Conventional Commits (see above).
- The PR description should **never be overwritten** from one session to the
  next.  Only **append** new information incrementally at the bottom of the
  description when additional changes are made.
