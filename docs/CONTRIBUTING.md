# Contributing (Step 0.1)

## Local setup
```bash
uv venv && uv sync --all-groups
uv run ruff check . && uv run ruff format --check .
uv run mypy src && uv run pytest -q
```
## Pre-commit
```bash
uv tool run pre-commit install
uv tool run pre-commit run --all-files
```
## Branch & commits

Branch from `dev` with prefixes: `feature/*`, `chore/*`, `fix/*`

Conventional Commits (e.g., `chore: add uv-based CI, ruff, mypy, pytest`)