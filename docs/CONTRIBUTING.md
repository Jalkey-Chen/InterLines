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

## Result pattern

InterLines uses a tiny, typed `Result[T, E]` (`src/interlines/core/result.py`) to make
success/failure explicit in planner/agent code paths.

**Do**

- Return `Result` from functions that can fail in normal operation.
- Use `map` / `flat_map` for composition instead of `try/except` in inner loops.
- Convert to exceptions only at boundaries (CLI, API handlers) when appropriate.

**Avoid**

- Throwing exceptions for expected control‐flow.
- Returning `None` for failure — use `Err("reason")` instead.

## Testing & typing

- Tests run under `pytest -q` and `mypy --strict` via pre-commit and CI.
- Do not import `pytest` types in annotations; use `typing.Any` for fixtures if needed.
- Keep modules dependency-light; prefer standard library utilities.