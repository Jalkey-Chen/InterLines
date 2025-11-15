# InterLines

> Public Knowledge Interpreter — turn complex papers/policies into public briefs.

## Dev Quickstart

```bash
uv venv && uv sync --all-groups
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
````

## CLI (Step 0.4)

The `interlines` CLI exposes three subcommands:

* `version` — print the installed version
* `env` — show effective configuration; add `--json` for machine-readable output
* `doctor` — run environment checks; exits non-zero if something is missing

```bash
# Show version
uv run interlines version
#> InterLines 0.0.1

# Show environment (human-readable)
uv run interlines env
#> environment = dev
#> log_level   = INFO
#> openai_key  = <unset>

# Show environment (JSON)
uv run interlines env --json
#> {"environment":"dev","log_level":"INFO","openai_key":null}

# Diagnose local setup (Python/lib/env)
uv run interlines doctor
```

### Config reference

InterLines uses typed settings via Pydantic:

* `INTERLINES_ENV` — `dev` (default) | `test` | `prod`
* `LOG_LEVEL` — `DEBUG` | `INFO` (default) | `WARNING` | `ERROR` | `CRITICAL`
* `OPENAI_API_KEY` — optional (for later LLM steps)

`.env` loading order (highest precedence first): real env vars → `.env.local`, `.env.dev`,
`.env.test`, `.env.prod` → `.env`.

**In code**

```python
from interlines.core.settings import settings, get_logger

env = settings.environment
log = get_logger()
log.info("InterLines booted with %s", env)
```

## Pre-commit

```bash
uv tool run pre-commit install
# Then just `git commit` — hooks (ruff, mypy) run only on changed files.
```