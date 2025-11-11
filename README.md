# 行间 Interlines

A multi-agent generative system for **public knowledge translation** with a **historical lens**.
This repo uses **uv** for dependency & workflow management, and ships **Docker** & **Compose** for deployment.

## Dev Quickstart

```bash
uv venv && uv sync --all-groups
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
```
