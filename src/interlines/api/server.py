"""Uvicorn runner for the InterLines FastAPI application.

This module provides a small, typed `run()` function used by the console script
`interlines-api` (declared in pyproject.toml). It bootstraps the ASGI app from
:mod:`interlines.api.app`, selects sensible defaults for host/port, and configures
logging based on the typed settings defined in :mod:`interlines.core.settings`.

Design goals
------------
1) **No side effects on import** — the server starts only when `run()` is called.
2) **12-factor friendly** — host/port can be configured via environment variables:
   - `API_HOST` (default: `"127.0.0.1"`)
   - `API_PORT` (default: `8000`)
   - `API_RELOAD` (default: `True` in dev, `False` otherwise)
3) **Consistent logging** — uvicorn log level follows our `LOG_LEVEL` setting.

Examples
--------
Run with defaults:
    $ uv run interlines-api

Change port (Windows/Unix):
    $ API_PORT=8080 uv run interlines-api

Enable auto-reload explicitly:
    $ API_RELOAD=true uv run interlines-api
"""

from __future__ import annotations

import os

import uvicorn

from interlines.core.settings import get_logger, load_settings

from .app import get_app


def _bool_from_env(name: str, default: bool) -> bool:
    """Parse a boolean-like environment variable.

    Accepts common truthy/falsey forms (case-insensitive):
    - True: "1", "true", "yes", "on"
    - False: "0", "false", "no", "off"

    Parameters
    ----------
    name : str
        Environment variable name to read.
    default : bool
        Fallback value when variable is absent or unparsable.

    Returns
    -------
    bool
        Parsed boolean or the provided default.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return default


def run(
    host: str | None = None,
    port: int | None = None,
    reload: bool | None = None,
) -> None:
    """Start the InterLines API server using uvicorn.

    Parameters
    ----------
    host : Optional[str], default None
        Bind address. If None, falls back to `API_HOST` or `"127.0.0.1"`.
    port : Optional[int], default None
        Bind port. If None, falls back to `API_PORT` or `8000`.
    reload : Optional[bool], default None
        Uvicorn's auto-reload. If None, uses `API_RELOAD` or `settings.is_dev`.

    Notes
    -----
    - We refresh settings via `load_settings()` at call time so that changes to env
      made by tests or orchestration are honored.
    - We avoid global app instances here and build the app via `get_app()` to keep
      future test isolation straightforward.
    """
    settings = load_settings()
    log = get_logger("interlines.api.server")

    # --- Resolve host (ensure concrete str for mypy) -------------------------
    # Use os.environ.get + "or" to produce a concrete str, then annotate.
    env_host = os.environ.get("API_HOST")  # Optional[str]
    bind_host: str = host if host is not None else (env_host or "127.0.0.1")

    # --- Resolve port (ensure concrete int for mypy) -------------------------
    if port is not None:
        bind_port: int = port
    else:
        env_port = os.environ.get("API_PORT")  # Optional[str]
        bind_port = int(env_port) if (env_port is not None and env_port.isdigit()) else 8000

    # --- Resolve reload (ensure concrete bool for mypy) ----------------------
    if reload is not None:
        use_reload: bool = reload
    else:
        use_reload = _bool_from_env("API_RELOAD", default=settings.is_dev)

    log.info(
        "Starting InterLines API (host=%s, port=%s, reload=%s, level=%s)",
        bind_host,
        bind_port,
        use_reload,
        settings.log_level,
    )

    uvicorn.run(
        app=get_app(),
        host=bind_host,
        port=bind_port,
        reload=use_reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    # Allow `python -m interlines.api.server` during local dev.
    run()
