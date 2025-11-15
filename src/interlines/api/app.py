"""Minimal FastAPI application for InterLines (Step 0.3).

This module exposes a factory `get_app()` that builds a FastAPI instance with:
- Basic project metadata (title/version)
- A health-check endpoint at `GET /health` returning status and environment

Design notes
------------
- We import `settings` and `get_logger` from the core settings module so the app
  reflects the current environment and uses a consistent logger format/level.
- Keeping a factory function allows future tests to create isolated app instances
  without side effects (useful once we add middlewares and dependency wiring).

Example
-------
>>> from interlines.api.app import get_app
>>> app = get_app()
"""

from __future__ import annotations

from fastapi import FastAPI

from interlines import __version__
from interlines.core.settings import get_logger, settings


def get_app() -> FastAPI:
    """Create and return the InterLines FastAPI application.

    Returns
    -------
    FastAPI
        An app instance configured with metadata and a simple `/health` route.
    """
    log = get_logger("interlines.api")

    app = FastAPI(
        title="InterLines API",
        version=__version__,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Lightweight health probe.

        Returns
        -------
        dict[str, str]
            JSON payload with a constant OK status, current environment, and version.
        """
        return {
            "status": "ok",
            "environment": settings.environment,
            "version": __version__,
        }

    log.info("FastAPI app created (env=%s, version=%s)", settings.environment, __version__)
    return app


# Optional: convenience module-level app for ASGI servers (e.g., uvicorn module path)
app: FastAPI = get_app()
