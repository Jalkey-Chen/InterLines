"""
FastAPI Application Factory & Configuration.

Milestone
---------
M6 | Interface & Deployment
Step 6.1 | API interpret/brief/health

Updates in Commit 2:
- Initialized the global JobStore in the lifespan event.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from interlines.api.job_store import JobStore

# Note: We will implement the routers in subsequent commits.
# from interlines.api.routers import health, interpret


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    ASGI Lifespan context manager.

    Handles startup and shutdown logic.
    - **Startup**: Initialize the in-memory job store singleton.
    - **Shutdown**: Clean up resources (if any).
    """
    print("[InterLines API] Starting up...")

    # Initialize the global JobStore instance
    JobStore.get_instance()
    print("[InterLines API] JobStore initialized.")

    yield

    print("[InterLines API] Shutting down...")


def create_app() -> FastAPI:
    """
    Construct and configure the InterLines FastAPI application.

    Returns
    -------
    FastAPI
        The configured ASGI application ready to be served by Uvicorn.
    """
    app = FastAPI(
        title="InterLines API",
        description="Public Translation Interface (Expert -> Citizen)",
        version="0.6.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # -----------------------------------------------------------------------
    # Middleware
    # -----------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # Global Exception Handlers
    # -----------------------------------------------------------------------
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": str(exc),
                "path": request.url.path,
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Bad Request",
                "detail": str(exc),
            },
        )

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------
    @app.get("/health", tags=["System"])
    async def health_check() -> dict[str, str]:
        """Simple liveness probe."""
        return {"status": "ok", "version": "0.6.0"}

    return app


__all__ = ["create_app"]
