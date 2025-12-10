"""
FastAPI Application Factory & Configuration.

Milestone
---------
M6 | Interface & Deployment
Step 6.1 | API interpret/brief/health

This module initializes the FastAPI application instance. It is responsible for:
1.  **Middleware Setup**: CORS (Cross-Origin Resource Sharing) for frontend access.
2.  **Exception Handling**: Global handlers to ensure all errors return structured JSON.
3.  **Routing**: Mounting the API routers (jobs, health).
4.  **Lifecycle**: Managing startup/shutdown events (e.g., initializing the job store).

Design Pattern
--------------
We use an **Application Factory** pattern (`create_app`). This allows for:
-   Easy testing (spinning up separate app instances per test).
-   Configuration injection (passing distinct settings for Dev/Prod).

Updates in Commit 3:
- Registered `interpret.router` to expose /interpret and /jobs endpoints.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from interlines.api.job_store import JobStore

# NEW: Import the router module
from interlines.api.routers import interpret


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    ASGI Lifespan context manager.

    Handles startup and shutdown logic.
    - **Startup**: Initialize the in-memory job store singleton.
    - **Shutdown**: Clean up resources (if any).
    """
    print("[InterLines API] Starting up...")

    # Initialize the global JobStore instance so it's ready before any requests.
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
    # Enable CORS to allow requests from any frontend (React, Vue, cURL, etc.)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, restrict this to specific domains.
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # Global Exception Handlers
    # -----------------------------------------------------------------------
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Catch-all handler to ensure unhandled exceptions return structured JSON.

        Instead of a generic 500 HTML page, we return:
        {
            "error": "Internal Server Error",
            "detail": "..." (str(exc))
        }
        """
        # In a real system, we would log the full traceback here.
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
        """Map Python ValueErrors to HTTP 400 Bad Request."""
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
    # Register the Interpretation Router (POST /interpret, GET /jobs/{id})
    app.include_router(interpret.router)

    # Simple health check for immediate liveness probing
    @app.get("/health", tags=["System"])
    async def health_check() -> dict[str, str]:
        """Simple liveness probe."""
        return {"status": "ok", "version": "0.6.0"}

    return app


__all__ = ["create_app"]
