"""Smoke tests for the InterLines FastAPI application.

This test module verifies the minimal API contract established in Step 0.3:
- The application factory `get_app()` builds a FastAPI instance without side effects.
- `GET /health` responds with HTTP 200 and a JSON payload that includes:
    * "status": constant string "ok"
    * "environment": one of {"dev", "test", "prod"}
    * "version": matches `interlines.__version__`

Notes
-----
- We avoid importing `pytest` types to keep mypy strict-friendly without extra stubs.
- `fastapi.testclient` depends on `requests`; ensure it is present in dev/test deps.
"""

from __future__ import annotations

from typing import Final

from fastapi.testclient import TestClient

from interlines import __version__ as PKG_VERSION
from interlines.api.app import get_app

# Allowed environment labels exposed by the /health endpoint
ALLOWED_ENVS: Final[set[str]] = {"dev", "test", "prod"}


def test_health_endpoint_contract() -> None:
    """`GET /health` returns a stable shape and expected values.

    Steps
    -----
    1) Build an app instance via `get_app()` to avoid cross-test side effects.
    2) Issue a GET to `/health`.
    3) Assert HTTP 200 and validate the JSON fields and their values.
    """
    app = get_app()
    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 200, "Health endpoint should return HTTP 200"

    data = resp.json()
    # Shape checks
    assert isinstance(data, dict), "Health payload must be a JSON object"
    assert {"status", "environment", "version"}.issubset(
        data.keys()
    ), "Health payload missing required keys"

    # Value checks
    assert data["status"] == "ok", "Health status should be constant 'ok'"
    assert data["environment"] in ALLOWED_ENVS, "Unknown environment label in health payload"
    assert data["version"] == PKG_VERSION, "Package version should match /health payload"
