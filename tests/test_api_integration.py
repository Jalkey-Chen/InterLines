# tests/test_api_integration.py
"""
Integration Tests for the InterLines HTTP API.

Milestone
---------
M6 | Interface & Deployment
Step 6.1 | API interpret/brief/health

Focus
-----
These tests verify the HTTP contract (request/response schemas) and the
async job state machine. They DO NOT execute the actual LLM pipeline;
instead, the pipeline is mocked to return canned responses.

Scenarios
---------
1. **Health Check**: Verify service is up.
2. **Happy Path**: Submit -> 202 -> Poll (Pending) -> Poll (Completed).
3. **Error Handling**: Verify 404s for missing jobs.
"""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from interlines.api.app import create_app
from interlines.api.job_store import get_job_store
from interlines.api.schemas import JobStatus


# Fixed (MyPy): Added type ignore because mypy running inside pre-commit
# often cannot see pytest's internal type stubs for decorators.
@pytest.fixture  # type: ignore[misc]
def client() -> Generator[TestClient, None, None]:
    """
    Pytest fixture to create a clean API client for each test.

    This ensures the JobStore singleton is reset or isolated if possible.
    Since JobStore is a singleton, we explicitly clear it here.
    """
    # 1. Reset the Singleton Store
    store = get_job_store()
    store._jobs.clear()

    # 2. Create App & Client
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_health_check(client: TestClient) -> None:
    """GET /health should return 200 OK and version info."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_submit_and_poll_flow(client: TestClient) -> None:
    """
    Verify the full Async Job Lifecycle:
    POST /interpret -> 202 Accepted -> GET /jobs/{id} -> Completed.
    """
    # -----------------------------------------------------------------------
    # 1. Setup Mock Pipeline
    # -----------------------------------------------------------------------
    # We mock the internal `run_pipeline` function called by the background worker.
    # We want it to return a valid "success" structure.
    mock_pipeline_output = {
        "public_brief": {
            "title": "Mock Title",
            "summary": "Mock Summary",
            "sections": [{"heading": "Section 1", "body": "Body 1", "bullets": ["b1"]}],
        },
        # Other fields are ignored by the API schema, but needed by return type
        "blackboard": {},
        "parsed_chunks": [],
        "explanations": [],
        "relevance_notes": [],
        "terms": [],
        "timeline_events": [],
        "public_brief_md_path": None,
    }

    # Patch where it is imported in `background.py`
    with patch("interlines.api.background.run_pipeline") as mock_run:
        mock_run.return_value = mock_pipeline_output

        # -----------------------------------------------------------------------
        # 2. Submit Job (POST)
        # -----------------------------------------------------------------------
        payload = {
            "text": "This is a test text for API integration.",
            "enable_history": False,
            "use_llm_planner": False,
        }
        resp_post = client.post("/interpret", json=payload)

        assert resp_post.status_code == 202
        data_post = resp_post.json()
        job_id = data_post["job_id"]
        assert job_id
        assert data_post["status"] == "pending"

        # NOTE: TestClient runs BackgroundTasks *synchronously* after the request
        # is returned. So by the time we get here, `run_pipeline_task` has
        # likely already finished in the mock environment.

        # -----------------------------------------------------------------------
        # 3. Poll Job (GET)
        # -----------------------------------------------------------------------
        resp_get = client.get(f"/jobs/{job_id}")
        assert resp_get.status_code == 200
        data_get = resp_get.json()

        # It should be completed because TestClient executes tasks immediately
        assert data_get["status"] == JobStatus.COMPLETED
        assert data_get["result"] is not None

        brief = data_get["result"]["brief"]
        assert brief["title"] == "Mock Title"
        assert brief["summary"] == "Mock Summary"
        assert len(brief["sections"]) == 1

        # Verify our mock was called with correct args
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["input_text"] == payload["text"]
        assert call_kwargs["enable_history"] is False


def test_get_non_existent_job(client: TestClient) -> None:
    """GET /jobs/{id} with unknown ID should return 404."""
    response = client.get("/jobs/fake-uuid-1234")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_pipeline_failure_handling(client: TestClient) -> None:
    """Verify that pipeline exceptions are captured as JobStatus.FAILED."""

    # Patch the pipeline to raise an exception
    with patch("interlines.api.background.run_pipeline") as mock_run:
        mock_run.side_effect = RuntimeError("Simulated Pipeline Crash")

        # Submit
        resp = client.post("/interpret", json={"text": "Crash me please, make it longer"})
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        # Poll (TestClient runs background task immediately, so it should have failed)
        resp_get = client.get(f"/jobs/{job_id}")
        data = resp_get.json()

        assert data["status"] == JobStatus.FAILED
        assert "Pipeline Error" in data["error"]
        assert "Simulated Pipeline Crash" in data["error"]
