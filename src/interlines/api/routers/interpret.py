"""
API Routes for Interpretation Jobs.

Milestone
---------
M6 | Interface & Deployment
Step 6.1 | API interpret/brief/health

This module defines the REST endpoints for submitting and monitoring
analysis jobs.

Endpoints
---------
- `POST /interpret`: Submit a text for analysis (Async).
- `GET /jobs/{job_id}`: Poll the status and retrieve results.

Design Decisions
----------------
- **Asynchronous Handoff**: The POST endpoint returns 202 Accepted immediately.
- **RESTful Resource**: Jobs are treated as resources identified by UUID.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from interlines.api.background import run_pipeline_task
from interlines.api.job_store import get_job_store
from interlines.api.schemas import InterpretRequest, JobInfo

router = APIRouter(tags=["Interpretation"])


@router.post(
    "/interpret",
    response_model=JobInfo,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a new analysis job",
)
async def submit_interpretation(
    request: InterpretRequest,
    background_tasks: BackgroundTasks,
) -> JobInfo:
    """
    Dispatch a new analysis pipeline job.

    This endpoint accepts the source text and configuration, creates a tracking
    ID, schedules the background worker, and returns immediately.

    Client Workflow
    ---------------
    1. Receive `job_id` from this response.
    2. Poll `GET /jobs/{job_id}` until status is 'completed'.
    """
    store = get_job_store()

    # 1. Create Job Entry (Pending state)
    job_id = store.create_job()

    # 2. Schedule Background Execution
    # FastAPI handles the thread pool management for us.
    background_tasks.add_task(
        run_pipeline_task,
        job_id=job_id,
        text=request.text,
        enable_history=request.enable_history,
        use_llm_planner=request.use_llm_planner,
    )

    # 3. Return Initial State
    # The job is guaranteed to exist in the store at this point.
    job_info = store.get_job(job_id)
    if not job_info:
        # Should be unreachable given create_job logic
        raise HTTPException(status_code=500, detail="Failed to create job")

    return job_info


@router.get(
    "/jobs/{job_id}",
    response_model=JobInfo,
    summary="Get job status and results",
)
async def get_job_status(job_id: str) -> JobInfo:
    """
    Retrieve the current status or final result of a job.

    Returns
    -------
    JobInfo
        Contains `status` (pending/processing/completed/failed) and,
        if completed, the `result` object (public brief).
    """
    store = get_job_store()
    job = store.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    return job


__all__ = ["router"]
