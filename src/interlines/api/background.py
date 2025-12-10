# src/interlines/api/background.py
"""
Background Task Runner for the Analysis Pipeline.

Milestone
---------
M6 | Interface & Deployment
Step 6.1 | API interpret/brief/health

This module provides the worker function used by FastAPI's `BackgroundTasks`.
It wraps the synchronous `run_pipeline` call with exception handling and
state management logic.

Updates
-------
- Updated `run_pipeline` call to use the new `input_data` argument signature.
"""

from __future__ import annotations

from interlines.api.job_store import get_job_store
from interlines.api.schemas import InterpretResult, PublicBriefPayload
from interlines.pipelines.public_translation import run_pipeline


def run_pipeline_task(
    job_id: str,
    text: str,
    enable_history: bool,
    use_llm_planner: bool,
) -> None:
    """
    Execute the pipeline in a background thread and update the job store.

    This function is intended to be scheduled via `FastAPI.BackgroundTasks`.
    It never raises exceptions to the caller; instead, it captures them and
    marks the job as FAILED.

    Parameters
    ----------
    job_id:
        The UUID of the job to update.
    text:
        Raw input text for analysis.
    enable_history:
        Flag passed to the pipeline strategy.
    use_llm_planner:
        Flag passed to the pipeline orchestrator.
    """
    store = get_job_store()

    # 1. Transition to PROCESSING
    # This indicates the worker thread has actually picked up the task.
    store.mark_processing(job_id)

    try:
        # 2. Invoke the Core Pipeline
        # This is a synchronous call that may take 10-60 seconds.
        # It handles all internal agent interactions and blackboard writes.
        pipeline_output = run_pipeline(
            # Fixed: Updated argument name from 'input_text' to 'input_data'
            # to match the refactored pipeline signature.
            input_data=text,
            enable_history=enable_history,
            use_llm_planner=use_llm_planner,
        )

        # 3. Transform Output to API Schema
        # The pipeline returns a TypedDict; we validate it into our Pydantic model.
        # This ensures the API contract is strictly met.
        brief_payload = PublicBriefPayload.model_validate(pipeline_output["public_brief"])

        api_result = InterpretResult(brief=brief_payload)

        # 4. Transition to COMPLETED
        store.mark_completed(job_id, api_result)

    except Exception as exc:
        # 5. Handle Failures
        # Capture the full traceback for debugging (could be logged here).
        error_msg = f"Pipeline Error: {exc}"
        # detailed_trace = traceback.format_exc()  # In prod, log this.

        store.mark_failed(job_id, error_msg)


__all__ = ["run_pipeline_task"]
