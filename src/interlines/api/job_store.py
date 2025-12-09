"""
In-Memory Job Store for Async Task Management.

Milestone
---------
M6 | Interface & Deployment
Step 6.1 | API interpret/brief/health

This module implements a simple, thread-safe(ish) store for tracking the
lifecycle of analysis jobs.

Responsibilities
----------------
- **Create**: Generate UUIDs for new requests and mark them PENDING.
- **Read**: Retrieve current status and results by Job ID.
- **Update**: Transition jobs from PROCESSING -> COMPLETED/FAILED.

Note on Persistence
-------------------
This is a volatile memory store. If the server restarts, all job history is
lost. This is acceptable for the M6 MVP. For production, this class would be
swapped with a Redis or Database implementation.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import ClassVar

from interlines.api.schemas import InterpretResult, JobInfo, JobStatus


class JobStore:
    """
    A simple dictionary-backed store for JobInfo objects.
    """

    # Singleton instance placeholder (initialized in app startup)
    _instance: ClassVar[JobStore | None] = None

    def __init__(self) -> None:
        self._jobs: dict[str, JobInfo] = {}

    @classmethod
    def get_instance(cls) -> JobStore:
        """Accessor for the global singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def create_job(self) -> str:
        """
        Register a new job ID and initialize its state to PENDING.

        Returns
        -------
        str
            The generated UUID4 string for the new job.
        """
        job_id = str(uuid.uuid4())
        now = datetime.now(UTC)

        info = JobInfo(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=now,
            error=None,
            result=None,
        )
        self._jobs[job_id] = info
        return job_id

    def get_job(self, job_id: str) -> JobInfo | None:
        """Retrieve job metadata, or None if not found."""
        return self._jobs.get(job_id)

    def mark_processing(self, job_id: str) -> None:
        """Transition a job to PROCESSING state."""
        if job := self._jobs.get(job_id):
            job.status = JobStatus.PROCESSING

    def mark_completed(self, job_id: str, result: InterpretResult) -> None:
        """Transition a job to COMPLETED and attach the result."""
        if job := self._jobs.get(job_id):
            job.status = JobStatus.COMPLETED
            job.result = result

    def mark_failed(self, job_id: str, error: str) -> None:
        """Transition a job to FAILED and attach the error message."""
        if job := self._jobs.get(job_id):
            job.status = JobStatus.FAILED
            job.error = error


# Global accessor for convenience
def get_job_store() -> JobStore:
    return JobStore.get_instance()


__all__ = ["JobStore", "get_job_store"]
