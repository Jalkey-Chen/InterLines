"""API placeholder for Step 0.1.

We define a tiny runner to satisfy the `interlines-api` script entry defined in pyproject.
A real FastAPI app (with `/health`) will be added in Step 0.3. For now, we only expose a stub.
"""

from __future__ import annotations


def run() -> None:
    """Placeholder server runner.

    This function exists so that `uv run interlines-api` does not fail before Step 0.3.
    It simply prints a message and exits. CI never calls this; it's here to satisfy
    package metadata and to keep future diffs minimal.
    """
    print("InterLines API placeholder (Step 0.3 will add FastAPI /health).")
