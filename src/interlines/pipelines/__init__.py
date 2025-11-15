"""Pipeline entry points for InterLines.

Currently exposed:

- :func:`run_pipeline` — stub public-translation pipeline
  (parser → explainer → brief), implemented in ``public_translation.py``.
"""

from __future__ import annotations

from .public_translation import PipelineResult, PublicBriefPayload, run_pipeline

__all__ = ["run_pipeline", "PipelineResult", "PublicBriefPayload"]
