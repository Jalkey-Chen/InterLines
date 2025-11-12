"""ReviewReport — evaluation results for artifacts/process outputs.

This module defines two Pydantic models:

- ReviewCriteria: a dimension-wise score card (inherits Artifact).
- ReviewReport: an aggregate report that *requires* a `criteria` object,
  plus an overall score, comments, and suggested actions.

Contract notes
--------------
- All `Artifact` fields (`kind`, `version`, `confidence`) are required by design.
- `criteria` in `ReviewReport` is required to align with `schemas/review_report.v1.json`.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .artifact import Artifact


class ReviewCriteria(Artifact):
    """Score breakdown for common review dimensions.

    Each score is a calibrated value within [0.0, 1.0].
    """

    accuracy: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    clarity: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    completeness: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    safety: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5


class ReviewReport(Artifact):
    """Aggregate review with overall score and freeform comments.

    Fields
    ------
    overall : float in [0,1]
        Overall evaluation score.
    criteria : ReviewCriteria
        REQUIRED nested criteria object (no default) per JSON schema.
    comments : list[str]
        Optional freeform remarks from reviewers.
    actions : list[str]
        Optional actionable follow-ups or remediation steps.
    """

    overall: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    criteria: ReviewCriteria  # REQUIRED
    comments: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)


__all__ = ["ReviewCriteria", "ReviewReport"]
