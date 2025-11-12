"""ReviewReport — evaluation results for artifacts/process outputs."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .artifact import Artifact


class ReviewCriteria(Artifact):
    """Score breakdown for common review dimensions."""

    kind: str = Field(default="review_criteria.v1")
    version: str = Field(default="1.0.0")

    accuracy: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    clarity: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    completeness: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    safety: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5


class ReviewReport(Artifact):
    """Aggregate review with overall score and freeform comments."""

    kind: str = Field(default="review_report.v1")
    version: str = Field(default="1.0.0")

    overall: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    criteria: ReviewCriteria = Field(default_factory=ReviewCriteria)
    comments: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)


__all__ = ["ReviewCriteria", "ReviewReport"]
