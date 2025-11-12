"""RelevanceNote — why an item matters to the current task."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .artifact import Artifact


class RelevanceNote(Artifact):
    """A scored note explaining why something is relevant."""

    kind: str = Field(default="relevance.v1")
    version: str = Field(default="1.0.0")

    target: str = Field(description="What this note is about (id/ref/label)")
    rationale: str = Field(description="Why the target is relevant now")
    score: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.5, description="Relevance score in [0,1]"
    )


__all__ = ["RelevanceNote"]
