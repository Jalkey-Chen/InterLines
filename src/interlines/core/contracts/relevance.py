"""RelevanceNote — why an item matters to the current task."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .artifact import Artifact


class RelevanceNote(Artifact):
    """A scored note explaining why something is relevant.

    Required fields (per JSON schema):
    - kind, version, confidence (from Artifact)
    - target, rationale, score (defined here)
    """

    target: str = Field(description="What this note is about (id/ref/label)")
    rationale: str = Field(description="Why the target is relevant now")
    # Required: no default -> appears in JSON Schema "required"
    score: Annotated[
        float,
        Field(ge=0.0, le=1.0, description="Relevance score in [0,1]"),
    ]
