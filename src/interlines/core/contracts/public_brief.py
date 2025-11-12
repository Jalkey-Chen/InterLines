"""PublicBrief — a higher-level composition targeting general audiences."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .artifact import Artifact


class BriefSection(BaseModel):
    """A titled section with rich text and optional bullet highlights."""

    heading: str
    body: str
    bullets: list[str] = Field(default_factory=list)


class PublicBrief(Artifact):
    """A structured brief with a title, summary, and sections."""

    kind: str = Field(default="public_brief.v1")
    version: str = Field(default="1.0.0")

    title: str
    summary: str
    sections: list[BriefSection] = Field(default_factory=list)


__all__ = ["PublicBrief", "BriefSection"]
