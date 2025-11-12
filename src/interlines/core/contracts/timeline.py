"""TimelineEvent — a dated event used for historical/evolutionary views."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import Field

from .artifact import Artifact


class TimelineEvent(Artifact):
    """A single event on a timeline with optional tags and sources."""

    kind: str = Field(default="timeline_event.v1")
    version: str = Field(default="1.0.0")

    when: date | datetime = Field(description="Date/time of the event")
    title: str
    description: str | None = Field(default=None)
    tags: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


__all__ = ["TimelineEvent"]
