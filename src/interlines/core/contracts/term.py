"""TermCard — a glossary entry with definition and examples."""

from __future__ import annotations

from pydantic import Field

from .artifact import Artifact


class TermCard(Artifact):
    """A defined term used across InterLines content."""

    term: str
    definition: str
    aliases: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


__all__ = ["TermCard"]
