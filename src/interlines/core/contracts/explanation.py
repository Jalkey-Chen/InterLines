"""ExplanationCard — a structured, referenceable explanation unit."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .artifact import Artifact


class EvidenceItem(BaseModel):
    """Minimal evidence pointer supporting the explanation."""

    text: str = Field(description="Quoted or paraphrased support")
    source: str | None = Field(default=None, description="Citation string, URL, or opaque locator")


class ExplanationCard(Artifact):
    """A compact explanation of a claim with evidence and rationale."""

    claim: str
    rationale: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    summary: str | None = None


__all__ = ["ExplanationCard", "EvidenceItem"]
