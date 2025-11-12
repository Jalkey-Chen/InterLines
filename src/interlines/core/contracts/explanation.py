"""ExplanationCard — a structured, referenceable explanation unit."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .artifact import Artifact


class EvidenceItem(BaseModel):
    """Minimal evidence pointer supporting the explanation."""

    text: str = Field(description="Quoted or paraphrased support")
    source: str | None = Field(default=None, description="Citation string, URL, or opaque locator")


class ExplanationCard(Artifact):
    """A compact explanation of a claim with evidence and rationale.

    Fields
    ------
    claim : str
        The central statement being explained/justified.
    rationale : str
        Coherent reasoning supporting why the claim holds.
    evidence : list[EvidenceItem]
        Optional supporting items referencing sources or earlier derivations.
    summary : Optional[str]
        A one- or two-sentence TL;DR of the explanation.
    """

    kind: str = Field(default="explanation.v1", description="Artifact kind tag")
    version: str = Field(default="1.0.0", description="Explanation schema version")

    claim: str
    rationale: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    summary: str | None = None


__all__ = ["ExplanationCard", "EvidenceItem"]
