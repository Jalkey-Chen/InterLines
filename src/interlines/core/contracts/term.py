"""TermCard — a glossary entry with definition and examples."""

from __future__ import annotations

from pydantic import Field

from .artifact import Artifact


class TermCard(Artifact):
    """A defined term used across InterLines content.

    Fields
    ------
    term : str
        The canonical term (lower snake/camel/kebab allowed).
    definition : str
        A concise, plain-language definition.
    aliases : list[str]
        Optional alternate spellings or synonyms.
    examples : list[str]
        Short usage or illustrative examples.
    sources : list[str]
        Citations or URLs that ground the definition.
    """

    kind: str = Field(default="term.v1")
    version: str = Field(default="1.0.0")

    term: str
    definition: str
    aliases: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


__all__ = ["TermCard"]
