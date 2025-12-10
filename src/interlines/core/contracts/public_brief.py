"""PublicBrief — a higher-level composition targeting general audiences."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from .artifact import Artifact
from .explanation import ExplanationCard
from .relevance import RelevanceNote
from .review import ReviewReport
from .term import TermCard
from .timeline import TimelineEvent


class BriefSection(BaseModel):
    """
    A titled section with rich text and optional bullet highlights.

    (Legacy support for generic sections).
    """

    heading: str
    body: str
    bullets: list[str] = Field(default_factory=list)


class PublicBrief(Artifact):
    """
    A structured brief aggregating all agent outputs.

    This is the final composite artifact produced by the Planner pipeline.
    It includes the core summary, detailed explanations, glossary, timeline,
    audience relevance notes, and the quality assurance report.
    """

    title: str = Field(description="A catchy, accessible title for the brief.")

    summary: str = Field(description="A high-level executive summary.")

    # --- Legacy / Generic ---
    sections: list[BriefSection] = Field(
        default_factory=list, description="Generic sections (legacy support)."
    )

    # --- Core Explainer Agent ---
    explanations: Annotated[
        list[ExplanationCard],
        Field(default_factory=list, description="Detailed breakdown of claims and evidence."),
    ]

    # --- History Agent ---
    timeline: Annotated[
        list[TimelineEvent],
        Field(
            default_factory=list, description="Key chronological events extracted from the text."
        ),
    ]

    evolution_narrative: Annotated[
        str | None,
        Field(default=None, description="A short narrative describing how the topic evolved."),
    ]

    # --- Explainer/Glossary Agent ---
    glossary: Annotated[
        list[TermCard],
        Field(
            default_factory=list, description="Definitions of technical terms found in the text."
        ),
    ]

    # --- Citizen Agent ---
    relevance_notes: Annotated[
        list[RelevanceNote],
        Field(default_factory=list, description="'Why it matters' notes for specific audiences."),
    ]

    # --- Editor Agent ---
    review_report: Annotated[
        ReviewReport | None,
        Field(default=None, description="Quality assurance report and scores from the Editor."),
    ]


__all__ = ["PublicBrief", "BriefSection"]
