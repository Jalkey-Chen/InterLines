"""
Public translation pipeline (parser → explainer → brief, stub phase).

This module wires together the early-stage components of the InterLines
pipeline for a single, simple use case:

    raw text  →  parsed_chunks  →  explanation cards  →  public brief (stub)

At this stage, everything is synchronous and in-process. The goal is to
provide a clean, testable orchestration function that other entry points
(e.g., CLI `interlines interpret`, HTTP API handlers) can call.

Core entry point
----------------
- ``run_pipeline(input_text: str, enable_history: bool = False)``

  1. Creates a fresh in-memory ``Blackboard``.
  2. Runs the parser agent to produce ``parsed_chunks``.
  3. Runs the explainer agent (or its stub) to produce explanation cards.
  4. Assembles a *stub* public brief from those cards.
  5. Writes all major artifacts back to the blackboard and records a trace
     snapshot.
  6. Returns a small summary dictionary for easy inspection in tests and
     notebooks.

Design notes
------------
- We intentionally keep the return type as a plain nested dict plus the
  blackboard. This is stable for tests and can later be extended to emit
  proper Pydantic `PublicBrief` models without breaking callers.
- The brief assembly is deliberately simplistic and clearly marked as a stub
  so it can be upgraded in a later step.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NotRequired, TypedDict

from interlines.agents import parser
from interlines.agents.explainer_agent import run_explainer_stub
from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.explanation import ExplanationCard

# Blackboard keys used by this pipeline.
_PARSED_CHUNKS_KEY = "parsed_chunks"
_EXPLANATIONS_KEY = "explanations"
_PUBLIC_BRIEF_KEY = "public_brief"

# Human-facing title for the stub brief. Tests import this constant.
PIPELINE_BRIEF_TITLE = "InterLines public translation (stub)"


class PublicBriefPayload(TypedDict):
    """JSON-safe stub payload representing a public brief.

    The structure is intentionally minimal at this stage. It is designed
    to be easy to render in a CLI or notebook and to evolve into a proper
    `PublicBrief` contract in later phases.
    """

    title: str
    highlights: list[str]
    sections: list[dict[str, Any]]
    history_note: NotRequired[str]


class PipelineResult(TypedDict):
    """Return type for :func:`run_pipeline`.

    This keeps tests and simple callers decoupled from the internal blackboard
    layout while still exposing key artifacts.
    """

    blackboard: Blackboard
    parsed_chunks: list[str]
    explanations: list[ExplanationCard]
    public_brief: PublicBriefPayload


def _select_card_by_level(
    cards: Sequence[Mapping[str, Any]],
    level: str,
) -> Mapping[str, Any] | None:
    """Return the first explanation card whose ``level`` matches ``level``.

    This helper is kept for future use when cards carry an explicit
    ``level`` field (e.g. "one_sentence", "three_paragraph", "deep_dive").
    It is not currently used by the stub brief builder.
    """
    for card in cards:
        if card.get("level") == level:
            return card
    return None


def _build_public_brief_stub(
    *,
    title: str,
    source_text: str,
    cards: Sequence[ExplanationCard],
    enable_history: bool,
) -> PublicBriefPayload:
    """Assemble a simple public-facing brief from explanation cards.

    This is a purely local, LLM-free implementation used for tests and
    offline execution. It turns the available :class:`ExplanationCard`
    instances into a minimal but readable brief structure.
    """
    # Highlights: take up to the first three non-empty claims.
    highlights: list[str] = []
    for card in cards:
        if card.claim:
            highlights.append(card.claim)
        if len(highlights) >= 3:
            break

    if not highlights:
        highlights = ["No explanations are available in this stub pipeline."]

    # Sections: one section per explanation card.
    sections: list[dict[str, Any]] = []
    for card in cards:
        body = card.rationale or card.claim or ""
        sections.append(
            {
                "title": card.claim or "Explanation",
                "body": body,
            }
        )

    if not sections:
        sections.append(
            {
                "title": "Explanation",
                "body": "No detailed rationale is available in this stub pipeline.",
            }
        )

    brief: PublicBriefPayload = {
        "title": title,
        "highlights": highlights,
        "sections": sections,
    }

    if enable_history:
        brief["history_note"] = (
            "History view is enabled, but detailed timeline generation is "
            "not yet implemented in this stub."
        )

    return brief


def run_pipeline(input_text: str, enable_history: bool = False) -> PipelineResult:
    """Run the public-translation pipeline on ``input_text``.

    Parameters
    ----------
    input_text:
        Raw text to be processed (e.g., an abstract, policy section, or article).
    enable_history:
        Flag indicating whether the planner intends to include historical /
        contextual layers. In the stub phase this only flows into the brief
        payload but later it will control which agents are activated.

    Returns
    -------
    PipelineResult
        A dictionary containing:
        - ``blackboard``: the in-memory :class:`Blackboard` instance used.
        - ``parsed_chunks``: list of paragraph records (``{"id", "text"}``).
        - ``explanations``: list of :class:`ExplanationCard` instances.
        - ``public_brief``: stub public brief payload assembled from the cards.

    Side effects
    ------------
    - Writes the following keys into the blackboard:
        - ``"parsed_chunks"``  — list[Mapping[str, Any]]
        - ``"explanations"``   — list[ExplanationCard]
        - ``"public_brief"``   — PublicBriefPayload
    - Records a final trace snapshot with note
      ``"pipeline: public_translation complete"``.
    """
    bb = Blackboard()

    # 1) Parse raw text into chunks and record a trace snapshot.
    parsed_chunks = parser.parser_agent(
        input_text,
        bb,
        key=_PARSED_CHUNKS_KEY,
        min_chars=1,
        make_trace=True,
    )

    # 2) Run the explainer stub to produce explanation cards.
    explanation_cards = run_explainer_stub(
        bb,
        source_key=_PARSED_CHUNKS_KEY,
        target_key=_EXPLANATIONS_KEY,
    )

    # 3) Assemble a stub public brief from the explanation cards.
    brief = _build_public_brief_stub(
        title=PIPELINE_BRIEF_TITLE,
        source_text=input_text,
        cards=explanation_cards,
        enable_history=enable_history,
    )

    # 4) Persist the brief and take a final trace snapshot.
    bb.put(_PUBLIC_BRIEF_KEY, brief)
    bb.trace("pipeline: public_translation complete")

    result: PipelineResult = {
        "blackboard": bb,
        "parsed_chunks": parsed_chunks,
        "explanations": explanation_cards,
        "public_brief": brief,
    }
    return result


__all__ = ["run_pipeline", "PipelineResult", "PublicBriefPayload", "PIPELINE_BRIEF_TITLE"]
