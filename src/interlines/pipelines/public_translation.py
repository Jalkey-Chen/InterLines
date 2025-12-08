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


class PublicBriefMeta(TypedDict):
    """Meta-information about the brief, used by UIs and tests."""

    num_chunks: int
    num_cards: int
    enable_history: bool
    input_preview: str


class PublicBriefPayload(TypedDict):
    """JSON-safe stub payload representing a public brief.

    The structure is intentionally minimal at this stage. It is designed
    to be easy to render in a CLI or notebook and to evolve into a proper
    `PublicBrief` contract in later phases.
    """

    # Machine-facing type tag, used by tests and future UIs.
    kind: str

    # Small meta block summarizing the pipeline run.
    meta: PublicBriefMeta

    # Human-facing content.
    title: str
    highlights: list[str]
    sections: list[dict[str, Any]]

    # Optional note used when history is enabled.
    history_note: NotRequired[str]


class PipelineResult(TypedDict):
    """Return type for :func:`run_pipeline`.

    This keeps tests and simple callers decoupled from the internal blackboard
    layout while still exposing key artifacts.
    """

    blackboard: Blackboard
    parsed_chunks: list[str]
    explanations: list[dict[str, Any]]
    public_brief: PublicBriefPayload


def _make_input_preview(source_text: str, max_len: int = 240) -> str:
    """Build a short preview of the original input text."""
    preview = source_text.strip()
    if not preview:
        return ""
    if len(preview) <= max_len:
        return preview
    return preview[: max_len - 3].rstrip() + "..."


def _card_to_dict(card: ExplanationCard | Mapping[str, Any]) -> dict[str, Any]:
    """Normalize an explanation card to a plain dict.

    The explainer agent returns `ExplanationCard` instances, but the
    pipeline API and the blackboard should expose JSON-safe dicts.
    """
    # Already a mapping (including plain dict) - just make a shallow copy.
    if isinstance(card, Mapping):
        return dict(card)

    # For ExplanationCard (a Pydantic model), use its built-in serializers.
    if hasattr(card, "model_dump"):
        # Pydantic v2 style
        return card.model_dump()
    if hasattr(card, "dict"):
        # Pydantic v1 style
        return card.dict()

    # Very defensive fallback - should rarely be needed.
    return {key: getattr(card, key) for key in dir(card) if not key.startswith("_")}


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
    cards: Sequence[dict[str, Any]],
    enable_history: bool,
    num_chunks: int,
) -> PublicBriefPayload:
    """Assemble a simple public-facing brief from explanation cards.

    This is a purely local, LLM-free implementation used for tests and
    offline execution. It turns the available :class:`ExplanationCard`
    instances into a minimal but readable brief structure.
    """
    # Highlights: take up to the first three non-empty claims.
    highlights: list[str] = []
    for card in cards:
        claim = str(card.get("claim", "") or "")
        if claim:
            highlights.append(claim)
        if len(highlights) >= 3:
            break

    if not highlights:
        highlights = ["No explanations are available in this stub pipeline."]

    # Sections: one section per explanation card.
    sections: list[dict[str, Any]] = []
    for card in cards:
        claim = str(card.get("claim", "") or "")
        rationale = str(card.get("rationale", "") or "")
        title_text = claim or "Explanation"
        body = rationale or claim or ""
        sections.append(
            {
                "title": title_text,
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
        "kind": "public_brief.stub.v1",
        "meta": {
            "num_chunks": num_chunks,
            "num_cards": len(cards),
            "enable_history": enable_history,
            "input_preview": _make_input_preview(source_text),
        },
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
    # Record a human-readable marker for the beginning of the pipeline.
    bb.trace(f"pipeline: public_translation start (enable_history={enable_history})")

    # 1) Parse raw text into chunks and record a trace snapshot.
    parsed_chunks = parser.parser_agent(
        input_text,
        bb,
        key=_PARSED_CHUNKS_KEY,
        min_chars=1,
        make_trace=True,
    )

    # 2) Run the explainer stub to produce explanation cards.
    raw_cards = run_explainer_stub(
        bb,
        source_key=_PARSED_CHUNKS_KEY,
        target_key=_EXPLANATIONS_KEY,
    )

    # 3) Convert ExplanationCard objects to plain dicts for external use.
    cards_as_dicts = [_card_to_dict(card) for card in raw_cards]

    # Keep the blackboard consistent with the public API: store dicts.
    bb.put(_EXPLANATIONS_KEY, cards_as_dicts)

    # 4) Assemble a stub public brief from the explanation dicts.
    brief = _build_public_brief_stub(
        title=PIPELINE_BRIEF_TITLE,
        source_text=input_text,
        cards=cards_as_dicts,
        enable_history=enable_history,
        num_chunks=len(parsed_chunks),
    )

    # 5) Persist the brief and take a final trace snapshot.
    bb.put(_PUBLIC_BRIEF_KEY, brief)
    # Record a marker for the end of the pipeline.
    bb.trace("pipeline: public_translation complete")

    result: PipelineResult = {
        "blackboard": bb,
        "parsed_chunks": parsed_chunks,
        "explanations": cards_as_dicts,
        "public_brief": brief,
    }
    return result


__all__ = ["run_pipeline", "PipelineResult", "PublicBriefPayload", "PIPELINE_BRIEF_TITLE"]
