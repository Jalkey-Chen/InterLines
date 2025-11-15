"""
Public translation pipeline (parser → explainer → brief, stub phase).

This module wires together the early-stage components of the InterLines
pipeline for a single, simple use case:

    raw text  →  parsed_chunks  →  explanation cards  →  public brief (stub)

At this stage, everything is still synchronous and in-process. The goal is to
provide a clean, testable orchestration function that other entry points
(e.g., CLI `interlines interpret`, HTTP API handlers) can call.

Core entry point
----------------
- ``run_pipeline(input_text: str, enable_history: bool = False)``

  1. Creates a fresh in-memory ``Blackboard``.
  2. Runs the parser agent to produce ``parsed_chunks``.
  3. Runs the explainer agent stub to produce three explanation cards.
  4. Assembles a *stub* public brief from those cards.
  5. Writes all major artifacts back to the blackboard and records a trace
     snapshot.
  6. Returns a small summary dictionary for easy inspection in tests and
     notebooks.

Design notes
------------
- We intentionally keep the return type as a plain nested dict plus the
  blackboard. This is stable for tests and can be later extended to emit
  proper Pydantic `PublicBrief` models without breaking callers.
- The brief assembly is deliberately simplistic (headline + two sections)
  and clearly marked as a stub so it can be upgraded in a later step.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

from interlines.agents.explainer_agent import (
    LEVEL_DEEP_DIVE,
    LEVEL_ONE_SENTENCE,
    LEVEL_THREE_PARAGRAPH,
    run_explainer_stub,
)
from interlines.agents.parser import parser_agent
from interlines.core.blackboard.memory import Blackboard


class PublicBriefPayload(TypedDict):
    """JSON-safe stub payload representing a public brief.

    Fields roughly mirror the future `PublicBrief` contract but are intentionally
    minimal here to keep the pipeline focused on data flow rather than schema
    details.
    """

    kind: str
    version: str
    title: str
    summary: str
    sections: list[dict[str, str]]
    meta: dict[str, Any]


class PipelineResult(TypedDict):
    """Return type for :func:`run_pipeline`.

    This keeps tests and simple callers decoupled from the internal blackboard
    layout while still exposing key artifacts.
    """

    blackboard: Blackboard
    parsed_chunks: list[str]
    explanations: list[dict[str, Any]]
    public_brief: PublicBriefPayload


def _select_card_by_level(
    cards: Sequence[Mapping[str, Any]],
    level: str,
) -> Mapping[str, Any] | None:
    """Return the first explanation card whose ``level`` matches ``level``.

    Parameters
    ----------
    cards:
        Sequence of explanation card-like dicts produced by the explainer agent.
    level:
        Explanation level identifier (e.g. ``"one_sentence"``).

    Returns
    -------
    dict[str, Any] | None
        The matching card, or ``None`` if no card with the requested level
        exists.
    """
    for card in cards:
        if card.get("level") == level:
            return card
    return None


def _build_public_brief_stub(
    input_text: str,
    chunks: Sequence[str],
    cards: Sequence[Mapping[str, Any]],
    enable_history: bool,
) -> PublicBriefPayload:
    """Assemble a minimal public brief payload from explanation cards.

    The mapping strategy in the stub phase is:

    - Title: derived from the one-sentence explanation's ``claim``.
    - Summary: taken from the one-sentence explanation's ``rationale``.
    - Sections:
        1. “Three-paragraph summary” — uses the corresponding explanation
           rationale.
        2. “Deep-dive commentary” — uses the deep-dive rationale.
    - Meta-information:
        - ``source_kind``: "stub"
        - ``num_chunks``: number of parsed chunks from the parser.
        - ``num_cards``: number of explanation cards.
        - ``enable_history``: flag forwarded from the planner (for future use).

    If some levels are missing, we fall back gracefully, choosing whatever
    explanation text is available.
    """
    one = _select_card_by_level(cards, LEVEL_ONE_SENTENCE) or (cards[0] if cards else {})
    three = _select_card_by_level(cards, LEVEL_THREE_PARAGRAPH) or one
    deep = _select_card_by_level(cards, LEVEL_DEEP_DIVE) or three

    title = str(one.get("claim") or "Public Brief (stub)")
    summary = str(one.get("rationale") or "")

    sections: list[dict[str, str]] = [
        {
            "id": "summary",
            "title": "Three-paragraph summary (stub)",
            "content": str(three.get("rationale") or summary),
        },
        {
            "id": "deep_dive",
            "title": "Deep-dive commentary (stub)",
            "content": str(deep.get("rationale") or summary),
        },
    ]

    meta: dict[str, Any] = {
        "source_kind": "stub",
        "num_chunks": len(chunks),
        "num_cards": len(cards),
        "enable_history": enable_history,
        # We keep the raw input here only for very early debugging; this is
        # *not* ideal for long texts and will likely be removed or truncated
        # when the real brief builder arrives.
        "input_preview": input_text[:280],
    }

    return {
        "kind": "public_brief.v1.stub",
        "version": "v1-stub",
        "title": title,
        "summary": summary,
        "sections": sections,
        "meta": meta,
    }


def run_pipeline(input_text: str, enable_history: bool = False) -> PipelineResult:
    """Run the public-translation pipeline on ``input_text``.

    Parameters
    ----------
    input_text:
        Raw text to be processed (e.g., an abstract, policy section, or article).
    enable_history:
        Flag indicating whether the planner intends to include historical /
        contextual layers. In the stub phase this only flows into ``meta`` but
        later it will control which agents are activated.

    Returns
    -------
    PipelineResult
        A dictionary containing:
        - ``blackboard``: the in-memory :class:`Blackboard` instance used.
        - ``parsed_chunks``: list of paragraph-like strings.
        - ``explanations``: list of explanation card-like dicts.
        - ``public_brief``: stub public brief payload assembled from the cards.

    Side effects
    ------------
    - Writes the following keys into the blackboard:
        - ``"parsed_chunks"`` — list[str]
        - ``"explanations"`` — list[dict[str, Any]]
        - ``"public_brief"`` — dict (stub payload)
    - Records a final trace snapshot with note
      ``"pipeline: public_translation complete"``.
    """
    bb = Blackboard()

    # 1) Parse raw text into chunks and record a trace snapshot.
    parsed_chunks = parser_agent(
        input_text,
        bb,
        key="parsed_chunks",
        min_chars=1,
        make_trace=True,
    )

    # 2) Run the explainer stub to produce three explanation cards.
    explanation_cards = run_explainer_stub(
        bb,
        source_key="parsed_chunks",
        target_key="explanations",
    )

    # 3) Assemble a stub public brief from the explanation cards.
    brief = _build_public_brief_stub(
        input_text=input_text,
        chunks=parsed_chunks,
        cards=explanation_cards,
        enable_history=enable_history,
    )
    bb.put("public_brief", brief)

    # 4) Final trace snapshot for the whole pipeline.
    bb.trace("pipeline: public_translation complete")

    return {
        "blackboard": bb,
        "parsed_chunks": parsed_chunks,
        "explanations": explanation_cards,
        "public_brief": brief,
    }


__all__ = ["run_pipeline", "PipelineResult", "PublicBriefPayload"]
