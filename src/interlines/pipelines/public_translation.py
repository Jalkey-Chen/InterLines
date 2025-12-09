"""
Public translation pipeline: from raw text to public-facing artifacts.

This module orchestrates the main InterLines "public translation" flow:

    raw text
      → parsed_chunks (parser_agent)
      → explanation cards (explainer_agent)
      → relevance notes (citizen_agent)
      → jargon / term cards (jargon_agent)
      → timeline events + evolution narrative (history_agent, optional)
      → review report (editor_agent)
      → Markdown brief on disk (brief_builder)

The goal is to provide a single synchronous entry point that other
frontends (CLI, HTTP API, notebooks) can call in tests and prototypes.

Core entry point
----------------
- ``run_pipeline(input_text: str, enable_history: bool = False)``

which returns a JSON-safe ``PipelineResult`` containing:

- The shared :class:`Blackboard` instance.
- Parsed chunks as a list of strings.
- Explanation / relevance / term / timeline artifacts as dicts.
- A structured ``PublicBrief`` payload built from explanations.
- The path to a generated Markdown brief (if any).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict, cast

from interlines.agents.brief_builder import (
    _PUBLIC_BRIEF_MD_KEY,
    run_brief_builder,
)
from interlines.agents.citizen_agent import run_citizen
from interlines.agents.editor_agent import run_editor
from interlines.agents.explainer_agent import run_explainer
from interlines.agents.history_agent import run_history
from interlines.agents.jargon_agent import run_jargon
from interlines.agents.parser import parser_agent
from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.explanation import ExplanationCard
from interlines.core.contracts.public_brief import BriefSection, PublicBrief
from interlines.core.contracts.relevance import RelevanceNote
from interlines.core.contracts.review import ReviewReport
from interlines.core.contracts.term import TermCard
from interlines.core.contracts.timeline import TimelineEvent
from interlines.core.planner.strategy import build_plan
from interlines.core.result import Result

# Blackboard keys (kept consistent with individual agents).
_PARSED_CHUNKS_KEY = "parsed_chunks"
_EXPLANATIONS_KEY = "explanations"
_RELEVANCE_NOTES_KEY = "relevance_notes"
_TERMS_KEY = "terms"
_TIMELINE_KEY = "timeline_events"
_PUBLIC_BRIEF_KEY = "public_brief"

# Public brief defaults.
PIPELINE_BRIEF_TITLE = "InterLines Public Brief"


class BriefSectionPayload(TypedDict):
    """JSON-safe representation of a single brief section."""

    heading: str
    body: str
    bullets: list[str]


class PublicBriefPayload(TypedDict):
    """JSON-safe representation of :class:`PublicBrief`."""

    title: str
    summary: str
    sections: list[BriefSectionPayload]


class PipelineResult(TypedDict):
    """JSON-safe result object returned by :func:`run_pipeline`.

    Fields
    ------
    blackboard:
        The shared :class:`Blackboard` used by all agents.
    parsed_chunks:
        Paragraph-level chunks produced by the parser.
    explanations:
        List of explanation cards (dict form) from the explainer agent.
    relevance_notes:
        List of public-facing relevance notes (dict form) from the citizen agent.
    terms:
        List of jargon / terminology cards (dict form).
    timeline_events:
        List of timeline events (dict form). May be empty if history is disabled.
    review_report:
        Structured review report (dict form) from the editor agent, or ``None``.
    public_brief:
        Structured public brief payload built from explanations.
    public_brief_md_path:
        Filesystem path to the generated Markdown brief, or ``None`` if
        brief generation failed.
    """

    blackboard: Blackboard
    parsed_chunks: list[str]
    explanations: list[dict[str, Any]]
    relevance_notes: list[dict[str, Any]]
    terms: list[dict[str, Any]]
    timeline_events: list[dict[str, Any]]
    review_report: dict[str, Any] | None
    public_brief: PublicBriefPayload
    public_brief_md_path: str | None


def _as_list(value: Any) -> list[Any]:
    """Normalize an arbitrary value into a list for iteration."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return list(value)
    return [value]


def _artifact_to_dict(obj: Any) -> dict[str, Any]:
    """Convert Pydantic artifacts or mappings into plain dicts.

    The pipeline exposes artifacts as JSON-safe dicts rather than model
    instances so that callers can serialize them directly.
    """
    if isinstance(obj, Mapping):
        return dict(obj)

    # Pydantic v1 style: `.dict()`
    if hasattr(obj, "dict"):
        # mypy sees `.dict()` as `Any`, so we cast to the expected type.
        return cast("dict[str, Any]", obj.dict())

    # Pydantic v2 style: `.model_dump()`
    if hasattr(obj, "model_dump"):
        # Same here: cast the result to `dict[str, Any]` for static typing.
        return cast("dict[str, Any]", obj.model_dump())

    # Fallback: best-effort representation for unknown types.
    return {"value": repr(obj)}


def _preview(text: str, max_len: int = 240) -> str:
    """Return a short, human-readable preview for summaries.

    The function is intentionally simple: it truncates on character count
    and avoids splitting inside the last word if we have to shorten.
    """
    s = (text or "").strip()
    if len(s) <= max_len:
        return s
    cut = s[: max_len - 3].rsplit(" ", 1)[0]
    if not cut:
        cut = s[: max_len - 3]
    return cut.rstrip() + "..."


def _build_public_brief_from_explanations(
    cards: list[ExplanationCard] | list[Mapping[str, Any]],
) -> PublicBrief:
    """Construct a simple :class:`PublicBrief` from explanation cards.

    Strategy
    --------
    - Title: use the first card's claim, falling back to a generic title
      if none is available.
    - Summary: use the first card's rationale (or claim) truncated with
      :func:`_preview`.
    - Sections: one section per card, with the claim as the heading and
      the rationale as the body.
    """
    if not cards:
        brief = PublicBrief(
            kind="public_brief.v1",
            version="1.0.0",
            confidence=0.0,
            title=PIPELINE_BRIEF_TITLE,
            summary="No explanations were available to construct a brief.",
            sections=[],
        )
        return brief

    def _get_field(card: Any, key: str) -> str:
        """Retrieve a field from a card-like object as a stripped string."""
        if isinstance(card, Mapping):
            value = card.get(key)
        else:
            value = getattr(card, key, None)
        return "" if value is None else str(value).strip()

    first = cards[0]
    first_claim = _get_field(first, "claim")
    first_rationale = _get_field(first, "rationale")

    title = first_claim or PIPELINE_BRIEF_TITLE
    summary_source = first_rationale or first_claim
    summary = _preview(summary_source, max_len=280)

    sections: list[BriefSection] = []
    for idx, card in enumerate(cards, start=1):
        claim = _get_field(card, "claim")
        rationale = _get_field(card, "rationale")

        if not claim and not rationale:
            continue

        heading = claim or f"Explanation {idx}"
        body = rationale or claim

        sections.append(
            BriefSection(
                heading=heading,
                body=body,
                bullets=[],
            ),
        )

    brief = PublicBrief(
        kind="public_brief.v1",
        version="1.0.0",
        confidence=1.0,
        title=title,
        summary=summary,
        sections=sections,
    )
    return brief


def _unwrap_or_fail(label: str, result: Result[Any, str]) -> Any:
    """Unwrap a :class:`Result` or raise a RuntimeError with context.

    The pipeline is currently designed for "happy path" unit tests, so
    failures are surfaced as exceptions instead of being propagated as
    nested Result objects.
    """
    if result.is_ok():
        return result.unwrap()
    message = result.unwrap_err()
    raise RuntimeError(f"{label} failed: {message}")  # pragma: no cover - defensive


def run_pipeline(input_text: str, enable_history: bool = False) -> PipelineResult:
    """Run the full public-translation pipeline on ``input_text``.

    Parameters
    ----------
    input_text:
        Raw source text to be interpreted. Typically an academic abstract
        or a short policy document.
    enable_history:
        If True, the history agent will be invoked to generate
        :class:`TimelineEvent` objects and an evolution narrative.

    Returns
    -------
    PipelineResult
        A JSON-safe dictionary aggregating key artifacts and the shared
        :class:`Blackboard` produced during the run.
    """
    bb = Blackboard()

    # ------------------------------------------------------------------
    # Planner: build and record the execution strategy DAG.
    # ------------------------------------------------------------------
    plan = build_plan(enable_history=enable_history)
    bb.put("planner_dag", plan.to_payload())
    bb.trace("planner: public_translation plan")

    # ------------------------------------------------------------------
    # 1. Parse raw text into paragraphs.
    # ------------------------------------------------------------------
    bb.trace("pipeline: public_translation start")
    parsed_chunks = parser_agent(
        input_text,
        bb,
        key=_PARSED_CHUNKS_KEY,
        min_chars=1,
        make_trace=True,
    )
    bb.trace(f"pipeline: parser complete ({len(parsed_chunks)} chunks)")

    # ------------------------------------------------------------------
    # 2. Generate multi-layer explanations.
    # ------------------------------------------------------------------
    explainer_result = run_explainer(bb)
    explanation_cards = cast(
        "list[ExplanationCard]",
        _unwrap_or_fail("explainer", explainer_result),
    )
    bb.trace(f"pipeline: explainer complete ({len(explanation_cards)} cards)")

    # ------------------------------------------------------------------
    # 3. Generate citizen-facing relevance notes.
    # ------------------------------------------------------------------
    citizen_result = run_citizen(bb)
    relevance_notes = cast(
        "list[RelevanceNote]",
        _unwrap_or_fail("citizen", citizen_result),
    )
    bb.trace(f"pipeline: citizen complete ({len(relevance_notes)} notes)")

    # ------------------------------------------------------------------
    # 4. Extract jargon / terminology.
    # ------------------------------------------------------------------
    jargon_result = run_jargon(bb)
    term_cards = cast(
        "list[TermCard]",
        _unwrap_or_fail("jargon", jargon_result),
    )
    bb.trace(f"pipeline: jargon complete ({len(term_cards)} terms)")

    # ------------------------------------------------------------------
    # 5. History & timeline (optional).
    # ------------------------------------------------------------------
    timeline_events: list[TimelineEvent] = []
    if enable_history:
        history_result = run_history(bb)
        timeline_events = cast(
            "list[TimelineEvent]",
            _unwrap_or_fail("history", history_result),
        )
        bb.trace(
            f"pipeline: history complete ({len(timeline_events)} timeline events)",
        )
    else:
        # Make sure the key exists even if we skip the agent.
        bb.put(_TIMELINE_KEY, [])
        bb.trace("pipeline: history skipped (enable_history=False)")

    # ------------------------------------------------------------------
    # 6. Editor / validator gate.
    # ------------------------------------------------------------------
    editor_result = run_editor(bb)
    review_report = cast(
        "ReviewReport",
        _unwrap_or_fail("editor", editor_result),
    )
    bb.trace("pipeline: editor complete")

    # ------------------------------------------------------------------
    # 7. Structured brief + Markdown brief.
    # ------------------------------------------------------------------
    brief_model = _build_public_brief_from_explanations(explanation_cards)
    bb.put(_PUBLIC_BRIEF_KEY, brief_model)

    md_path_str: str | None = None
    try:
        md_result = run_brief_builder(
            bb,
            run_id="public-translation",
            reports_dir=None,
        )
        path = _unwrap_or_fail("brief_builder", md_result)
        md_path_str = str(path)
    except Exception:
        # Keep the pipeline result usable even if Markdown generation fails.
        md_path_str = None

    # Ensure the blackboard has the markdown path recorded as a string,
    # even if brief_builder already stored it.
    if md_path_str is not None:
        bb.put(_PUBLIC_BRIEF_MD_KEY, md_path_str)

    bb.trace("pipeline: public_translation complete")

    # ------------------------------------------------------------------
    # Assemble JSON-safe PipelineResult.
    # ------------------------------------------------------------------
    explanations_dicts = [_artifact_to_dict(c) for c in explanation_cards]
    relevance_dicts = [_artifact_to_dict(n) for n in _as_list(relevance_notes)]
    terms_dicts = [_artifact_to_dict(t) for t in term_cards]
    timeline_dicts = [_artifact_to_dict(ev) for ev in timeline_events]

    review_dict: dict[str, Any] | None
    if isinstance(review_report, Mapping):
        review_dict = dict(review_report)
    elif hasattr(review_report, "dict"):
        review_dict = review_report.dict()
    elif hasattr(review_report, "model_dump"):
        review_dict = review_report.model_dump()
    else:
        review_dict = None

    brief_payload = cast(
        PublicBriefPayload,
        brief_model.model_dump(),
    )

    result: PipelineResult = {
        "blackboard": bb,
        "parsed_chunks": parsed_chunks,
        "explanations": explanations_dicts,
        "relevance_notes": relevance_dicts,
        "terms": terms_dicts,
        "timeline_events": timeline_dicts,
        "review_report": review_dict,
        "public_brief": brief_payload,
        "public_brief_md_path": md_path_str,
    }
    return result


__all__ = [
    "run_pipeline",
    "PipelineResult",
    "PublicBriefPayload",
    "PIPELINE_BRIEF_TITLE",
]
