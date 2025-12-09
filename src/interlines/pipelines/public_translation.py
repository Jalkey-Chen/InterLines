# src/interlines/pipelines/public_translation.py
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

The pipeline is deliberately structured as a sequence of small, testable
steps. Each step:

- Reads from / writes to a shared :class:`Blackboard`.
- Can be stubbed in tests so CI does not rely on external LLM calls.
- Emits trace snapshots to support debugging and provenance.

High-level design
-----------------
The public-translation pipeline is intended to be:

- **Deterministic at the orchestration layer**:
  The "planner" decides which agents to invoke and in what order
  (now DAG-driven), while each agent is free to use non-deterministic
  sampling internally.

- **LLM-provider-agnostic**:
  Agents talk to the LLM client through well-defined contracts. This
  module does not depend on concrete model IDs.

- **Blackboard-centric**:
  All intermediate artifacts (parsed chunks, explanations, terms,
  timeline events, review reports, etc.) are stored on a central
  :class:`Blackboard` to simplify inspection, reproducibility, and
  downstream processing.

Entry point
-----------
The main entry point is :func:`run_pipeline`, which returns a
JSON-safe ``PipelineResult`` containing:

- The shared :class:`Blackboard` instance.
- Structured parser output: ``list[dict]``.
- Explanation, relevance, term, and timeline artifacts as dicts.
- A ``PublicBrief`` payload (converted to plain dict).
- A Markdown brief path (if brief-builder succeeds).

This version (Commit 2 ⋅ Step 5.1) replaces the previous hard-coded
execution order with a **DAG-driven orchestration layer**, enabling
dynamic replanning in future milestones (e.g., LLM-backed PlannerAgent).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict, cast

# Agent imports
from interlines.agents.brief_builder import (
    _PUBLIC_BRIEF_MD_KEY,
    run_brief_builder,
)
from interlines.agents.citizen_agent import run_citizen
from interlines.agents.editor_agent import run_editor
from interlines.agents.explainer_agent import run_explainer
from interlines.agents.history_agent import run_history
from interlines.agents.jargon_agent import run_jargon
from interlines.agents.parser_agent import parser_agent

# Core imports
from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.explanation import ExplanationCard
from interlines.core.contracts.public_brief import BriefSection, PublicBrief
from interlines.core.contracts.relevance import RelevanceNote
from interlines.core.contracts.review import ReviewReport
from interlines.core.contracts.term import TermCard
from interlines.core.contracts.timeline import TimelineEvent
from interlines.core.planner.dag import DAG
from interlines.core.planner.strategy import build_plan
from interlines.core.result import Result

# Blackboard keys
_PARSED_CHUNKS_KEY = "parsed_chunks"
_EXPLANATIONS_KEY = "explanations"
_RELEVANCE_NOTES_KEY = "relevance_notes"
_TERMS_KEY = "terms"
_TIMELINE_KEY = "timeline_events"
_PUBLIC_BRIEF_KEY = "public_brief"

PIPELINE_BRIEF_TITLE = "InterLines Public Brief"


# ---------------------------------------------------------------------------
# JSON-safe output payload types
# ---------------------------------------------------------------------------


class BriefSectionPayload(TypedDict):
    heading: str
    body: str
    bullets: list[str]


class PublicBriefPayload(TypedDict):
    title: str
    summary: str
    sections: list[BriefSectionPayload]


class PipelineResult(TypedDict):
    blackboard: Blackboard
    parsed_chunks: list[dict[str, Any]]
    explanations: list[dict[str, Any]]
    relevance_notes: list[dict[str, Any]]
    terms: list[dict[str, Any]]
    timeline_events: list[dict[str, Any]]
    review_report: dict[str, Any] | None
    public_brief: PublicBriefPayload
    public_brief_md_path: str | None


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _artifact_to_dict(obj: Any) -> dict[str, Any]:
    """
    Convert structured artifacts to plain Python dicts.

    The pipeline must be JSON-safe. Agent outputs may be:
    - Pydantic models (with ``model_dump``)
    - Dataclasses
    - Existing dicts
    - Unknown objects (fallback: repr)

    This helper centralizes the logic and keeps run_pipeline readable.
    """
    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        return cast(dict[str, Any], obj.model_dump())
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {"value": repr(obj)}


def _as_list(x: Any | Sequence[Any]) -> list[Any]:
    """Normalize possibly-scalar values into lists."""
    if x is None:
        return []
    if isinstance(x, Sequence) and not isinstance(x, str | bytes):
        return list(x)
    return [x]


def _unwrap_or_fail(label: str, result: Result[Any, Any]) -> Any:
    """
    Extract the success value from a :class:`Result`.

    If the agent failed, embed context so failures are easy to debug.
    """
    if result.is_ok():
        return result.unwrap()
    raise RuntimeError(f"{label} failed: {result.unwrap_err()}")


def _build_public_brief_from_explanations(cards: list[ExplanationCard]) -> PublicBrief:
    """
    Construct a :class:`PublicBrief` from explanation cards.

    This function is intentionally pure and testable. It mirrors what a
    future LLM agent *might* produce when assembling narrative structure.
    """
    raw = [_artifact_to_dict(c) for c in cards]

    # ---------------- Summary -----------------
    summary_parts = []
    for c in raw:
        snip = c.get("summary") or c.get("text")
        if snip:
            summary_parts.append(str(snip).strip())
    summary = "\n\n".join(summary_parts)

    # ---------------- Sections ----------------
    grouped: dict[str, list[dict[str, Any]]] = {}
    for c in raw:
        topic = str(c.get("topic") or "Main points")
        grouped.setdefault(topic, []).append(c)

    sections: list[BriefSection] = []
    for topic, group in grouped.items():
        bullets = [
            str(c.get("summary") or c.get("text") or "").strip()
            for c in group
            if (c.get("summary") or c.get("text"))
        ]
        sections.append(
            BriefSection(
                heading=topic,
                body="\n".join(bullets),
                bullets=bullets,
            )
        )

    # Artifact schema requires kind/version/confidence
    return PublicBrief(
        kind="public_brief",
        version="1.0.0",
        confidence=1.0,
        title=PIPELINE_BRIEF_TITLE,
        summary=summary,
        sections=sections,
    )


# ---------------------------------------------------------------------------
# DAG-driven pipeline entry point
# ---------------------------------------------------------------------------


# ruff: noqa: C901
def run_pipeline(input_text: str, enable_history: bool = False) -> PipelineResult:
    """
    Run the entire public-translation workflow under DAG orchestration.

    Steps (determined by planner):
        parse → translate → (timeline) → narrate → review → brief

    The DAG execution loop ensures the pipeline is:
    - extensible (new steps can be added without rewriting run_pipeline)
    - introspectable (planner_dag recorded in blackboard)
    - ready for LLM-driven dynamic replanning in future milestones.
    """
    bb = Blackboard()

    # ---------------- Planner -----------------
    plan_spec, _legacy = build_plan(enable_history)
    dag = DAG.from_plan_spec(plan_spec)
    bb.put("planner_dag", dag.to_payload())
    bb.trace("planner: public_translation plan")

    # ---------------- State -------------------
    parsed_chunks: list[dict[str, Any]] = []
    explanation_cards: list[ExplanationCard] = []
    relevance_notes: list[RelevanceNote] | list[dict[str, Any]] = []
    term_cards: list[TermCard] = []
    timeline_events: list[TimelineEvent] = []
    review_report: ReviewReport | dict[str, Any] | None = None
    brief_model: PublicBrief | None = None
    md_path: str | None = None

    if not plan_spec.enable_history:
        bb.put(_TIMELINE_KEY, [])
        bb.trace("pipeline: history disabled")

    # ----------------------------------------------------------------------
    # Execute DAG
    # ----------------------------------------------------------------------
    for step in dag.topological_order():
        # ---------------------- parse ----------------------
        if step == "parse":
            parsed_chunks = parser_agent(
                input_text,
                bb,
                key=_PARSED_CHUNKS_KEY,
                min_chars=1,
                make_trace=True,
            )
            bb.trace("planner: executed step parse")
            continue

        # -------------------- translate --------------------
        if step == "translate":
            explainer_res = run_explainer(bb)
            explanation_cards = cast(
                list[ExplanationCard],
                _unwrap_or_fail("explainer", explainer_res),
            )

            jargon_res = run_jargon(bb)
            term_cards = cast(
                list[TermCard],
                _unwrap_or_fail("jargon", jargon_res),
            )

            bb.trace("planner: executed step translate")
            continue

        # -------------------- timeline ---------------------
        if step == "timeline":
            hist_res = run_history(bb)
            timeline_events = cast(
                list[TimelineEvent],
                _unwrap_or_fail("history", hist_res),
            )
            bb.trace("planner: executed step timeline")
            continue

        # -------------------- narrate ----------------------
        if step == "narrate":
            citizen_res = run_citizen(bb)
            relevance_notes = cast(
                list[RelevanceNote],
                _unwrap_or_fail("citizen", citizen_res),
            )
            bb.trace("planner: executed step narrate")
            continue

        # -------------------- review -----------------------
        if step == "review":
            editor_res = run_editor(bb)
            review_report = cast(
                ReviewReport,
                _unwrap_or_fail("editor", editor_res),
            )
            bb.trace("planner: executed step review")
            continue

        # --------------------- brief -----------------------
        if step == "brief":
            brief_model = _build_public_brief_from_explanations(explanation_cards)
            bb.put(_PUBLIC_BRIEF_KEY, brief_model)

            try:
                md_res = run_brief_builder(bb, run_id="public-translation")
                path = _unwrap_or_fail("brief_builder", md_res)
                md_path = str(path)
                bb.put(_PUBLIC_BRIEF_MD_KEY, md_path)
            except Exception:
                md_path = None

            bb.trace("planner: executed step brief")
            continue

        # ------------------- Unknown -----------------------
        raise RuntimeError(f"Unknown planner step {step!r}")

    bb.trace("pipeline: public_translation complete")

    # ----------------------------------------------------------------------
    # Safe review_report conversion
    # ----------------------------------------------------------------------
    if review_report is None:
        review_dict = None
    elif isinstance(review_report, Mapping):
        review_dict = dict(review_report)
    elif hasattr(review_report, "model_dump"):
        review_dict = review_report.model_dump()
    else:
        review_dict = None

    # ----------------------------------------------------------------------
    # Safe brief payload conversion
    # ----------------------------------------------------------------------
    if brief_model is None:
        brief_payload: PublicBriefPayload = {
            "title": "",
            "summary": "",
            "sections": [],
        }
    else:
        brief_payload = cast(PublicBriefPayload, brief_model.model_dump())

    # ----------------------------------------------------------------------
    # Final JSON-safe result
    # ----------------------------------------------------------------------
    result: PipelineResult = {
        "blackboard": bb,
        "parsed_chunks": parsed_chunks,
        "explanations": [_artifact_to_dict(c) for c in explanation_cards],
        "relevance_notes": [_artifact_to_dict(r) for r in _as_list(relevance_notes)],
        "terms": [_artifact_to_dict(t) for t in term_cards],
        "timeline_events": [_artifact_to_dict(t) for t in timeline_events],
        "review_report": review_dict,
        "public_brief": brief_payload,
        "public_brief_md_path": md_path,
    }
    return result


__all__ = [
    "run_pipeline",
    "PipelineResult",
    "PublicBriefPayload",
    "PIPELINE_BRIEF_TITLE",
]
