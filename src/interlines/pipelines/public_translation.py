"""
Public translation pipeline: from raw text (or file) to public-facing artifacts.

Milestone
---------
M5 | Planner Agent
Step 5.3 | Integrated Single-Round Replan
Step 5.4 | Planner Report & Richer Trace
M6 | Interface & Deployment
Step 6.1 | Integrated Brief Builder (LLM-backed)

This module orchestrates the main InterLines "public translation" flow using a
DAG-driven approach with an optional, intelligent refinement loop.

Flow Overview
-------------
1. **Phase 1: Initial Execution**
   - The pipeline consults the Planner (LLM or Rule-based) to build a DAG.
   - It executes agents in topological order.

2. **Phase 2: Intelligent Replan (Step 5.3)**
   - After the Editor runs, the Planner inspects the ``ReviewReport``.
   - If quality scores are low, a transient refinement DAG is executed.

3. **Phase 3: Final Assembly**
   - Invokes the LLM-backed Brief Builder to generate a Markdown report.
   - Assembles structured JSON for API responses.

Design Principles
-----------------
- **Deterministic Orchestration**: The pipeline code controls *when* agents run.
- **Blackboard-Centric**: All state is shared via :class:`Blackboard`.
- **Traceability**: Every step emits a trace snapshot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict, cast

# Agents
from interlines.agents.brief_builder import run_brief_builder
from interlines.agents.citizen_agent import run_citizen
from interlines.agents.editor_agent import run_editor
from interlines.agents.explainer_agent import run_explainer
from interlines.agents.history_agent import run_history
from interlines.agents.jargon_agent import run_jargon
from interlines.agents.parser_agent import parser_agent
from interlines.agents.planner_agent import PlannerAgent, PlannerContext

# Core & Infrastructure
from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.explanation import ExplanationCard
from interlines.core.contracts.planner import PlannerPlanSpec, PlanReport
from interlines.core.contracts.public_brief import BriefSection, PublicBrief
from interlines.core.contracts.review import ReviewReport
from interlines.core.planner.dag import DAG
from interlines.core.planner.strategy import build_plan
from interlines.core.result import Result
from interlines.llm.client import LLMClient

# --------------------------------------------------------------------------- #
# Blackboard keys and constants
# --------------------------------------------------------------------------- #

_PARSED_CHUNKS_KEY = "parsed_chunks"
_EXPLANATIONS_KEY = "explanations"
_RELEVANCE_NOTES_KEY = "relevance_notes"
_TERMS_KEY = "terms"
_TIMELINE_KEY = "timeline_events"
_REVIEW_REPORT_KEY = "review_report"
_PUBLIC_BRIEF_KEY = "public_brief"
_PUBLIC_BRIEF_MD_KEY = "public_brief_md_path"

_PLANNER_PLAN_KEY = "planner_plan_spec.initial"
_PLANNER_REPLAN_KEY = "planner_plan_spec.replan"
_PLANNER_REPORT_KEY = "planner_report"
_PLANNER_DAG_KEY = "planner_dag"

PIPELINE_BRIEF_TITLE = "InterLines - Public Brief"


# --------------------------------------------------------------------------- #
# Public result types
# --------------------------------------------------------------------------- #


class PublicBriefPayload(TypedDict):
    """JSON-safe subset of :class:`PublicBrief` exposed by the pipeline API."""

    title: str
    summary: str
    sections: list[Mapping[str, object]]


class PipelineResult(TypedDict):
    """Structured payload returned by :func:`run_pipeline`.

    Attributes
    ----------
    public_brief_md_path : str | None
        The filesystem path to the generated Markdown report.
    """

    blackboard: Blackboard
    parsed_chunks: list[Mapping[str, object]]
    explanations: list[Mapping[str, object]]
    relevance_notes: list[Mapping[str, object]]
    terms: list[Mapping[str, object]]
    timeline_events: list[Mapping[str, object]]
    public_brief: PublicBriefPayload
    public_brief_md_path: str | None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _artifact_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a Pydantic model or mapping into a plain dict."""
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return cast("dict[str, Any]", obj.model_dump())
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "dict"):
        return cast("dict[str, Any]", obj.dict())
    return {"value": obj}


def _as_list(x: Any) -> list[Any]:
    """Normalize a value into a list."""
    if x is None:
        return []
    if isinstance(x, Sequence) and not isinstance(x, str | bytes):
        return list(x)
    return [x]


def _unwrap_or_fail(stage: str, result: Result[Any, str]) -> Any:
    """Unwrap a Result or raise a RuntimeError with context."""
    if result.is_ok():
        return result.unwrap()
    raise RuntimeError(f"{stage} failed: {result.unwrap_err()}")


def _build_public_brief_fallback(
    cards: list[ExplanationCard],
) -> PublicBrief:
    """
    Construct a fallback in-memory PublicBrief object for the API JSON response.

    This ensures the 'public_brief' key in the output is populated even though
    the actual report generation happens in the BriefBuilder (Markdown).
    """
    raw = [_artifact_to_dict(c) for c in cards]

    # Simple synthesis logic for the JSON payload
    summary_parts = []
    for c in raw:
        text = c.get("summary") or c.get("text") or c.get("claim")
        if text:
            summary_parts.append(str(text).strip())

    summary = "\n\n".join(summary_parts[:3])  # Limit summary length

    sections: list[BriefSection] = []
    if raw:
        sections.append(
            BriefSection(
                heading="Key Insights",
                body="Generated from expert explanations.",
                bullets=[str(c.get("claim", "")) for c in raw],
            )
        )

    return PublicBrief(
        kind="public_brief.v1",
        version="1.0.0",
        confidence=1.0,
        title=PIPELINE_BRIEF_TITLE,
        summary=summary,
        sections=sections,
        # Explicitly populate required fields with safe defaults to satisfy Pydantic
        explanations=cards,
        timeline=[],
        evolution_narrative=None,
        glossary=[],
        relevance_notes=[],
        review_report=None,
    )


def _execute_step(
    step: str,
    input_data: str | Path,
    bb: Blackboard,
    worker_llm: LLMClient | None = None,
) -> None:
    """
    Dispatch logic for individual pipeline steps.

    Uses distinct variables for results to satisfy Mypy type checking.
    """
    if step == "parse":
        parsed_chunks_raw = parser_agent(input_data, bb, llm=worker_llm)
        bb.put(_PARSED_CHUNKS_KEY, parsed_chunks_raw)

    elif step in ("translate", "explainer_refine"):
        # Explicit variable name to avoid generic type recycling errors (Mypy).
        explainer_res = run_explainer(bb)
        cards = _unwrap_or_fail("explainer_agent", explainer_res)
        bb.put(_EXPLANATIONS_KEY, [_artifact_to_dict(c) for c in cards])
        bb.trace(f"step '{step}' finished")

    elif step in ("narrate", "citizen_refine"):
        citizen_res = run_citizen(bb)
        notes = _unwrap_or_fail("citizen_agent", citizen_res)
        bb.put(_RELEVANCE_NOTES_KEY, [_artifact_to_dict(n) for n in notes])
        bb.trace(f"step '{step}' finished")

    elif step in ("jargon", "jargon_refine"):
        jargon_res = run_jargon(bb)
        terms = _unwrap_or_fail("jargon_agent", jargon_res)
        bb.put(_TERMS_KEY, [_artifact_to_dict(t) for t in terms])

    elif step in ("timeline", "history_refine"):
        history_res = run_history(bb)
        events = _unwrap_or_fail("history_agent", history_res)
        bb.put(_TIMELINE_KEY, [_artifact_to_dict(e) for e in events])

    elif step in ("review", "editor"):
        editor_res = run_editor(bb)
        _unwrap_or_fail("editor_agent", editor_res)
        bb.trace(f"step '{step}' finished")

    elif step == "brief":
        # Final Sink: Generate Markdown Report
        brief_res = run_brief_builder(bb, run_id="latest")
        path = _unwrap_or_fail("brief_builder_agent", brief_res)
        print(f"   [Pipeline] Brief generated at: {path}")
        bb.trace("step 'brief' finished")

    else:
        print(f"   [WARN] Unknown step '{step}', skipping.")


def _create_planner_context(input_data: str | Path, enable_history: bool) -> PlannerContext:
    """Determine document metadata and create the planner context."""
    doc_kind = "text"
    char_count = 0

    if isinstance(input_data, Path):
        doc_kind = input_data.suffix.lstrip(".")
        # We don't read file size here to keep it fast
    elif isinstance(input_data, str):
        if len(input_data) < 256 and Path(input_data).exists():
            p = Path(input_data)
            doc_kind = p.suffix.lstrip(".")
        else:
            char_count = len(input_data)

    return PlannerContext(
        task_type="public_translation",
        document_kind=doc_kind,
        approx_char_count=char_count,
        language="en",
        enable_history_requested=enable_history,
    )


def _execute_dag(
    dag: DAG,
    input_data: str | Path,
    bb: Blackboard,
    worker_llm: LLMClient | None,
) -> None:
    """Execute the steps in a DAG in topological order."""
    for step in dag.topological_order():
        if step == "narrate":
            # Legacy grouping: narrate often implies jargon + citizen
            _execute_step("narrate", input_data, bb, worker_llm)
            _execute_step("jargon", input_data, bb, worker_llm)
        else:
            _execute_step(step, input_data, bb, worker_llm)


def _attempt_refinement(
    bb: Blackboard,
    planner: PlannerAgent,
    ctx: PlannerContext,
    plan_spec: PlannerPlanSpec,
    input_data: str | Path,
    worker_llm: LLMClient | None,
) -> PlannerPlanSpec | None:
    """
    Inspect the ReviewReport and trigger refinement if needed.

    Returns the replan_spec if refinement occurred, else None.
    """
    review_report_raw = bb.get(_REVIEW_REPORT_KEY)
    if not review_report_raw:
        return None

    # Ensure report is a Pydantic object
    report_data = review_report_raw
    if isinstance(report_data, dict):
        try:
            report_data = ReviewReport(**report_data)
        except Exception:
            return None

    if not isinstance(report_data, ReviewReport):
        return None

    # Ask Planner for decision
    replan_spec = planner.replan(bb, ctx, plan_spec, report_data)

    if replan_spec.should_replan and replan_spec.replan_steps:
        bb.put(_PLANNER_REPLAN_KEY, replan_spec.model_dump())

        # Execute Refinement DAG
        refine_dag = DAG.from_plan_spec(
            plan_spec.model_copy(update={"steps": replan_spec.replan_steps})
        )
        _execute_dag(refine_dag, input_data, bb, worker_llm)

        # Force re-run of brief builder if not explicitly planned
        if "brief" not in replan_spec.replan_steps:
            _execute_step("brief", input_data, bb, worker_llm)

        return replan_spec

    return None


# --------------------------------------------------------------------------- #
# Main Entry Point
# --------------------------------------------------------------------------- #


def run_pipeline(
    input_data: str | Path,
    enable_history: bool = False,
    use_llm_planner: bool = True,
) -> PipelineResult:
    """
    Orchestrate the Public Translation Pipeline (Expert -> Citizen).

    This pipeline transforms raw text into a structured JSON brief and a
    polished Markdown report. Refactored to reduce cyclomatic complexity.

    Parameters
    ----------
    input_data : str | Path
        Source content (raw string or file path).
    enable_history : bool
        If True, enables the History agent branch (timeline generation).
    use_llm_planner : bool
        If True, uses the LLM-backed PlannerAgent for dynamic routing and
        Refinement loops.

    Returns
    -------
    PipelineResult
        A dictionary containing the Blackboard state, artifacts, and outputs.
    """
    # 1. Setup
    bb = Blackboard()
    bb.put("input_data", str(input_data))
    worker_llm = LLMClient.from_env(default_model_alias="planner")

    # 2. Plan (Phase 1)
    ctx = _create_planner_context(input_data, enable_history)
    planner: PlannerAgent | None = None

    if use_llm_planner:
        planner = PlannerAgent(llm=worker_llm)
        plan_spec = planner.plan(bb, ctx)
    else:
        plan_spec, _ = build_plan(enable_history)

    dag = DAG.from_plan_spec(plan_spec)
    bb.put(_PLANNER_PLAN_KEY, plan_spec.model_dump())
    bb.put(_PLANNER_DAG_KEY, dag.to_payload())
    bb.trace("phase 1: plan ready")

    # 3. Execute (Phase 1)
    _execute_dag(dag, input_data, bb, worker_llm)

    # 4. Review & Refine (Phase 2)
    replan_spec: PlannerPlanSpec | None = None
    if use_llm_planner and planner:
        replan_spec = _attempt_refinement(bb, planner, ctx, plan_spec, input_data, worker_llm)

    # 5. Reporting & Assembly (Phase 3)
    plan_report = PlanReport(
        strategy=plan_spec.strategy,
        enable_history=plan_spec.enable_history,
        initial_steps=plan_spec.steps,
        refine_used=bool(replan_spec and replan_spec.should_replan),
        replan_steps=replan_spec.replan_steps if replan_spec else None,
        replan_reason=replan_spec.replan_reason if replan_spec else None,
        notes=plan_spec.notes,
    )
    bb.put(_PLANNER_REPORT_KEY, plan_report.model_dump())

    # Build final API payload
    cards = cast("list[ExplanationCard]", _as_list(bb.get(_EXPLANATIONS_KEY)))
    brief_model = _build_public_brief_fallback(cards)
    brief_dict = _artifact_to_dict(brief_model)

    md_path = cast(str | None, bb.get(_PUBLIC_BRIEF_MD_KEY))
    bb.put(_PUBLIC_BRIEF_KEY, brief_dict)
    bb.trace("pipeline: complete")

    return {
        "blackboard": bb,
        "parsed_chunks": _as_list(bb.get(_PARSED_CHUNKS_KEY)),
        "explanations": _as_list(bb.get(_EXPLANATIONS_KEY)),
        "relevance_notes": _as_list(bb.get(_RELEVANCE_NOTES_KEY)),
        "terms": _as_list(bb.get(_TERMS_KEY)),
        "timeline_events": _as_list(bb.get(_TIMELINE_KEY)),
        "public_brief": {
            "title": cast(str, brief_dict.get("title", "")),
            "summary": cast(str, brief_dict.get("summary", "")),
            "sections": cast(list[Mapping[str, object]], brief_dict.get("sections", [])),
        },
        "public_brief_md_path": md_path,
    }


__all__ = ["run_pipeline", "PipelineResult", "PublicBriefPayload"]
