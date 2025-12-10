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
   - It executes agents in topological order:
     raw text → parser → explainer → citizen/jargon → history → editor.

2. **Phase 2: Intelligent Replan (Step 5.3)**
   - After the Editor runs, the Planner inspects the ``ReviewReport``.
   - If quality scores (readability, factuality) are below thresholds, the
     Planner triggers a **Single-Round Replan**.
   - A new, transient DAG is constructed containing only the necessary
     refinement steps (e.g., ``explainer_refine``, ``citizen_refine``).
   - These steps are executed to update the artifacts on the blackboard.

3. **Phase 3: Reporting & Assembly (Step 5.4 & 6.1)**
   - A structured :class:`PlanReport` is generated to summarize the decisions.
   - The **Brief Builder** (LLM) is invoked to generate the final Markdown report.
   - The final brief object is assembled for the API response.

Design Principles
-----------------
- **Deterministic Orchestration**: The pipeline code controls *when* agents
  run, but agents control *what* they generate.
- **Blackboard-Centric**: All state is shared via :class:`Blackboard`, allowing
  the Replan phase to read outputs from the Initial phase.
- **Traceability**: Every step, including replan decisions and final reports,
  emits a trace snapshot for debugging.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict, cast

# --- Agent Imports ---
from interlines.agents.brief_builder import run_brief_builder
from interlines.agents.citizen_agent import run_citizen
from interlines.agents.editor_agent import run_editor
from interlines.agents.explainer_agent import run_explainer
from interlines.agents.history_agent import run_history
from interlines.agents.jargon_agent import run_jargon
from interlines.agents.parser_agent import parser_agent
from interlines.agents.planner_agent import PlannerAgent, PlannerContext

# --- Core Infrastructure ---
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

# Keys used to stash artifacts on the blackboard. These are intentionally
# simple strings so that they are easy to inspect in traces and tests.
_PARSED_CHUNKS_KEY = "parsed_chunks"
_EXPLANATIONS_KEY = "explanations"
_RELEVANCE_NOTES_KEY = "relevance_notes"
_TERMS_KEY = "terms"
_TIMELINE_KEY = "timeline_events"
_REVIEW_REPORT_KEY = "review_report"
_PUBLIC_BRIEF_KEY = "public_brief"
_PUBLIC_BRIEF_MD_KEY = "public_brief_md_path"

# Planner provenance keys for observability
_PLANNER_PLAN_KEY = "planner_plan_spec.initial"
_PLANNER_REPLAN_KEY = "planner_plan_spec.replan"
_PLANNER_REPORT_KEY = "planner_report"
_PLANNER_DAG_KEY = "planner_dag"

# Fixed (RUF001): Used hyphen-minus instead of ambiguous en-dash.
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

    This mirrors what a future HTTP API or CLI might marshal into JSON.

    Attributes
    ----------
    blackboard:
        The full blackboard state (for debugging/tracing).
    public_brief_md_path:
        The filesystem path to the generated Markdown report (Primary Deliverable).
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
    """Convert a Pydantic model or plain mapping into a regular ``dict``.

    The pipeline frequently stores artifacts on the blackboard and inside
    :class:`PipelineResult`. To keep serialization simple and JSON-safe, we
    aggressively convert models to plain dictionaries.

    Supported inputs
    ----------------
    - Pydantic v2 models (with ``model_dump``).
    - ``dict`` instances (returned as-is).
    - Any object with a ``dict()``-style method (best-effort call).
    """
    if obj is None:
        return {}

    if hasattr(obj, "model_dump"):
        # Fixed (Mypy): explicit cast for stricter return type checking.
        return cast("dict[str, Any]", obj.model_dump())  # pydantic v2

    if isinstance(obj, dict):
        # Shallow copy so callers do not accidentally mutate our state.
        return dict(obj)

    if hasattr(obj, "dict"):
        # Fixed (Mypy): explicit cast.
        return cast("dict[str, Any]", obj.dict())

    # Very defensive fallback: make a tiny best-effort dict.
    return {"value": obj}


def _as_list(x: Any) -> list[Any]:
    """Normalize ``x`` into a list.

    - ``None`` → ``[]``
    - sequence types (except ``str``/``bytes``) → list(x)
    - everything else → ``[x]``
    """
    if x is None:
        return []
    # Fixed (UP038): Use `|` union syntax for isinstance checks.
    if isinstance(x, Sequence) and not isinstance(x, str | bytes):
        return list(x)
    return [x]


def _unwrap_or_fail(stage: str, result: Result[Any, str]) -> Any:
    """Extract the value from a :class:`Result` or raise a RuntimeError.

    In the current prototype we treat failures as hard errors rather than
    trying to repair mid-pipeline. This makes bugs surface quickly in tests
    and keeps the control flow readable.
    """
    if result.is_ok():
        return result.unwrap()
    # Include the failing stage in the error message to ease debugging.
    raise RuntimeError(f"{stage} failed: {result.unwrap_err()}")


def _build_public_brief_fallback(
    cards: list[ExplanationCard],
) -> PublicBrief:
    """
    Construct a fallback in-memory PublicBrief object.

    This is used to populate the API JSON response (`public_brief`).
    Note that the *real* synthesis logic now lives in the `brief_builder`
    agent (which writes Markdown), but we still need a structured object
    for the frontend to consume immediately.
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
        # --- FIX: Explicitly populate missing fields ---
        # We satisfy Pylance strictness by providing all fields, even if empty.
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
    Execute a single logical step in the pipeline.

    This helper centralizes the dispatch logic, mapping abstract DAG node
    labels (e.g., "translate", "explainer_refine") to concrete Agent calls.
    It supports both the **Initial Phase** and the **Replan Phase**.

    Parameters
    ----------
    step:
        Logical step name from the Planner's ``steps`` or ``replan_steps``.
    input_data:
        Raw input text OR file path (passed to the Parser Agent).
    bb:
        Shared blackboard instance.
    worker_llm:
        Optional LLM client for agents that require it.
    """
    if step == "parse":
        # Parser uses the "balanced" model to intelligently split PDFs.
        parsed_chunks_raw = parser_agent(input_data, bb, llm=worker_llm)
        bb.put(_PARSED_CHUNKS_KEY, parsed_chunks_raw)

    elif step in ("translate", "explainer_refine"):
        # "translate" = Initial Pass; "explainer_refine" = Replan Pass
        # Both use the same agent, which reads/writes "explanations"
        explainer_res = run_explainer(bb)
        cards = _unwrap_or_fail("explainer_agent", explainer_res)
        bb.put(_EXPLANATIONS_KEY, [_artifact_to_dict(c) for c in cards])
        bb.trace(f"step '{step}' finished")

    elif step in ("narrate", "citizen_refine"):
        # "narrate" = Initial Pass; "citizen_refine" = Replan Pass
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
        # "review" = Initial Pass; "editor" = Re-verification Pass
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
        # We don't read the full file just for the planner prompt context.
    elif isinstance(input_data, str):
        if len(input_data) < 256 and Path(input_data).exists():
            # Heuristic: if string looks like a path, treat it as one.
            p = Path(input_data)
            doc_kind = p.suffix.lstrip(".")
        else:
            # Raw text
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

    Logic:
    1. Check if a ReviewReport exists on the blackboard.
    2. Ask the PlannerAgent to evaluate it.
    3. If `should_replan` is True, construct a transient DAG.
    4. Execute the refinement DAG.

    Returns
    -------
    PlannerPlanSpec | None
        The replan specification if refinement occurred, else None.
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
            # If report is malformed, skip refinement safely
            return None

    if not isinstance(report_data, ReviewReport):
        return None

    # Ask Planner for decision
    replan_spec = planner.replan(bb, ctx, plan_spec, report_data)

    if replan_spec.should_replan and replan_spec.replan_steps:
        bb.put(_PLANNER_REPLAN_KEY, replan_spec.model_dump())

        # --- CRITICAL TRACE ---
        # This trace is required by E2E tests to verify replan activation.
        bb.trace(f"planner: triggering replan -> {replan_spec.replan_steps}")

        # Execute Refinement DAG
        refine_dag = DAG.from_plan_spec(
            plan_spec.model_copy(update={"steps": replan_spec.replan_steps})
        )
        _execute_dag(refine_dag, input_data, bb, worker_llm)

        # Force re-run of brief builder if not explicitly planned
        # This ensures the final Markdown file reflects the refined content.
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
    DAG-driven pipeline entry point with Single-Round Replanning (Step 5.3).

    This function orchestrates the full lifecycle of a translation request:
    1. **Plan**: Consults the Planner (LLM or Rule) to get an execution plan.
    2. **Execute**: Runs the initial DAG (Phase 1).
    3. **Evaluate**: Checks the ``ReviewReport`` generated by the Editor.
    4. **Replan (Optional)**: If quality is low, asks Planner for refinement steps.
    5. **Refine**: Executes the refinement DAG (Phase 2).
    6. **Assemble**: Collects final artifacts into a result payload.

    Parameters
    ----------
    input_data:
        Raw input text OR a pathlib.Path to a file (PDF, DOCX, TXT).
        The ParserAgent will handle file extraction automatically.
    enable_history:
        User preference for including the history/timeline branch.
    use_llm_planner:
        If ``True``, uses :class:`PlannerAgent` for routing and replanning.
        If ``False``, uses legacy rule-based planning (no replan capability).

    Returns
    -------
    PipelineResult
        A dictionary containing the Blackboard state, artifacts, and outputs.
    """
    # =========================================================================
    # 0. Initialization
    # =========================================================================
    bb = Blackboard()
    # Store input for observability.
    bb.put("input_data", str(input_data))

    # Initialize a shared LLM client for agents that need it (Parser, Planner).
    worker_llm = LLMClient.from_env(default_model_alias="planner")

    # =========================================================================
    # 1. Phase 1: Initial Planning
    # =========================================================================
    ctx = _create_planner_context(input_data, enable_history)
    planner: PlannerAgent | None = None

    if use_llm_planner:
        # Use LLM-backed planner for dynamic routing
        planner = PlannerAgent(llm=worker_llm)
        plan_spec = planner.plan(bb, ctx)
    else:
        # Fallback to deterministic rules (Legacy Step 5.1)
        plan_spec, _ = build_plan(enable_history)

    dag = DAG.from_plan_spec(plan_spec)

    # Persist plan provenance
    bb.put(_PLANNER_PLAN_KEY, plan_spec.model_dump())
    bb.put(_PLANNER_DAG_KEY, dag.to_payload())
    # Ensure this trace matches E2E test expectations exactly
    bb.trace("planner: phase 1 plan")

    # =========================================================================
    # 2. Phase 1: Execution (Initial Pass)
    # =========================================================================
    _execute_dag(dag, input_data, bb, worker_llm)

    # =========================================================================
    # 3. Phase 2: Intelligent Replan (Step 5.3)
    # =========================================================================
    replan_spec: PlannerPlanSpec | None = None
    if use_llm_planner and planner:
        replan_spec = _attempt_refinement(bb, planner, ctx, plan_spec, input_data, worker_llm)

    # =========================================================================
    # 4. Phase 3: Reporting & Final Assembly (Step 5.4)
    # =========================================================================
    # Generate the executive summary of the planner's decisions.
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
    # --- FIX: Add the trace that tests were failing on ---
    bb.trace("planner: report written")

    # Build final API payload
    cards = cast("list[ExplanationCard]", _as_list(bb.get(_EXPLANATIONS_KEY)))
    brief_model = _build_public_brief_fallback(cards)
    brief_dict = _artifact_to_dict(brief_model)

    # Retrieve the path generated by brief_builder
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
