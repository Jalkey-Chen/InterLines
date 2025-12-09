"""
Public translation pipeline: from raw text to public-facing artifacts.

Milestone
---------
M5 | Planner Agent
Step 5.3 | Integrated Single-Round Replan
Step 5.4 | Planner Report & Richer Trace

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

3. **Phase 3: Reporting & Assembly (Step 5.4)**
   - A structured :class:`PlanReport` is generated to summarize the decisions
     (strategy used, whether replan happened, why).
   - This report is written to the blackboard for observability.
   - The final brief is assembled and returned.

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
from typing import Any, TypedDict, cast

from interlines.agents.brief_builder import run_brief_builder
from interlines.agents.citizen_agent import run_citizen
from interlines.agents.editor_agent import run_editor
from interlines.agents.explainer_agent import run_explainer
from interlines.agents.history_agent import run_history
from interlines.agents.jargon_agent import run_jargon
from interlines.agents.parser_agent import parser_agent
from interlines.agents.planner_agent import PlannerAgent, PlannerContext
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

# Planner provenance keys
_PLANNER_PLAN_KEY = "planner_plan_spec.initial"
_PLANNER_REPLAN_KEY = "planner_plan_spec.replan"
_PLANNER_REPORT_KEY = "planner_report"  # New in Step 5.4
_PLANNER_DAG_KEY = "planner_dag"

# Fixed (RUF001): Used hyphen-minus instead of ambiguous en-dash.
PIPELINE_BRIEF_TITLE = "InterLines - Public Brief"


# --------------------------------------------------------------------------- #
# Public result types
# --------------------------------------------------------------------------- #


class PublicBriefPayload(TypedDict):
    """JSON-safe subset of :class:`PublicBrief` exposed by the pipeline."""

    title: str
    summary: str
    sections: list[Mapping[str, object]]


class PipelineResult(TypedDict):
    """Structured payload returned by :func:`run_pipeline`.

    This mirrors what a future HTTP API or CLI might marshal into JSON.
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

    This is mainly used when reading from the blackboard, where callers may
    have stored either a single artifact or a collection.
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

    Parameters
    ----------
    stage:
        Human-readable label of the agent or step, used only for error
        messages and traces.
    result:
        A :class:`Result` returned by one of the agents.
    """
    if result.is_ok():
        return result.unwrap()
    # Include the failing stage in the error message to ease debugging.
    raise RuntimeError(f"{stage} failed: {result.unwrap_err()}")


def _build_public_brief_from_explanations(
    cards: list[ExplanationCard],
) -> PublicBrief:
    """
    Construct a :class:`PublicBrief` from explanation cards (pure logic).

    This helper synthesizes a structured brief object even if the
    ``brief_builder`` agent (which writes Markdown to disk) was not invoked
    or if we just need the in-memory representation for the API response.

    The logic is deliberately simple:

    - Use ``summary`` or ``text`` from the explanation cards to form a
      multi-paragraph summary.
    - Group cards by a coarse ``topic`` field (if present).
    - Turn each group into a :class:`BriefSection` with bullet-style lines.
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

    return PublicBrief(
        kind="public_brief.v1",
        version="1.0.0",
        confidence=1.0,
        title=PIPELINE_BRIEF_TITLE,
        summary=summary,
        sections=sections,
    )


def _execute_step(step: str, input_text: str, bb: Blackboard) -> None:
    """
    Execute a single logical step in the pipeline.

    This helper centralizes the dispatch logic, mapping abstract DAG node
    labels (e.g., "translate", "explainer_refine") to concrete Agent calls.
    It supports both the **Initial Phase** and the **Replan Phase**.

    Parameters
    ----------
    step:
        Logical step name from the Planner's ``steps`` or ``replan_steps``.
    input_text:
        Raw input text (only used by the Parser).
    bb:
        Shared blackboard instance.
    """
    if step == "parse":
        parsed_chunks_raw = parser_agent(input_text, bb)
        bb.put(_PARSED_CHUNKS_KEY, parsed_chunks_raw)

    elif step in ("translate", "explainer_refine"):
        # "translate" = Initial Pass
        # "explainer_refine" = Replan Pass
        # Both use the same agent, which reads/writes "explanations"
        explainer_result = run_explainer(bb)
        cards = _unwrap_or_fail("explainer_agent", explainer_result)
        # Explicit write-back ensures clarity, though agents usually do it.
        bb.put(_EXPLANATIONS_KEY, [_artifact_to_dict(c) for c in cards])

    elif step in ("narrate", "citizen_refine"):
        # "narrate" = Initial Pass
        # "citizen_refine" = Replan Pass
        citizen_result = run_citizen(bb)
        notes = _unwrap_or_fail("citizen_agent", citizen_result)
        bb.put(_RELEVANCE_NOTES_KEY, [_artifact_to_dict(n) for n in notes])

    elif step in ("jargon", "jargon_refine"):
        # Note: In legacy DAGs, "jargon" is often grouped with "narrate",
        # but the Planner treats them as distinct logical capabilities.
        jargon_result = run_jargon(bb)
        terms = _unwrap_or_fail("jargon_agent", jargon_result)
        bb.put(_TERMS_KEY, [_artifact_to_dict(t) for t in terms])

    elif step in ("timeline", "history_refine"):
        history_result = run_history(bb)
        events = _unwrap_or_fail("history_agent", history_result)
        bb.put(_TIMELINE_KEY, [_artifact_to_dict(ev) for ev in events])

    elif step == "review":
        # Initial review pass
        _res = run_editor(bb)
        _unwrap_or_fail("editor_agent", _res)

    elif step == "editor":
        # Re-review pass during refinement (verifies fixes)
        _res = run_editor(bb)
        _unwrap_or_fail("editor_agent (replan)", _res)

    elif step == "brief":
        # Fixed (Mypy): Use distinct variable to avoid type conflict with _res above.
        _brief_res = run_brief_builder(bb)
        _unwrap_or_fail("brief_builder_agent", _brief_res)


# --------------------------------------------------------------------------- #
# DAG-driven pipeline entry point
# --------------------------------------------------------------------------- #


def run_pipeline(  # noqa: C901
    input_text: str,
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
    input_text:
        Raw input text to be parsed and translated.
    enable_history:
        User preference for including the history/timeline branch.
    use_llm_planner:
        If ``True``, uses :class:`PlannerAgent` for routing and replanning.
        If ``False``, uses legacy rule-based planning (no replan capability).
    """
    # =========================================================================
    # 0. Initialization
    # =========================================================================
    bb = Blackboard()
    bb.put("input_text", input_text)

    # =========================================================================
    # 1. Phase 1: Initial Planning
    # =========================================================================
    ctx = PlannerContext(
        task_type="public_translation",
        document_kind="generic",
        approx_char_count=len(input_text),
        language="en",
        enable_history_requested=enable_history,
    )
    planner_agent: PlannerAgent | None = None

    if use_llm_planner:
        # Use LLM-backed planner for dynamic routing
        planner_agent = PlannerAgent(llm=LLMClient.from_env(default_model_alias="planner"))
        plan_spec = planner_agent.plan(bb, ctx)
    else:
        # Fallback to deterministic rules (Legacy Step 5.1)
        plan_spec, _ = build_plan(enable_history=enable_history)

    dag = DAG.from_plan_spec(plan_spec)

    # Persist plan provenance
    bb.put(_PLANNER_PLAN_KEY, plan_spec.model_dump())
    bb.put(_PLANNER_DAG_KEY, dag.to_payload())
    bb.trace("planner: phase 1 plan")

    # =========================================================================
    # 2. Phase 1: Execution (Initial Pass)
    # =========================================================================
    for step in dag.topological_order():
        # Special handling for legacy "narrate" step which implied jargon+citizen
        if step == "narrate":
            _execute_step("narrate", input_text, bb)
            _execute_step("jargon", input_text, bb)
        else:
            _execute_step(step, input_text, bb)

    # =========================================================================
    # 3. Phase 2: Intelligent Replan (Step 5.3)
    # =========================================================================
    review_report_raw = bb.get(_REVIEW_REPORT_KEY)
    replan_spec: PlannerPlanSpec | None = None

    if use_llm_planner and planner_agent and review_report_raw:
        # Reconstruct ReviewReport object for the Planner
        report_obj = None
        if isinstance(review_report_raw, dict):
            try:
                report_obj = ReviewReport(**review_report_raw)
            except Exception:
                pass
        elif isinstance(review_report_raw, ReviewReport):
            report_obj = review_report_raw

        if report_obj:
            # Ask Planner: "Given this report, do we need to fix anything?"
            replan_spec = planner_agent.replan(bb, ctx, plan_spec, report_obj)

            if replan_spec.should_replan and replan_spec.replan_steps:
                bb.trace(f"planner: triggering replan -> {replan_spec.replan_steps}")

                # Store the replan decision for transparency
                bb.put(_PLANNER_REPLAN_KEY, replan_spec.model_dump())

                # Construct a transient DAG for just the refinement steps
                replan_subset_spec = plan_spec.model_copy(
                    update={"steps": replan_spec.replan_steps}
                )
                replan_dag = DAG.from_plan_spec(replan_subset_spec)

                # Execute Refinement Steps
                for step in replan_dag.topological_order():
                    _execute_step(step, input_text, bb)

                # If "brief" wasn't explicitly in the replan steps (it usually isn't),
                # we re-run it now to ensure the Markdown file on disk reflects
                # the refined content.
                if "brief" not in replan_spec.replan_steps:
                    _execute_step("brief", input_text, bb)

    # =========================================================================
    # 4. Phase 3: Reporting & Final Assembly (Step 5.4)
    # =========================================================================
    # Generate the executive summary of the planner's decisions.
    # This provides a single source of truth for "what strategy ran?".
    plan_report = PlanReport(
        strategy=plan_spec.strategy,
        enable_history=plan_spec.enable_history,
        initial_steps=plan_spec.steps,
        replan_steps=replan_spec.replan_steps
        if (replan_spec and replan_spec.should_replan)
        else None,
        refine_used=replan_spec.should_replan if replan_spec else False,
        replan_reason=replan_spec.replan_reason if replan_spec else None,
        notes=plan_spec.notes,
    )
    bb.put(_PLANNER_REPORT_KEY, plan_report.model_dump())
    bb.trace("planner: report written")

    # Fetch final state (potentially updated by Phase 2)
    explanations_raw = _as_list(bb.get(_EXPLANATIONS_KEY))
    notes_raw = _as_list(bb.get(_RELEVANCE_NOTES_KEY))
    terms_raw = _as_list(bb.get(_TERMS_KEY))
    events_raw = _as_list(bb.get(_TIMELINE_KEY))
    brief_md_path = cast(str | None, bb.get("public_brief_md_path"))

    # Synthesize the public brief object for the return payload.
    # We cast to list[ExplanationCard] because we know the data shape matches,
    # even if they are dicts at runtime (helper handles dicts).
    cards_any = cast("list[ExplanationCard]", explanations_raw)
    brief_model = _build_public_brief_from_explanations(cards_any)
    brief_dict = _artifact_to_dict(brief_model)

    bb.put(_PUBLIC_BRIEF_KEY, brief_dict)
    bb.trace("pipeline: complete")

    brief_payload: PublicBriefPayload = {
        "title": cast(str, brief_dict.get("title", PIPELINE_BRIEF_TITLE)),
        "summary": cast(str, brief_dict.get("summary", "")),
        "sections": cast(
            list[Mapping[str, object]],
            brief_dict.get("sections", []),
        ),
    }

    return {
        "blackboard": bb,
        "parsed_chunks": cast(list[Mapping[str, object]], _as_list(bb.get(_PARSED_CHUNKS_KEY))),
        "explanations": cast(list[Mapping[str, object]], explanations_raw),
        "relevance_notes": cast(list[Mapping[str, object]], notes_raw),
        "terms": cast(list[Mapping[str, object]], terms_raw),
        "timeline_events": cast(list[Mapping[str, object]], events_raw),
        "public_brief": brief_payload,
        "public_brief_md_path": brief_md_path,
    }


__all__ = ["run_pipeline", "PipelineResult", "PublicBriefPayload"]
