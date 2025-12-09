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
  (possibly using a DAG in later milestones), while each agent is free
  to use non-deterministic sampling internally.
- **LLM-provider-agnostic**:
  Agents talk to the LLM client through well-defined contracts. This
  module does not depend on concrete model IDs.
- **Blackboard-centric**:
  All intermediate artifacts (parsed chunks, explanations, terms, etc.)
  are stored on a central :class:`Blackboard` to simplify inspection and
  downstream processing.

Entry point
-----------
The main entry point is :func:`run_pipeline`, which returns a
JSON-safe ``PipelineResult`` containing:

- The shared :class:`Blackboard` instance.
- Parsed chunks as a list of ``{"id": ..., "text": ...}`` dicts.
- Explanation / relevance / term / timeline artifacts as dicts.
- A structured ``PublicBrief`` payload built from explanations.
- The path to a generated Markdown brief (if any).

In Step 5.2 the planner becomes pluggable:

- ``use_llm_planner=False`` keeps the rule-based planner from Step 5.1.
- ``use_llm_planner=True`` (default) routes through :class:`PlannerAgent`,
  which asks an LLM to propose a plan (steps, thresholds, notes) and then
  converts it into a :class:`DAG` for execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
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
from interlines.core.contracts.public_brief import BriefSection, PublicBrief
from interlines.core.contracts.relevance import RelevanceNote
from interlines.core.contracts.term import TermCard
from interlines.core.contracts.timeline import TimelineEvent
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
_PUBLIC_BRIEF_KEY = "public_brief"
_PLANNER_PLAN_KEY = "planner_plan_spec.initial"
_PLANNER_DAG_KEY = "planner_dag"

# Title used for auto-constructed briefs when the brief-builder agent is not
# invoked (e.g., in very small test cases).
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


# --------------------------------------------------------------------------- #
# Brief construction (fallback when brief-builder is not used)
# --------------------------------------------------------------------------- #


def _build_public_brief_from_explanations(
    cards: list[ExplanationCard],
) -> PublicBrief:
    """
    Construct a :class:`PublicBrief` from explanation cards.

    This function is intentionally pure and testable. It mirrors what a
    future LLM agent *might* produce when assembling narrative structure.

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

    # Artifact schema requires kind/version/confidence
    return PublicBrief(
        kind="public_brief.v1",
        version="1.0.0",
        confidence=1.0,
        title=PIPELINE_BRIEF_TITLE,
        summary=summary,
        sections=sections,
    )


# --------------------------------------------------------------------------- #
# DAG-driven pipeline entry point
# --------------------------------------------------------------------------- #


def run_pipeline(
    input_text: str,
    enable_history: bool = False,
    use_llm_planner: bool = True,
) -> PipelineResult:
    """
    DAG-driven pipeline entry point for the public-translation workflow.

    Parameters
    ----------
    input_text:
        Raw input text to be parsed and translated.
    enable_history:
        If ``True``, the caller expresses a preference for including the
        history/timeline branch. The *planner* (rule-based or LLM-backed)
        may still decide to drop that branch in some cases.
    use_llm_planner:
        When ``True`` (default), the high-level execution plan is produced
        by :class:`PlannerAgent` using an LLM ("semantic routing"). When
        ``False``, the legacy rule-based planner from Step 5.1 is used as a
        safe, deterministic fallback.
    """
    # ------------- Blackboard + initial state -------------
    bb = Blackboard()
    bb.put("input_text", input_text)

    # ------------- Planning (rule-based vs LLM-backed) -------------
    if use_llm_planner:
        # Build a minimal planning context. This can be enriched later with
        # more document metadata (e.g., content type, source, user persona).
        ctx = PlannerContext(
            task_type="public_translation",
            document_kind="generic",
            approx_char_count=len(input_text),
            language="en",  # TODO: plug in a lightweight language detector.
            enable_history_requested=enable_history,
        )
        # Fixed (Mypy): Use `from_env` for proper initialization defaults.
        planner = PlannerAgent(llm=LLMClient.from_env(default_model_alias="planner"))
        plan_spec = planner.plan(bb, ctx)
    else:
        # Preserve the legacy rule-based behavior from Step 5.1.
        plan_spec, _legacy_dag = build_plan(enable_history=enable_history)

    dag = DAG.from_plan_spec(plan_spec)

    # Record both the structured plan and the DAG on the blackboard so that
    # tests, debug tools, and future UIs can introspect them.
    bb.put(_PLANNER_PLAN_KEY, plan_spec.model_dump())
    bb.put(_PLANNER_DAG_KEY, dag.to_payload())
    bb.trace("planner: public_translation plan")

    # ------------- Execute agents along DAG -------------
    explanation_cards: list[ExplanationCard] = []
    relevance_notes: list[RelevanceNote] = []
    term_cards: list[TermCard] = []
    timeline_events: list[TimelineEvent] = []
    brief_model: PublicBrief | None = None
    brief_md_path: str | None = None

    for step in dag.topological_order():
        if step == "parse":
            # Fixed (Mypy): parser_agent returns list[dict], not models.
            parsed_chunks_raw = parser_agent(input_text, bb)
            bb.put(_PARSED_CHUNKS_KEY, parsed_chunks_raw)

        elif step == "translate":
            # Explainer + citizen + jargon together form the "public
            # translation" layer, but they remain distinct agents so that
            # we can rewire them in future milestones.
            # Fixed (Logic/Mypy): Agents read directly from blackboard.
            explainer_result: Result[list[ExplanationCard], str] = run_explainer(bb)
            explanation_cards = _unwrap_or_fail("explainer_agent", explainer_result)
            bb.put(
                _EXPLANATIONS_KEY,
                [_artifact_to_dict(card) for card in explanation_cards],
            )

        elif step == "narrate":
            # Citizen: relevance framing and audience-facing language.
            citizen_result: Result[list[RelevanceNote], str] = run_citizen(bb)
            relevance_notes = _unwrap_or_fail("citizen_agent", citizen_result)
            bb.put(
                _RELEVANCE_NOTES_KEY,
                [_artifact_to_dict(note) for note in relevance_notes],
            )

            # Jargon: terms and definitions.
            jargon_result: Result[list[TermCard], str] = run_jargon(bb)
            term_cards = _unwrap_or_fail("jargon_agent", jargon_result)
            bb.put(
                _TERMS_KEY,
                [_artifact_to_dict(term) for term in term_cards],
            )

        elif step == "timeline":
            # History: optional temporal lens and evolution narrative.
            history_result: Result[list[TimelineEvent], str] = run_history(bb)
            timeline_events = _unwrap_or_fail("history_agent", history_result)
            bb.put(
                _TIMELINE_KEY,
                [_artifact_to_dict(ev) for ev in timeline_events],
            )

        elif step == "review":
            # Editor: factuality, bias, clarity checks. The editor sees the
            # whole artifact graph (explanations, terms, relevance, timeline).
            # Fixed (F841): Assign unused result to `_`.
            _review_result = run_editor(bb)
            _ = _unwrap_or_fail("editor_agent", _review_result)
            # We currently do not store the whole review on the blackboard;
            # this can be added once the report schema stabilizes.

        elif step == "brief":
            # Brief builder: final assembly into a public-facing Markdown brief.
            # Fixed (Mypy): Result type is Result[Path, str].
            brief_result: Result[Path, str] = run_brief_builder(bb)
            path_obj = _unwrap_or_fail("brief_builder_agent", brief_result)
            brief_md_path = str(path_obj)

        # Unknown steps are ignored; this makes it easier to experiment with
        # planner outputs without breaking older pipeline versions.

    # Fallback / Synthesis:
    # If brief_builder ran, it produced a file, but we still need the
    # structured PublicBrief object for the return payload. We synthesize it
    # here using the deterministic helper.
    brief_model = _build_public_brief_from_explanations(explanation_cards)
    bb.put(_PUBLIC_BRIEF_KEY, _artifact_to_dict(brief_model))

    # Record a final snapshot to help with debugging and tests.
    bb.trace("pipeline: public_translation complete")

    # ------------- Assemble JSON-safe result -------------
    brief_dict = _artifact_to_dict(brief_model)
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
        "parsed_chunks": cast(
            list[Mapping[str, object]],
            bb.get(_PARSED_CHUNKS_KEY) or [],
        ),
        "explanations": cast(
            list[Mapping[str, object]],
            bb.get(_EXPLANATIONS_KEY) or [],
        ),
        "relevance_notes": cast(
            list[Mapping[str, object]],
            bb.get(_RELEVANCE_NOTES_KEY) or [],
        ),
        "terms": cast(
            list[Mapping[str, object]],
            bb.get(_TERMS_KEY) or [],
        ),
        "timeline_events": cast(
            list[Mapping[str, object]],
            bb.get(_TIMELINE_KEY) or [],
        ),
        "public_brief": brief_payload,
        "public_brief_md_path": brief_md_path,
    }


__all__ = ["run_pipeline", "PipelineResult", "PublicBriefPayload"]
