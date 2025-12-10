"""
End-to-end tests for the public-translation pipeline, using optional stubs.

These tests exercise the wiring in ``interlines.pipelines.public_translation``.
Refactored to move inner functions to module level to reduce cyclomatic complexity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import interlines.pipelines.public_translation as pipeline_mod
from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.planner import PlannerPlanSpec
from interlines.core.planner.strategy import expected_path
from interlines.core.result import ok
from interlines.pipelines.public_translation import (
    PipelineResult,
    PublicBriefPayload,
    run_pipeline,
)

# --- Module-level Stubs (moved out to reduce complexity) ---


def _fake_run_explainer(bb: Blackboard) -> Any:
    return ok(
        [
            {
                "kind": "explanation.stub",
                "version": "1.0.0",
                "confidence": 1.0,
                "claim": "Stub explanation for testing.",
                "rationale": "This is a stub rationale produced by the test suite.",
                "summary": "Stub summary of explanation.",
                "text": "Stub explanation for testing.",
                "evidence": [],
            }
        ]
    )


def _fake_run_citizen(bb: Blackboard) -> Any:
    return ok(
        [
            {
                "kind": "relevance_note.stub",
                "version": "1.0.0",
                "confidence": 0.9,
                "target": "stub-target",
                "rationale": "Stub reason why this matters to the reader.",
                "score": 0.9,
            }
        ]
    )


def _fake_run_jargon(bb: Blackboard) -> Any:
    return ok(
        [
            {
                "kind": "term.stub",
                "version": "1.0.0",
                "confidence": 0.8,
                "term": "stub term",
                "definition": "A placeholder term used only in tests.",
                "aliases": ["stub-term"],
                "examples": ["This pipeline uses a stub term in tests."],
                "sources": [],
            }
        ]
    )


def _fake_run_history(bb: Blackboard) -> Any:
    # We can't easily check 'enable_history' here without a closure,
    # so we return a default event. The test logic handles conditional checking.
    return ok(
        [
            {
                "kind": "timeline_event.stub",
                "version": "1.0.0",
                "confidence": 0.6,
                "when": "2000-01-01",
                "title": "Stub historical event",
                "description": "A fake event used to test the history branch.",
                "tags": ["test"],
                "sources": [],
            }
        ]
    )


def _fake_run_editor(bb: Blackboard) -> Any:
    return ok(
        {
            "kind": "review_report.stub",
            "version": "1.0.0",
            "confidence": 1.0,
            "overall": 1.0,
            "criteria": {
                "kind": "review_criteria.stub",
                "version": "1.0.0",
                "confidence": 1.0,
                "readability": 1.0,
                "factuality": 1.0,
                "bias": 1.0,
                "accuracy": 1.0,
                "clarity": 1.0,
                "completeness": 1.0,
                "safety": 1.0,
            },
            "comments": ["All checks passed in the stubbed editor."],
            "actions": [],
        }
    )


def _fake_run_brief_builder(
    bb: Blackboard,
    *,
    run_id: str = "run",
    reports_dir: str | None = None,
) -> Any:
    """Pretend to write a markdown file and return a fake path."""
    path = Path(f"artifacts/reports/{run_id}-stub.md")
    bb.put("public_brief_md_path", str(path))
    return ok(path)


class _FakePlannerAgent:
    """A stub planner that returns a deterministic plan without LLMs."""

    def __init__(self, llm: Any = None, model_alias: str = "") -> None:
        pass

    def plan(self, bb: Blackboard, ctx: Any) -> PlannerPlanSpec:
        if ctx.enable_history_requested:
            steps = ["parse", "translate", "timeline", "narrate", "review", "brief"]
        else:
            steps = ["parse", "translate", "narrate", "review", "brief"]

        return PlannerPlanSpec(
            strategy="stub_strategy",
            steps=steps,
            enable_history=ctx.enable_history_requested,
            notes="Stubbed plan for testing.",
        )

    def replan(self, *args: Any, **kwargs: Any) -> PlannerPlanSpec:
        return PlannerPlanSpec(strategy="stub", should_replan=False)


# --- Main Test Helper ---


def _run_pipeline_with_stubbed_agents(
    input_data: str,
    *,
    enable_history: bool,
) -> PipelineResult:
    """Run ``run_pipeline`` with all LLM-dependent agents stubbed out.

    Monkey-patches the agent entrypoints.
    """
    mod_any = cast(Any, pipeline_mod)

    orig_run_explainer = mod_any.run_explainer
    orig_run_citizen = mod_any.run_citizen
    orig_run_jargon = mod_any.run_jargon
    orig_run_history = mod_any.run_history
    orig_run_editor = mod_any.run_editor
    orig_run_brief_builder = mod_any.run_brief_builder
    orig_planner_cls = mod_any.PlannerAgent

    try:
        mod_any.run_explainer = _fake_run_explainer
        mod_any.run_citizen = _fake_run_citizen
        mod_any.run_jargon = _fake_run_jargon
        mod_any.run_history = _fake_run_history
        mod_any.run_editor = _fake_run_editor
        mod_any.run_brief_builder = _fake_run_brief_builder
        mod_any.PlannerAgent = _FakePlannerAgent

        return run_pipeline(input_data=input_data, enable_history=enable_history)
    finally:
        mod_any.run_explainer = orig_run_explainer
        mod_any.run_citizen = orig_run_citizen
        mod_any.run_jargon = orig_run_jargon
        mod_any.run_history = orig_run_history
        mod_any.run_editor = orig_run_editor
        mod_any.run_brief_builder = orig_run_brief_builder
        mod_any.PlannerAgent = orig_planner_cls


# --- Tests ---


def test_run_pipeline_with_history_produces_artifacts() -> None:
    """Full pipeline run (with history) should produce core artifacts."""
    input_data = "Stub input."

    result: PipelineResult = _run_pipeline_with_stubbed_agents(
        input_data,
        enable_history=True,
    )

    parsed = result["parsed_chunks"]
    assert isinstance(parsed, list)

    explanations = result["explanations"]
    notes = result["relevance_notes"]
    terms = result["terms"]
    events = result["timeline_events"]

    for collection in (explanations, notes, terms, events):
        assert isinstance(collection, list)
        assert all(isinstance(x, dict) for x in collection)

    brief: PublicBriefPayload = result["public_brief"]
    assert brief["title"]
    assert brief["summary"]
    assert isinstance(brief["sections"], list)

    md_path = result["public_brief_md_path"]
    assert isinstance(md_path, str)
    assert md_path.endswith("-stub.md")


def test_run_pipeline_without_history_skips_timeline() -> None:
    """When ``enable_history=False``, the history branch should be skipped."""
    input_data = "Short text."

    # NOTE: Our _fake_run_history stub returns an event blindly.
    # However, since the planner stub respects enable_history,
    # the 'timeline' step will NOT be in the DAG, so run_history won't be called.
    # Therefore, we expect empty events in the result.
    result: PipelineResult = _run_pipeline_with_stubbed_agents(
        input_data,
        enable_history=False,
    )

    events = result["timeline_events"]
    assert isinstance(events, list)
    assert events == []

    brief: PublicBriefPayload = result["public_brief"]
    assert brief["title"]


def test_pipeline_records_planner_dag_and_trace() -> None:
    """Planner DAG payload and trace snapshot should be recorded."""
    input_data = "Trace test."
    result: PipelineResult = _run_pipeline_with_stubbed_agents(
        input_data,
        enable_history=True,
    )
    bb = result["blackboard"]

    dag_payload = bb.get("planner_dag")
    assert isinstance(dag_payload, dict)
    assert dag_payload["strategy"] == "stub_strategy"
    assert tuple(dag_payload["topo"]) == expected_path(enable_history=True)

    snaps = bb.traces()
    assert snaps
    planner_snaps = [snap for snap in snaps if snap.note == "planner: phase 1 plan"]
    assert planner_snaps


def test_pipeline_records_final_trace() -> None:
    """Pipeline should record a final completion snapshot."""
    result: PipelineResult = _run_pipeline_with_stubbed_agents(
        "Trace test.",
        enable_history=False,
    )
    bb = result["blackboard"]
    snaps = bb.traces()
    last = snaps[-1]
    assert last.note is not None
    assert "pipeline: complete" in last.note
