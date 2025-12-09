"""
End-to-end tests for the public-translation pipeline, using optional stubs.

These tests exercise the wiring in ``interlines.pipelines.public_translation``:

- By default, we stub all LLM-backed agents so that CI and local runs do
  not depend on external API keys or network calls.
- The stubs intentionally return plain dicts and lists; the pipeline
  converts them to JSON-safe payloads via ``_artifact_to_dict``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import interlines.pipelines.public_translation as pipeline_mod
from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.planner import PlannerPlanSpec  # <--- 新增这一行
from interlines.core.planner.strategy import expected_path
from interlines.core.result import ok
from interlines.pipelines.public_translation import (
    PipelineResult,
    PublicBriefPayload,
    run_pipeline,
)


def _run_pipeline_with_stubbed_agents(  # noqa: C901
    input_text: str,
    *,
    enable_history: bool,
) -> PipelineResult:
    """Run ``run_pipeline`` with all LLM-dependent agents stubbed out.

    The helper monkey-patches the agent entrypoints imported by the
    :mod:`public_translation` module so that end-to-end tests can run in
    environments without API keys or network access.
    """
    # Work with the module as ``Any`` so mypy does not complain about
    # attribute existence or type mismatches when monkeypatching.
    mod_any = cast(Any, pipeline_mod)

    # Save original callables so we can restore them in a ``finally`` block.
    orig_run_explainer = mod_any.run_explainer
    orig_run_citizen = mod_any.run_citizen
    orig_run_jargon = mod_any.run_jargon
    orig_run_history = mod_any.run_history
    orig_run_editor = mod_any.run_editor
    orig_run_brief_builder = mod_any.run_brief_builder
    orig_planner_cls = mod_any.PlannerAgent

    # --- 1. Define Fake Functions ---

    def fake_run_explainer(bb: Blackboard) -> Any:
        """Return a single minimal explanation card as a plain mapping."""
        card = {
            "kind": "explanation.stub",
            "version": "1.0.0",
            "confidence": 1.0,
            "claim": "Stub explanation for testing.",
            "rationale": "This is a stub rationale produced by the test suite.",
            "summary": "Stub summary of explanation.",
            "text": "Stub explanation for testing.",
        }
        return ok([card])

    def fake_run_citizen(bb: Blackboard) -> Any:
        """Return one relevance note in dict form."""
        note = {
            "kind": "relevance_note.stub",
            "version": "1.0.0",
            "confidence": 0.9,
            "target": "stub-target",
            "rationale": "Stub reason why this matters to the reader.",
            "score": 0.9,
        }
        return ok([note])

    def fake_run_jargon(bb: Blackboard) -> Any:
        """Return a single terminology card as a dict."""
        term = {
            "kind": "term.stub",
            "version": "1.0.0",
            "confidence": 0.8,
            "term": "stub term",
            "definition": "A placeholder term used only in tests.",
            "aliases": ["stub-term"],
            "examples": ["This pipeline uses a stub term in tests."],
            "sources": [],
        }
        return ok([term])

    def fake_run_history(bb: Blackboard) -> Any:
        """Return zero or one timeline events depending on ``enable_history``."""
        if not enable_history:
            return ok([])
        event = {
            "kind": "timeline_event.stub",
            "version": "1.0.0",
            "confidence": 0.6,
            "when": "2000-01-01",
            "title": "Stub historical event",
            "description": "A fake event used to test the history branch.",
            "tags": ["test"],
            "sources": [],
        }
        return ok([event])

    def fake_run_editor(bb: Blackboard) -> Any:
        """Return a minimal review report as a plain mapping."""
        report = {
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
            },
            "comments": ["All checks passed in the stubbed editor."],
            "actions": [],
        }
        return ok(report)

    def fake_run_brief_builder(
        bb: Blackboard,
        *,
        run_id: str = "run",
        reports_dir: str | None = None,
    ) -> Any:
        """Pretend to write a markdown file and return a fake path."""
        # Fixed: Ensure filename format matches test expectation ("-stub.md")
        return ok(Path(f"artifacts/reports/{run_id}-stub.md"))

    # --- 2. Define Fake Planner Class ---

    class FakePlannerAgent:
        """A stub planner that returns a deterministic plan without LLMs."""

        def __init__(self, llm: Any = None, model_alias: str = "") -> None:
            pass

        def plan(self, bb: Blackboard, ctx: Any) -> PlannerPlanSpec:
            # Deterministically return steps based on the request flag
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

    try:
        # Install stubs on the module.
        mod_any.run_explainer = fake_run_explainer
        mod_any.run_citizen = fake_run_citizen
        mod_any.run_jargon = fake_run_jargon
        mod_any.run_history = fake_run_history
        mod_any.run_editor = fake_run_editor
        mod_any.run_brief_builder = fake_run_brief_builder
        mod_any.PlannerAgent = FakePlannerAgent

        # Run the real pipeline implementation with our fake agents.
        return run_pipeline(input_text, enable_history=enable_history)
    finally:
        # Restore original functions so that other tests are not affected.
        mod_any.run_explainer = orig_run_explainer
        mod_any.run_citizen = orig_run_citizen
        mod_any.run_jargon = orig_run_jargon
        mod_any.run_history = orig_run_history
        mod_any.run_editor = orig_run_editor
        mod_any.run_brief_builder = orig_run_brief_builder
        mod_any.PlannerAgent = orig_planner_cls


def test_run_pipeline_with_history_produces_artifacts() -> None:
    """Full pipeline run (with history) should produce core artifacts."""
    input_text = (
        "InterLines turns expert language into public language. "
        "It also provides a historical lens over time. "
        "Agents collaborate through a shared blackboard to build layered "
        "explanations."
    )

    result: PipelineResult = _run_pipeline_with_stubbed_agents(
        input_text,
        enable_history=True,
    )

    # Parsed chunks: non-empty list of dicts.
    parsed = result["parsed_chunks"]
    assert isinstance(parsed, list)
    assert parsed
    assert all(isinstance(x, dict) for x in parsed)

    seg = parsed[0]
    assert "id" in seg
    assert "text" in seg
    # In stub-mode (llm=None), parser_agent only guarantees id/text.
    assert isinstance(seg["id"], str)
    assert isinstance(seg["text"], str)

    # Explanation / relevance / term / timeline artifacts: lists of dicts.
    explanations = result["explanations"]
    notes = result["relevance_notes"]
    terms = result["terms"]
    events = result["timeline_events"]

    for collection in (explanations, notes, terms, events):
        assert isinstance(collection, list)
        assert all(isinstance(x, dict) for x in collection)

    # Public brief payload: basic shape and title/summary presence.
    brief: PublicBriefPayload = result["public_brief"]
    assert isinstance(brief["title"], str)
    assert brief["title"]
    assert isinstance(brief["summary"], str)
    assert brief["summary"]
    assert isinstance(brief["sections"], list)
    assert brief["sections"]

    # Markdown path: a non-empty string produced by the brief builder.
    md_path = result["public_brief_md_path"]
    assert isinstance(md_path, str)
    assert md_path.endswith("-stub.md")


def test_run_pipeline_without_history_skips_timeline() -> None:
    """When ``enable_history=False``, the history branch should be skipped."""
    input_text = "Short text for a non-history run."

    result: PipelineResult = _run_pipeline_with_stubbed_agents(
        input_text,
        enable_history=False,
    )

    events = result["timeline_events"]
    assert isinstance(events, list)
    assert events == []

    # The brief should still be produced.
    brief: PublicBriefPayload = result["public_brief"]
    assert brief["title"]
    assert brief["summary"]


def test_pipeline_records_planner_dag_and_trace() -> None:
    """Planner DAG payload and trace snapshot should be recorded."""
    input_text = "Trace and planner DAG test."
    result: PipelineResult = _run_pipeline_with_stubbed_agents(
        input_text,
        enable_history=True,
    )
    bb = result["blackboard"]

    # The planner DAG should be stored under a dedicated key.
    dag_payload = bb.get("planner_dag")
    assert isinstance(dag_payload, dict)
    assert dag_payload["strategy"] == "with_history"
    assert tuple(dag_payload["topo"]) == expected_path(enable_history=True)

    # A trace snapshot should contain the planner note.
    snaps = bb.traces()
    assert snaps
    notes = [snap.note for snap in snaps]
    assert "planner: public_translation plan" in notes

    planner_snaps = [snap for snap in snaps if snap.note == "planner: public_translation plan"]
    assert planner_snaps
    last = planner_snaps[-1]
    data = last.data
    assert "planner_dag" in data
    assert isinstance(data["planner_dag"], dict)


def test_pipeline_records_final_trace() -> None:
    """Pipeline should record a final completion snapshot on the blackboard."""
    result: PipelineResult = _run_pipeline_with_stubbed_agents(
        "Trace test text.",
        enable_history=False,
    )
    bb = result["blackboard"]

    snaps = bb.traces()
    assert len(snaps) >= 1

    last = snaps[-1]
    note = last.note
    assert note is not None
    assert "pipeline: public_translation complete" in note
