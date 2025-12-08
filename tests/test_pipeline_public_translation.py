"""
End-to-end tests for the public-translation pipeline, using optional stubs.

These tests exercise the wiring in ``interlines.pipelines.public_translation``:

- By default, we stub all LLM-backed agents so that CI and local runs do
  not depend on external API keys or network calls.
- If the environment variable ``INTERLINES_PIPELINE_TEST_MODE`` is set
  to ``"real"``, stubs are disabled and the tests exercise the real
  agents. In that mode, missing API keys will cause failures, which is
  useful as an integration-style self-check on a developer machine.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest  # type: ignore[import-not-found]

from interlines.core.blackboard.memory import Blackboard
from interlines.core.result import ok
from interlines.pipelines import public_translation as pipeline_mod
from interlines.pipelines.public_translation import (
    PIPELINE_BRIEF_TITLE,
    PipelineResult,
    run_pipeline,
)

# ---------------------------------------------------------------------------
# Test mode toggle
# ---------------------------------------------------------------------------

USE_STUBS = os.getenv("INTERLINES_PIPELINE_TEST_MODE", "stubs") != "real"
"""
Whether to stub out LLM-backed agents in pipeline tests.

- Default (when the env var is unset or not "real"): USE_STUBS is True and
  we use lightweight stubs that avoid network calls and API keys.
- If you set ``INTERLINES_PIPELINE_TEST_MODE=real``, USE_STUBS becomes False
  and the tests run against the real agents. In that mode, misconfigured
  environment variables (e.g. missing OPENAI_API_KEY) will surface as
  test failures, which is desirable for integration testing.
"""


def _patch_pipeline_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch LLM-backed agents in the pipeline module with lightweight stubs.

    If ``USE_STUBS`` is False (i.e. ``INTERLINES_PIPELINE_TEST_MODE=real``
    is set), this function becomes a no-op and the real agents are used.
    That lets local developers run integration-style tests that also
    validate environment configuration and API-key wiring.
    """
    if not USE_STUBS:
        # Integration mode: do NOT patch; use real agents.
        return

    def fake_run_explainer(bb: Blackboard) -> Any:
        """Stub explainer agent: produce one simple explanation card."""
        cards: list[dict[str, Any]] = [
            {
                "kind": "explanation.v1",
                "version": "1.0.0",
                "confidence": 0.9,
                "claim": "The policy aims to make expert language accessible.",
                "rationale": (
                    "The text describes how InterLines translates technical "
                    "content into public-facing explanations using agents."
                ),
                "evidence": [],
                "summary": None,
            },
        ]
        bb.put("explanations", cards)
        return ok(cards)

    def fake_run_citizen(bb: Blackboard) -> Any:
        """Stub citizen agent: produce one simple relevance note."""
        notes: list[dict[str, Any]] = [
            {
                "kind": "relevance.v1",
                "version": "1.0.0",
                "confidence": 0.8,
                "audience": "general_public",
                "title": "Why this matters for everyday readers",
                "summary": (
                    "The work helps people understand complex policies "
                    "without needing expert training."
                ),
                "bullets": [
                    "Makes technical research more accessible.",
                    "Supports better-informed public debate.",
                ],
                "sources": ["p1"],
            },
        ]
        bb.put("relevance_notes", notes)
        return ok(notes)

    def fake_run_jargon(bb: Blackboard) -> Any:
        """Stub jargon agent: produce one terminology card."""
        terms: list[dict[str, Any]] = [
            {
                "kind": "term.v1",
                "version": "1.0.0",
                "confidence": 0.85,
                "term": "Blackboard",
                "definition": (
                    "A shared in-memory space where agents read and write "
                    "intermediate artifacts."
                ),
                "aliases": ["shared memory"],
                "examples": [
                    "The parser writes parsed_chunks to the blackboard.",
                ],
                "sources": ["p1"],
            },
        ]
        bb.put("terms", terms)
        return ok(terms)

    def fake_run_history(bb: Blackboard) -> Any:
        """Stub history agent: produce one simple timeline event."""
        events: list[dict[str, Any]] = [
            {
                "kind": "timeline_event.v1",
                "version": "1.0.0",
                "confidence": 0.7,
                "when": "2020-01-01",
                "title": "InterLines concept proposed",
                "description": (
                    "Initial idea to coordinate multiple agents " "over a shared blackboard."
                ),
                "tags": ["concept"],
                "sources": ["doc:intro"],
            },
        ]
        bb.put("timeline_events", events)
        return ok(events)

    def fake_run_editor(bb: Blackboard) -> Any:
        """Stub editor agent: produce a minimal review report."""
        report: dict[str, Any] = {
            "kind": "review_report.v1",
            "version": "1.0.0",
            "confidence": 0.9,
            "overall": "pass",
            "issues": [],
            "notes": "Stubbed review: no issues detected.",
        }
        bb.put("review_report", report)
        return ok(report)

    def fake_run_brief_builder(
        bb: Blackboard,
        *,
        run_id: str,
        reports_dir: str | Path | None = None,
    ) -> Any:
        """Stub brief builder: pretend to generate a Markdown file.

        We return a synthetic path wrapped in ``Result.ok``; the pipeline
        only checks that it can be converted to a string, not that the
        file actually exists on disk.
        """
        path = Path(f"/tmp/{run_id}.md")
        return ok(path)

    # Patch the symbols used by the pipeline.
    monkeypatch.setattr(pipeline_mod, "run_explainer", fake_run_explainer)
    monkeypatch.setattr(pipeline_mod, "run_citizen", fake_run_citizen)
    monkeypatch.setattr(pipeline_mod, "run_jargon", fake_run_jargon)
    monkeypatch.setattr(pipeline_mod, "run_history", fake_run_history)
    monkeypatch.setattr(pipeline_mod, "run_editor", fake_run_editor)
    monkeypatch.setattr(pipeline_mod, "run_brief_builder", fake_run_brief_builder)


def test_run_pipeline_with_history_produces_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full pipeline run with history enabled should produce core artifacts."""
    _patch_pipeline_agents(monkeypatch)

    input_text = (
        "InterLines turns expert language into public language. "
        "It also provides a historical lens over time. "
        "Agents collaborate through a shared blackboard to build layered explanations."
    )

    result: PipelineResult = run_pipeline(input_text, enable_history=True)

    # The blackboard object should be present and typed correctly.
    bb = result["blackboard"]
    assert isinstance(bb, Blackboard)

    # Parsed chunks should be a non-empty list of strings.
    parsed_chunks = result["parsed_chunks"]
    assert isinstance(parsed_chunks, list)
    assert parsed_chunks, "parsed_chunks should not be empty"
    assert all(isinstance(chunk, str) for chunk in parsed_chunks)

    # Explanation, relevance, term, and timeline artifacts should be lists of dicts.
    explanations = result["explanations"]
    relevance_notes = result["relevance_notes"]
    terms = result["terms"]
    timeline_events = result["timeline_events"]

    for name, value in [
        ("explanations", explanations),
        ("relevance_notes", relevance_notes),
        ("terms", terms),
        ("timeline_events", timeline_events),
    ]:
        assert isinstance(value, list), f"{name} must be a list"
        assert all(isinstance(item, dict) for item in value), f"{name} items must be dicts"

    # History was enabled, so we expect at least one timeline event (in stub mode).
    if USE_STUBS:
        assert (
            timeline_events
        ), "timeline_events should not be empty when history is enabled in stub mode"

    # Review report should be either a dict or None if something failed gracefully.
    review_report = result["review_report"]
    assert review_report is None or isinstance(review_report, dict)

    # Public brief payload should expose the basic fields defined in PublicBriefPayload.
    public_brief = result["public_brief"]
    assert isinstance(public_brief, dict)

    # Required keys: title / summary / sections.
    assert "title" in public_brief
    assert "summary" in public_brief
    assert "sections" in public_brief

    assert isinstance(public_brief["title"], str)
    assert isinstance(public_brief["summary"], str)
    assert isinstance(public_brief["sections"], list)

    # There should be at least one section when explanations are present.
    if explanations:
        assert public_brief["sections"], "public_brief.sections should not be empty"

    # Markdown brief path is optional, but if present, it must be a string.
    md_path = result["public_brief_md_path"]
    assert md_path is None or isinstance(md_path, str)


def test_run_pipeline_without_history_has_empty_timeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running the pipeline with history disabled should yield an empty timeline."""
    _patch_pipeline_agents(monkeypatch)

    input_text = "Short text for pipeline test."

    result: PipelineResult = run_pipeline(input_text, enable_history=False)

    timeline_events = result["timeline_events"]
    assert isinstance(timeline_events, list)
    assert timeline_events == []

    # Public brief should still exist with a title and sections (possibly empty).
    public_brief = result["public_brief"]
    assert isinstance(public_brief, dict)
    assert "title" in public_brief
    assert isinstance(public_brief["title"], str)
    assert public_brief["title"] != ""

    # If no explanations are available, the title may fall back to PIPELINE_BRIEF_TITLE.
    assert public_brief["title"] or PIPELINE_BRIEF_TITLE


def test_pipeline_records_final_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipeline should record at least one trace snapshot on the blackboard."""
    _patch_pipeline_agents(monkeypatch)

    result: PipelineResult = run_pipeline(
        "Trace test text.",
        enable_history=False,
    )
    bb = result["blackboard"]

    snaps = bb.traces()
    assert len(snaps) >= 1

    last = snaps[-1]
    # ``note`` is expected to be a human-readable string.
    note = getattr(last, "note", None)
    assert isinstance(note, str)
    assert "pipeline" in note
