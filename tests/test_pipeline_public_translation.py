"""
End-to-end tests for the public-translation pipeline.

These tests exercise the wiring in ``interlines.pipelines.public_translation``:

- Ensure the pipeline returns a well-formed ``PipelineResult``.
- Verify that core artifacts (explanations, relevance notes, terms,
  timeline, review report, public brief) are present and JSON-safe.
- Check that the ``enable_history`` flag controls timeline behaviour.
- Confirm that the Blackboard records at least one trace entry.
"""

from __future__ import annotations

from interlines.core.blackboard.memory import Blackboard
from interlines.pipelines.public_translation import (
    PIPELINE_BRIEF_TITLE,
    PipelineResult,
    run_pipeline,
)


def test_run_pipeline_with_history_produces_artifacts() -> None:
    """Full pipeline run with history enabled should produce core artifacts.

    This test checks that the public-translation pipeline:

    - Accepts raw input text.
    - Produces parsed chunks as a non-empty list of strings.
    - Exposes explanation, relevance, term, and timeline artifacts as lists of dicts.
    - Produces a structured public brief payload with title/summary/sections.
    - Optionally returns a markdown path as a string.
    """
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


def test_run_pipeline_without_history_has_empty_timeline() -> None:
    """Running the pipeline with history disabled should yield an empty timeline.

    The pipeline still wires the same agents, but the history agent is
    skipped and the ``timeline_events`` key is explicitly set to an empty
    list. This keeps downstream consumers simple: they can always rely
    on the key existing.
    """
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


def test_pipeline_records_final_trace() -> None:
    """Pipeline should record at least one trace snapshot on the blackboard.

    The trace API is used for debugging and auditability. Here we only
    check that:

    - At least one trace snapshot is recorded.
    - The last snapshot's note mentions the pipeline, which should
      correspond to the completion marker in the implementation.
    """
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
