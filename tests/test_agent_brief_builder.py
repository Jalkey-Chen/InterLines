"""Tests for the Markdown public brief builder agent.

These tests exercise the behaviour of ``run_brief_builder``:

- Ensure that, given explanations/terms/timeline events on the blackboard,
  a Markdown file is generated in the requested directory.
- Verify that the file contains the expected sections and content.
- Verify that the blackboard records the output path.
- Check that calling the builder with an empty blackboard returns an error.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from interlines.agents.brief_builder import (
    _PUBLIC_BRIEF_MD_KEY,
    run_brief_builder,
)
from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.explanation import EvidenceItem, ExplanationCard
from interlines.core.contracts.term import TermCard
from interlines.core.contracts.timeline import TimelineEvent


def _make_explanation_card() -> ExplanationCard:
    """Create a minimal ExplanationCard for brief-builder tests.

    The card includes:
    - a short claim (used in the brief title and overview),
    - a short rationale paragraph,
    - one EvidenceItem with a non-empty source.
    """
    evidence = [
        EvidenceItem(
            text="The authors run a randomized evaluation.",
            source="paragraphs: p1, p2",
        ),
    ]
    return ExplanationCard(
        kind="explanation.v1",
        version="1.0.0",
        confidence=0.9,
        claim="The policy has measurable effects on outcomes.",
        rationale="The study compares treated and control groups over time.",
        evidence=evidence,
        summary=None,
    )


def _make_term_card() -> TermCard:
    """Create a small TermCard representing a glossary entry."""
    return TermCard(
        kind="term.v1",
        version="1.0.0",
        confidence=0.8,
        term="Randomized controlled trial",
        definition="An experiment where units are randomly assigned " "to treatment or control.",
        aliases=["RCT"],
        examples=[
            "The authors implement an RCT across several regions.",
        ],
        sources=["p1"],
    )


def _make_timeline_event() -> TimelineEvent:
    """Create a simple TimelineEvent for brief-builder tests."""
    return TimelineEvent(
        kind="timeline_event.v1",
        version="1.0.0",
        confidence=0.75,
        when=date(2020, 1, 1),
        title="Policy pilot launched",
        description="Initial small-scale deployment of the new policy.",
        tags=["pilot"],
        sources=["doc:history"],
    )


def test_brief_builder_generates_markdown_file(tmp_path: Path) -> None:
    """run_brief_builder should create a Markdown file and record its path.

    Steps
    -----
    1. Seed a Blackboard with:
       - one ExplanationCard,
       - one TermCard,
       - one TimelineEvent.
    2. Call ``run_brief_builder`` with a custom reports_dir under tmp_path.
    3. Assert that:
       - the Result is Ok,
       - the output file exists on disk,
       - key section titles and content appear in the file,
       - the blackboard stores the path under ``_PUBLIC_BRIEF_MD_KEY``.
    """
    bb = Blackboard()
    bb.put("explanations", [_make_explanation_card()])
    bb.put("terms", [_make_term_card()])
    bb.put("timeline_events", [_make_timeline_event()])

    reports_dir = tmp_path / "artifacts" / "reports"
    run_id = "test-brief-run"

    result = run_brief_builder(
        bb,
        run_id=run_id,
        reports_dir=reports_dir,
    )
    assert result.is_ok(), f"brief_builder returned error: {result}"

    output_path = result.unwrap()
    assert output_path.is_file()
    assert output_path.parent == reports_dir
    assert output_path.name == f"{run_id}.md"

    content = output_path.read_text(encoding="utf-8")

    # Top-level title should use the explanation claim.
    assert "The policy has measurable effects on outcomes." in content

    # Section headers should be present.
    assert "## Overview" in content
    assert "## Key terms" in content
    assert "## Timeline" in content

    # Term and timeline content should be rendered.
    assert "Randomized controlled trial" in content
    assert "Policy pilot launched" in content

    # Blackboard should record the path as a string.
    stored_path = bb.get(_PUBLIC_BRIEF_MD_KEY)
    assert isinstance(stored_path, str)
    assert stored_path == str(output_path)


def test_brief_builder_requires_some_artifacts() -> None:
    """Calling run_brief_builder on an empty Blackboard should fail.

    When no explanations, terms, or timeline events are present on the
    blackboard, the brief builder does not have any material to render
    and should therefore return an Err(Result) with an informative
    message rather than creating an empty file.
    """
    bb = Blackboard()

    result = run_brief_builder(bb, run_id="empty", reports_dir="unused")
    assert result.is_err()

    msg = str(result)
    assert "requires at least one" in msg
