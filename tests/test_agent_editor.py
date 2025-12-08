"""Tests for the editor/validator agent.

These tests exercise the lightweight quality gate implemented in
:mod:`interlines.agents.editor_agent`:

- Basic flow: aggregate readability and criteria computation.
- Missing-provenance detection for explanations, terms, and timeline.
- Behaviour when the blackboard is effectively empty.
"""

from __future__ import annotations

from datetime import date

from interlines.agents.editor_agent import _REVIEW_REPORT_KEY, run_editor
from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.explanation import EvidenceItem, ExplanationCard
from interlines.core.contracts.relevance import RelevanceNote
from interlines.core.contracts.review import ReviewCriteria, ReviewReport
from interlines.core.contracts.term import TermCard
from interlines.core.contracts.timeline import TimelineEvent


def _make_explanation(with_sources: bool = True) -> ExplanationCard:
    """Create a small ExplanationCard for testing.

    Parameters
    ----------
    with_sources:
        If True, the card will carry one EvidenceItem with a non-empty
        ``source`` string; otherwise the evidence list will be empty.

    Returns
    -------
    ExplanationCard
        A minimal explanation instance suitable for editor tests.
    """
    evidence: list[EvidenceItem] = []
    if with_sources:
        evidence.append(
            EvidenceItem(
                text="The study provides empirical evidence.",
                source="paragraphs: p1, p2",
            ),
        )
    return ExplanationCard(
        kind="explanation.v1",
        version="1.0.0",
        confidence=0.9,
        claim="The policy has measurable effects.",
        rationale="The authors run a controlled experiment to measure impact.",
        evidence=evidence,
        summary=None,
    )


def _make_term(with_sources: bool = True) -> TermCard:
    """Create a small TermCard for testing.

    Parameters
    ----------
    with_sources:
        If True, attach a single synthetic source identifier. When False,
        the term will have an empty ``sources`` list so that the editor
        can flag missing provenance.

    Returns
    -------
    TermCard
        A glossary entry suitable for editor tests.
    """
    srcs: list[str] = ["p1"] if with_sources else []
    return TermCard(
        kind="term.v1",
        version="1.0.0",
        confidence=0.8,
        term="Randomized controlled trial",
        definition="An experiment where units are randomly assigned.",
        aliases=["RCT"],
        examples=["The study uses an RCT in several regions."],
        sources=srcs,
    )


def _make_timeline_event(with_sources: bool = True) -> TimelineEvent:
    """Create a simple timeline event for testing.

    Parameters
    ----------
    with_sources:
        If True, attach a synthetic document-level source. If False, the
        event will have an empty ``sources`` list, triggering an editor
        provenance warning.

    Returns
    -------
    TimelineEvent
        A dated event used for history-related checks.
    """
    srcs: list[str] = ["doc:history"] if with_sources else []
    return TimelineEvent(
        kind="timeline_event.v1",
        version="1.0.0",
        confidence=0.75,
        when=date(2020, 1, 1),
        title="Policy pilot launched",
        description="Initial small scale deployment of the new policy.",
        tags=["pilot"],
        sources=srcs,
    )


def _make_relevance_note() -> RelevanceNote:
    """Create a basic RelevanceNote for testing.

    Returns
    -------
    RelevanceNote
        A relevance note with a non-empty rationale and high score.
    """
    return RelevanceNote(
        kind="relevance_note.v1",
        version="1.0.0",
        confidence=0.85,
        target="brief:main",
        rationale="This section connects the findings to everyday decisions.",
        score=0.9,
    )


def test_run_editor_basic_flow_builds_review_report() -> None:
    """Editor agent should produce a ReviewReport and store it on the blackboard.

    This is a positive-path, end-to-end style test:
    - We seed the blackboard with explanations, relevance notes, terms,
      timeline events, and an evolution narrative.
    - We run the editor and expect an Ok(Result) with a ReviewReport.
    - The report's criteria should have scores in [0, 1] and reasonably
      high completeness.
    - The report should also be written back to the blackboard under
      the `_REVIEW_REPORT_KEY`.
    """
    bb = Blackboard()
    bb.put("explanations", [_make_explanation(with_sources=True)])
    bb.put("relevance_notes", [_make_relevance_note()])
    bb.put("terms", [_make_term(with_sources=True)])
    bb.put("timeline_events", [_make_timeline_event(with_sources=True)])
    bb.put(
        "evolution_narrative",
        "Over two decades, the policy evolved from pilots to nationwide rollout.",
    )

    result = run_editor(bb)
    assert result.is_ok(), f"editor returned error: {result}"

    report = result.unwrap()
    assert isinstance(report, ReviewReport)

    # The same object should be persisted on the blackboard.
    stored = bb.get(_REVIEW_REPORT_KEY)
    assert stored is report

    criteria = report.criteria
    assert isinstance(criteria, ReviewCriteria)

    # All criteria and the overall score should be within [0, 1].
    assert 0.0 <= criteria.accuracy <= 1.0
    assert 0.0 <= criteria.clarity <= 1.0
    assert 0.0 <= criteria.completeness <= 1.0
    assert 0.0 <= criteria.safety <= 1.0
    assert 0.0 <= report.overall <= 1.0

    # Since we provided all core artifact types, completeness should be high.
    assert criteria.completeness >= 0.7

    # Comments should mention the readability summary line.
    joined_comments = "\n".join(report.comments)
    assert "[readability] Aggregate readability score:" in joined_comments


def test_run_editor_marks_missing_provenance() -> None:
    """Editor agent should flag artifacts that lack provenance.

    We intentionally construct:
    - an explanation with no evidence,
    - a term with an empty sources list,
    - a timeline event with an empty sources list.

    The editor is expected to:
    - emit provenance-focused comments for each artifact type,
    - add matching action strings that suggest how to fix the issues.
    """
    bb = Blackboard()
    # Explanation without any evidence.
    bb.put("explanations", [_make_explanation(with_sources=False)])
    # Term without sources.
    bb.put("terms", [_make_term(with_sources=False)])
    # Timeline event without sources.
    bb.put("timeline_events", [_make_timeline_event(with_sources=False)])

    result = run_editor(bb)
    assert result.is_ok(), f"editor returned error: {result}"

    report = result.unwrap()

    joined_comments = "\n".join(report.comments)
    joined_actions = "\n".join(report.actions)

    # We expect at least one provenance-related comment per artifact type.
    assert "Explanation[0]" in joined_comments
    assert "Term[0]" in joined_comments
    assert "TimelineEvent[0]" in joined_comments

    # And matching actions suggesting where to add sources.
    assert "explanation[0]" in joined_actions
    assert "term 'Randomized controlled trial'" in joined_actions
    assert "timeline event 'Policy pilot launched'" in joined_actions


def test_run_editor_with_empty_blackboard_is_conservative_but_succeeds() -> None:
    """Even with no artifacts, editor should return a valid (low-scoring) report.

    The editor is designed to degrade gracefully when there are no
    artifacts on the blackboard:
    - No exceptions should be raised.
    - The returned ReviewReport should have low completeness and
      zero clarity (no text segments to score).
    - Comments should explicitly mention that key artifact types
      were missing.
    """
    bb = Blackboard()
    result = run_editor(bb)
    assert result.is_ok(), f"editor returned error: {result}"

    report = result.unwrap()
    assert isinstance(report, ReviewReport)
    criteria = report.criteria

    # With no artifacts, completeness should be low and clarity minimal.
    assert criteria.completeness < 0.5
    assert criteria.clarity == 0.0

    joined_comments = "\n".join(report.comments)
    # The editor should explicitly note missing artifact types.
    assert "No ExplanationCard objects found." in joined_comments
    assert "No RelevanceNote objects found." in joined_comments
    assert "No TermCard objects found." in joined_comments
