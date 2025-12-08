"""
Editor/validator agent: lightweight quality gate over pipeline outputs.

Responsibilities (commit 1)
---------------------------
- Collect key textual artifacts from the blackboard:
  * ExplanationCard list under "explanations"
  * RelevanceNote list under "relevance_notes"
  * TermCard list under "terms"
  * Optional evolution narrative under "evolution_narrative"
- Compute a simple readability score using
  :func:`aggregate_readability`.
- Run a few rule-based checks to approximate:
  * accuracy   - presence of evidence for explanations
  * clarity    - mapped from readability score
  * completeness - presence of the main artifact types
  * safety     - currently a high default (no red flags detector yet)
- Build a :class:`ReviewReport` with numeric criteria and free-form
  comments describing any issues that should be fixed.

Later commits will extend this agent to:
- Mark missing provenance explicitly.
- Integrate more advanced factuality/bias checks if needed.
"""

from __future__ import annotations

from typing import Any

from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.explanation import ExplanationCard
from interlines.core.contracts.relevance import RelevanceNote
from interlines.core.contracts.review import ReviewCriteria, ReviewReport
from interlines.core.contracts.term import TermCard
from interlines.core.evals.readability import aggregate_readability
from interlines.core.result import Result, ok

# Blackboard keys used by the editor/validator.
_EXPLANATIONS_KEY = "explanations"
_RELEVANCE_NOTES_KEY = "relevance_notes"
_TERMS_KEY = "terms"
_EVOLUTION_NARRATIVE_KEY = "evolution_narrative"
_REVIEW_REPORT_KEY = "review_report"


def _as_list(value: Any) -> list[Any]:
    """Coerce a value into a list for easier iteration."""
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _collect_explanation_segments(bb: Blackboard) -> tuple[list[str], list[str], bool]:
    """Collect text segments and issues from explanation cards.

    Returns
    -------
    tuple[list[str], list[str], bool]
        A triple ``(segments, comments, has_explanations)`` where:

        - ``segments`` are text fragments contributing to readability.
        - ``comments`` describe structural issues in explanations.
        - ``has_explanations`` indicates whether any ExplanationCard
          instances were found at all.
    """
    segments: list[str] = []
    comments: list[str] = []

    explanations_raw = bb.get(_EXPLANATIONS_KEY)
    explanations: list[Any] = _as_list(explanations_raw)
    explanation_count = 0

    for item in explanations:
        if isinstance(item, ExplanationCard):
            explanation_count += 1
            if item.claim.strip():
                segments.append(item.claim)
            if item.rationale.strip():
                segments.append(item.rationale)
            else:
                comments.append(
                    "[explanations] Missing rationale in an explanation card.",
                )
        else:
            if item is not None:
                comments.append(
                    "[explanations] Non-ExplanationCard item encountered.",
                )

    if explanation_count == 0:
        comments.append("[explanations] No ExplanationCard objects found.")

    return segments, comments, explanation_count > 0


def _collect_relevance_segments(bb: Blackboard) -> tuple[list[str], list[str], bool]:
    """Collect text segments and issues from relevance notes."""
    segments: list[str] = []
    comments: list[str] = []

    notes_raw = bb.get(_RELEVANCE_NOTES_KEY)
    notes: list[Any] = _as_list(notes_raw)
    note_count = 0

    for item in notes:
        if isinstance(item, RelevanceNote):
            note_count += 1
            if item.rationale.strip():
                segments.append(item.rationale)
            else:
                comments.append("[relevance_notes] Note has empty rationale.")
        else:
            if item is not None:
                comments.append(
                    "[relevance_notes] Non-RelevanceNote item encountered.",
                )

    if note_count == 0:
        comments.append("[relevance_notes] No RelevanceNote objects found.")

    return segments, comments, note_count > 0


def _collect_term_segments(bb: Blackboard) -> tuple[list[str], list[str], bool]:
    """Collect text segments and issues from term cards (glossary)."""
    segments: list[str] = []
    comments: list[str] = []

    terms_raw = bb.get(_TERMS_KEY)
    terms: list[Any] = _as_list(terms_raw)
    term_count = 0

    for item in terms:
        if isinstance(item, TermCard):
            term_count += 1
            if item.definition.strip():
                segments.append(item.definition)
            else:
                comments.append("[terms] Term has empty definition.")
            for example in item.examples:
                if example.strip():
                    segments.append(example)
        else:
            if item is not None:
                comments.append("[terms] Non-TermCard item encountered.")

    if term_count == 0:
        comments.append("[terms] No TermCard objects found.")

    return segments, comments, term_count > 0


def _collect_narrative_segment(bb: Blackboard) -> tuple[list[str], list[str]]:
    """Collect the evolution narrative (if any) and related issues."""
    segments: list[str] = []
    comments: list[str] = []

    narrative = bb.get(_EVOLUTION_NARRATIVE_KEY)
    if isinstance(narrative, str) and narrative.strip():
        segments.append(narrative)
    elif narrative is None:
        # Absence of a narrative is acceptable; no comment needed.
        pass
    else:
        comments.append(
            "[evolution_narrative] Unexpected type; expected str or None.",
        )

    return segments, comments


def _score_criteria(
    *,
    readability: float,
    has_explanations: bool,
    has_notes: bool,
    has_terms: bool,
) -> ReviewCriteria:
    """Construct a :class:`ReviewCriteria` instance from simple signals.

    The mapping is intentionally simple and easy to explain:
    - clarity      ~ readability score
    - completeness ~ coverage of explanations, notes, and terms
    - accuracy     ~ presence of evidence in explanations (approx.)
    - safety       ~ currently a fixed high score (no red flags logic)
    """
    # Clarity is directly mapped from readability.
    clarity = readability

    # Completeness: start from 1.0 and subtract penalties.
    completeness = 1.0
    if not has_explanations:
        completeness -= 0.3
    if not has_notes:
        completeness -= 0.3
    if not has_terms:
        completeness -= 0.2
    if completeness < 0.0:
        completeness = 0.0

    # Accuracy: for now, assume moderate accuracy unless the editor
    # later adds more detailed checks. We keep it centred around 0.7.
    accuracy = 0.7

    # Safety: fixed high score until a dedicated safety classifier exists.
    safety = 0.9

    return ReviewCriteria(
        kind="review_criteria.v1",
        version="1.0.0",
        confidence=0.7,
        accuracy=accuracy,
        clarity=clarity,
        completeness=completeness,
        safety=safety,
    )


def run_editor(bb: Blackboard) -> Result[ReviewReport, str]:
    """Run the editor/validator agent and return a :class:`ReviewReport`.

    Steps
    -----
    1. Collect text segments and structural comments from the blackboard.
    2. Compute a readability score from all segments.
    3. Construct :class:`ReviewCriteria` based on readability and the
       presence/absence of core artifact types.
    4. Aggregate an overall score as the average of the four criteria.
    5. Store the :class:`ReviewReport` under ``"review_report"`` on the
       blackboard and return it.

    Parameters
    ----------
    bb:
        Shared :class:`Blackboard` instance for the current pipeline run.

    Returns
    -------
    Result[ReviewReport, str]
        ``Ok(ReviewReport)`` on success, or ``Err(str)`` if the editor
        cannot compute a meaningful report.
    """
    # Collect segments and issue comments from each component.
    exp_segments, exp_comments, has_explanations = _collect_explanation_segments(bb)
    note_segments, note_comments, has_notes = _collect_relevance_segments(bb)
    term_segments, term_comments, has_terms = _collect_term_segments(bb)
    narrative_segments, narrative_comments = _collect_narrative_segment(bb)

    all_segments: list[str] = []
    all_segments.extend(exp_segments)
    all_segments.extend(note_segments)
    all_segments.extend(term_segments)
    all_segments.extend(narrative_segments)

    comments: list[str] = []
    comments.extend(exp_comments)
    comments.extend(note_comments)
    comments.extend(term_comments)
    comments.extend(narrative_comments)

    readability = aggregate_readability(all_segments)

    criteria = _score_criteria(
        readability=readability,
        has_explanations=has_explanations,
        has_notes=has_notes,
        has_terms=has_terms,
    )

    # Overall is the simple mean of the four criteria dimensions.
    overall = (criteria.accuracy + criteria.clarity + criteria.completeness + criteria.safety) / 4.0

    # Add a short summary comment about readability.
    comments.append(
        f"[readability] Aggregate readability score: {readability:.2f} in [0,1].",
    )

    report = ReviewReport(
        kind="review_report.v1",
        version="1.0.0",
        confidence=0.7,
        overall=overall,
        criteria=criteria,
        comments=comments,
        actions=[],
    )

    bb.put(_REVIEW_REPORT_KEY, report)
    return ok(report)


__all__ = ["run_editor", "_REVIEW_REPORT_KEY"]
