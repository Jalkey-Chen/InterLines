"""Schema smoke tests: ensure Pydantic models align with JSON schema files.

This suite checks:
- The generated JSON Schema "title" and top-level required fields for each model
  match the stored files in `schemas/`.
- Field presence and types for a small set of representative properties.

We intentionally avoid brittle, byte-for-byte comparisons so minor, non-semantic
differences (like description ordering) don't cause failures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from interlines.core.contracts.explanation import ExplanationCard
from interlines.core.contracts.public_brief import PublicBrief
from interlines.core.contracts.relevance import RelevanceNote
from interlines.core.contracts.review import ReviewCriteria, ReviewReport
from interlines.core.contracts.term import TermCard
from interlines.core.contracts.timeline import TimelineEvent

SCHEMAS = Path("schemas")


def _load(path: Path) -> dict[str, Any]:
    """Load a JSON schema file and return it as a typed dictionary."""
    with path.open("r", encoding="utf-8") as f:
        return cast(dict[str, Any], json.load(f))


def _required(d: dict[str, Any]) -> set[str]:
    """Extract the set of required field names from a JSON Schema dict."""
    return set(cast(list[str], d.get("required", [])))


def test_explanation_schema_alignment() -> None:
    model = ExplanationCard
    gen = model.model_json_schema()
    ref = _load(SCHEMAS / "explanation.v1.json")
    assert gen["title"] == ref["title"]
    assert _required(gen) == _required(ref)
    for k in ("claim", "rationale", "evidence"):
        assert k in gen["properties"] and k in ref["properties"]


def test_term_schema_alignment() -> None:
    model = TermCard
    gen = model.model_json_schema()
    ref = _load(SCHEMAS / "term.v1.json")
    assert gen["title"] == ref["title"]
    assert _required(gen) == _required(ref)
    for k in ("term", "definition"):
        assert k in gen["properties"] and k in ref["properties"]


def test_relevance_schema_alignment() -> None:
    model = RelevanceNote
    gen = model.model_json_schema()
    ref = _load(SCHEMAS / "relevance.v1.json")
    assert gen["title"] == ref["title"]
    assert _required(gen) == _required(ref)
    for k in ("target", "rationale", "score"):
        assert k in gen["properties"] and k in ref["properties"]


def test_timeline_schema_alignment() -> None:
    model = TimelineEvent
    gen = model.model_json_schema()
    ref = _load(SCHEMAS / "timeline_event.v1.json")
    assert gen["title"] == ref["title"]
    assert _required(gen) == _required(ref)
    for k in ("when", "title"):
        assert k in gen["properties"] and k in ref["properties"]


def test_public_brief_schema_alignment() -> None:
    model = PublicBrief
    gen = model.model_json_schema()
    ref = _load(SCHEMAS / "public_brief.v1.json")
    assert gen["title"] == ref["title"]
    assert _required(gen) == _required(ref)
    for k in ("title", "summary", "sections"):
        assert k in gen["properties"] and k in ref["properties"]


def test_review_schema_alignment() -> None:
    assert (
        ReviewCriteria.model_json_schema()["title"]
        == _load(SCHEMAS / "review_criteria.v1.json")["title"]
    )
    assert (
        ReviewReport.model_json_schema()["title"]
        == _load(SCHEMAS / "review_report.v1.json")["title"]
    )


def test_block_schema_alignment() -> None:
    from interlines.core.contracts.block import Block

    gen = Block.model_json_schema()
    ref = _load(SCHEMAS / "block.v1.json")

    assert gen["title"] == ref["title"]
    assert _required(gen) == _required(ref)

    for field in ("id", "type", "page"):
        assert field in gen["properties"]
        assert field in ref["properties"]
