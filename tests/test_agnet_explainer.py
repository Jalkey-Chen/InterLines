"""Unit tests for the explainer agent placeholder (Step 2.2).

Notes
-----
- This test assumes you placed the explainer code in:
    src/interlines/agents/explainer_agent.py
  and that it exposes:
    - `run_explainer_stub(bb, ...)`
    - `DEFAULT_LEVELS`
"""

from __future__ import annotations

from interlines.agents.explainer_agent import DEFAULT_LEVELS, run_explainer_stub
from interlines.core.blackboard.memory import Blackboard


def test_explainer_uses_parsed_chunks_and_writes_to_blackboard() -> None:
    """Explainer reads `parsed_chunks`, produces all levels, and writes back."""
    bb = Blackboard()
    # Seed parser-like output on the blackboard.
    bb.put("parsed_chunks", ["First paragraph about PKI.", "Second paragraph."])

    cards = run_explainer_stub(bb)

    # Basic shape checks
    assert isinstance(cards, list)
    assert len(cards) == len(DEFAULT_LEVELS)
    assert bb.get("explanations") == cards

    levels = {card["level"] for card in cards}
    assert levels == set(DEFAULT_LEVELS)

    # Seed text from the first chunk should appear in at least one rationale.
    assert any("First paragraph about PKI." in card["rationale"] for card in cards)


def test_explainer_handles_missing_parser_output_gracefully() -> None:
    """Explainer still produces explanations when `parsed_chunks` is missing."""
    bb = Blackboard()
    # No `parsed_chunks` key set here.

    cards = run_explainer_stub(bb)

    assert len(cards) == len(DEFAULT_LEVELS)
    assert bb.get("explanations") == cards
    # We should still get a non-empty, obviously stubby rationale.
    for card in cards:
        assert isinstance(card["rationale"], str)
        assert "[stub:" in card["rationale"]
