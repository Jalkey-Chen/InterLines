"""Unit tests for the Parser Agent placeholder (Step 2.1)."""

from __future__ import annotations

from interlines.agents.parser_agent import parser_agent
from interlines.core.blackboard.memory import Blackboard


def test_parser_splits_and_writes_to_blackboard() -> None:
    """Input text is split into `parsed_chunks` and written to the blackboard."""
    bb = Blackboard()
    text = (
        "InterLines (行间) turns expert language into public language.\n"
        "It also provides a historical lens.\n"
        "\n"
        "Agents collaborate through a blackboard to parse → translate → narrate.\n"
        "\n"
        "This is Step 2.1 — a placeholder parser agent."
    )

    chunks = parser_agent(text, bb, key="parsed_chunks", min_chars=2, make_trace=True)

    # Basic checks: chunking and write-through
    assert isinstance(chunks, list)
    assert len(chunks) == 3
    assert bb.get("parsed_chunks") == chunks

    # Trace snapshot recorded
    snaps = bb.traces()
    assert len(snaps) == 1
    assert "parser_agent" in (snaps[0].note or "")
