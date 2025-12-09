"""Unit tests for the new hybrid Parser Agent.

Tests cover:
1) Legacy stub behavior (no llm passed)
2) Full LLM-backed semantic parsing (FakeLLM with JSON-only output)

Strong contract: each parsed segment must contain:
- id: str
- text: str
- page: int | None
- type: str
"""

from __future__ import annotations

import json
from typing import Any

from interlines.agents.parser_agent import parser_agent
from interlines.core.blackboard.memory import Blackboard
from interlines.llm.client import LLMClient

# ---------------------------------------------------------------------
# Fake LLM (mypy-safe)
# ---------------------------------------------------------------------


class FakeLLM(LLMClient):
    """Minimal FakeLLM returning deterministic JSON for parser tests."""

    def __init__(self) -> None:
        # Do NOT call super().__init__(), avoids real config loading
        pass

    def generate(
        self,
        messages: Any,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return a deterministic JSON string mimicking LLM output."""
        return json.dumps(
            {
                "segments": [
                    {
                        "id": "p0",
                        "text": "Hello world.",
                        "page": 1,
                        "type": "paragraph",
                    }
                ]
            }
        )


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_parser_stub_mode_splits_text_and_writes_to_blackboard() -> None:
    """Stub mode (no llm) should still produce structured segments."""
    bb = Blackboard()
    text = "A first paragraph.\n" "\n" "A second paragraph.\n" "\n" "A third paragraph."

    chunks = parser_agent(
        text,
        bb,
        key="parsed_chunks",
        min_chars=2,
        make_trace=True,
        llm=None,  # force stub fallback behavior
    )

    assert isinstance(chunks, list)
    assert len(chunks) == 3
    assert all(isinstance(x, dict) for x in chunks)

    # Strong contract: id, text must exist
    for seg in chunks:
        assert "id" in seg
        assert "text" in seg
        assert isinstance(seg["text"], str)
        assert seg["text"] != ""

    # Written to blackboard
    assert bb.get("parsed_chunks") == chunks

    # Trace exists
    traces = bb.traces()
    assert len(traces) == 1
    assert "parser_agent" in (traces[0].note or "")


def test_parser_llm_mode_produces_semantic_segments() -> None:
    """LLM-backed parser should produce fully structured segments."""
    bb = Blackboard()
    llm: LLMClient = FakeLLM()

    chunks = parser_agent(
        "some raw text",
        bb,
        key="parsed_chunks",
        llm=llm,  # triggers semantic parsing
        make_trace=True,
    )

    assert isinstance(chunks, list)
    assert len(chunks) == 1

    seg = chunks[0]
    assert isinstance(seg, dict)

    # Full contract:
    assert seg["id"] == "p0"
    assert seg["text"] == "Hello world."
    assert seg["page"] == 1
    assert seg["type"] == "paragraph"

    # Written to blackboard
    assert bb.get("parsed_chunks") == chunks

    # Trace exists
    traces = bb.traces()
    assert len(traces) == 1
    assert "parser_agent" in (traces[0].note or "")
