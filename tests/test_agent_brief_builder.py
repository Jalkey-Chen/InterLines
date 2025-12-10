"""Tests for the AI-powered Brief Builder Agent.

Since the Brief Builder is now non-deterministic (LLM-based), these tests
focus on:
1. Integration: Ensuring the prompt is built correctly from blackboard artifacts.
2. I/O: Ensuring the file is written to the correct path.
3. Content Inclusion: Verifying that key facts from the input make it into the output.

Note: In a real CI environment, you should mock `interlines.llm.client.LLMClient`
to avoid API costs and flakiness.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from interlines.agents.brief_builder import (
    _PUBLIC_BRIEF_MD_KEY,
    run_brief_builder,
)
from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.explanation import EvidenceItem, ExplanationCard
from interlines.core.contracts.term import TermCard


def _make_artifacts() -> tuple[list[ExplanationCard], list[TermCard]]:
    """Create sample artifacts for testing."""
    evidence = [EvidenceItem(text="Evidence A", source="p1")]
    exp = ExplanationCard(
        kind="explanation.v1",
        version="1.0.0",
        confidence=0.9,
        claim="Quantum computers use qubits.",
        rationale="Unlike classical bits, qubits exist in superposition.",
        evidence=evidence,
        summary=None,
    )

    term = TermCard(
        kind="term.v1",
        version="1.0.0",
        confidence=0.8,
        term="Superposition",
        definition="The ability to be in multiple states at once.",
        aliases=[],
        examples=[],
        sources=["p1"],
    )
    return [exp], [term]


@patch("interlines.agents.brief_builder._get_llm_client")
def test_brief_builder_calls_llm_and_saves_file(mock_get_client: MagicMock, tmp_path: Path) -> None:
    """
    Test that the builder correctly serializes artifacts, prompts the LLM,
    and saves the generated text.
    """
    # 1. Setup Mock LLM
    mock_client_instance = MagicMock()
    # Simulate the LLM returning a nice Markdown document
    mock_client_instance.generate.return_value = """
# Quantum Computing: A Brief

**Why it matters**: It changes everything.

## The Core Concept
Quantum computers use **qubits**. Unlike classical bits, qubits exist in superposition.

## Glossary
* **Superposition**: The ability to be in multiple states at once.
"""
    mock_get_client.return_value = mock_client_instance

    # 2. Setup Blackboard
    exps, terms = _make_artifacts()
    bb = Blackboard()
    bb.put("explanations", exps)
    bb.put("terms", terms)
    # Optional: add empty timeline/notes to ensure it handles missing keys gracefully
    bb.put("timeline_events", [])

    reports_dir = tmp_path / "reports"
    run_id = "ai_brief_test"

    # 3. Run Agent
    result = run_brief_builder(bb, run_id=run_id, reports_dir=reports_dir)

    # 4. Verification
    assert result.is_ok()
    output_path = result.unwrap()

    # Check File I/O
    assert output_path.exists()
    assert output_path.name == "ai_brief_test.md"

    content = output_path.read_text(encoding="utf-8")

    # Check Content Inclusion (The "Soul" check)
    # We verify that the AI actually wrote the inputs we gave it
    assert "Quantum computers use qubits" in content
    assert "Superposition" in content

    # Check Blackboard Update
    assert bb.get(_PUBLIC_BRIEF_MD_KEY) == str(output_path)

    # Check Prompt Construction (verify input data flow)
    # Inspect the call args to ensure the JSON was passed correctly
    call_args = mock_client_instance.generate.call_args
    # call_args[0] are positional args: (messages,)
    prompts = call_args[0][0]
    # prompts is a list of dicts: [{"role": "system", ...}, {"role": "user", ...}]
    user_msg = prompts[1]["content"]

    assert "Quantum computers use qubits" in user_msg
    assert "explanations" in user_msg


def test_brief_builder_handles_empty_blackboard() -> None:
    """The builder should fail gracefully if there is absolutely no data."""
    bb = Blackboard()
    # No data put into blackboard

    result = run_brief_builder(bb, run_id="fail_test", reports_dir="unused")
    assert result.is_err()
    assert "No artifacts found" in str(result)
