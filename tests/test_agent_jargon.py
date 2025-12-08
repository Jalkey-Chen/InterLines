"""Tests for the LLM-backed jargon (TermCard) agent."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from interlines.agents import jargon_agent
from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.term import TermCard


class FakeLLMClient:
    """Simple fake for :class:`LLMClient` used in jargon-agent tests.

    The fake records each call to :meth:`generate` and returns a static JSON
    payload with one or more term entries.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Record the call and return a fixed term-cards JSON payload."""
        self.calls.append(
            {
                "messages": list(messages),
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

        payload: dict[str, Any] = {
            "terms": [
                {
                    "term": "algorithmic bias",
                    "definition": (
                        "When a computer system systematically favours some people "
                        "or groups over others because of how it was built or trained."
                    ),
                    "aliases": ["AI bias", "model bias"],
                    "examples": [
                        "For example, a hiring tool that unfairly screens out "
                        "qualified candidates from a particular background."
                    ],
                    "confidence": 0.9,
                    "sources": ["p1"],
                },
                {
                    "term": "training data",
                    "definition": (
                        "The collection of examples a model learns from before "
                        "it is used in the real world."
                    ),
                    "aliases": ["learning data"],
                    "examples": [
                        "For instance, past job applications used to teach a "
                        "model how to rank new applicants."
                    ],
                    "confidence": 0.8,
                    "sources": ["p1", "p2"],
                },
            ]
        }

        return json.dumps(payload)


def test_jargon_agent_builds_term_cards(monkeypatch: Any) -> None:
    """run_jargon() should create TermCard objects from parsed chunks.

    Steps:
    1. Seed the blackboard with a small ``parsed_chunks`` list.
    2. Patch :func:`jargon_agent._get_llm_client` to return :class:`FakeLLMClient`.
    3. Call :func:`jargon_agent.run_jargon` and unwrap the :class:`Result`.
    4. Assert that:

       - A non-empty list of :class:`TermCard` is produced.
       - The cards are written back under the ``"terms"`` key.
       - The fake client was called with model alias ``"jargon"``.
    """
    # 1) Prepare blackboard with parser-style chunks.
    bb = Blackboard()
    bb.put(
        "parsed_chunks",
        [
            {"id": "p1", "text": "This paper studies algorithmic bias in hiring tools."},
            {
                "id": "p2",
                "text": "It explains how training data can shape who gets shortlisted.",
            },
        ],
    )

    # 2) Patch jargon agent to use our fake LLM client.
    fake_client = FakeLLMClient()

    def fake_get_llm_client() -> FakeLLMClient:
        """Return the singleton fake client for this test run."""
        return fake_client

    monkeypatch.setattr(jargon_agent, "_get_llm_client", fake_get_llm_client)

    # 3) Run the agent.
    result = jargon_agent.run_jargon(bb)
    assert result.is_ok(), f"jargon agent returned error: {result}"

    terms = result.unwrap()
    assert isinstance(terms, list)
    assert terms, "Expected at least one TermCard."

    # 4) Check TermCard shape.
    for card in terms:
        assert isinstance(card, TermCard)
        assert card.term.strip()
        assert card.definition.strip()
        # Aliases and examples should be lists (may be empty, but here we know they are not).
        assert isinstance(card.aliases, list)
        assert isinstance(card.examples, list)
        assert all(isinstance(a, str) for a in card.aliases)
        assert all(isinstance(e, str) for e in card.examples)
        # Confidence should be clamped to [0, 1].
        assert 0.0 <= card.confidence <= 1.0
        # Kind/version should match our convention.
        assert card.kind == "term.v1"
        assert card.version == "1.0.0"

    # Ensure they were also written back to the blackboard.
    stored = bb.get("terms")
    assert isinstance(stored, list)
    assert stored == terms

    # 5) Check that the jargon agent used the correct model alias.
    assert fake_client.calls, "Fake LLM was never called."
    call = fake_client.calls[0]
    assert call["model"] == "jargon"

    # Sanity-check that the prompt was built as chat-style messages.
    messages = call["messages"]
    assert isinstance(messages, list)
    assert messages, "Jargon agent should send at least one message."
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
