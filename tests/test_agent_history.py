"""Tests for the LLM-backed history (timeline) agent."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

from interlines.agents import history_agent
from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.timeline import TimelineEvent


class FakeLLMClient:
    """Simple fake for :class:`LLMClient` used in history-agent tests.

    The fake records each call to :meth:`generate` and returns a static
    JSON payload with a small set of timeline events and a narrative.
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
        """Record the call and return a fixed history JSON payload."""
        self.calls.append(
            {
                "messages": list(messages),
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

        payload: dict[str, Any] = {
            "events": [
                {
                    "when": "2010",
                    "title": "Early adoption of AI tools in policy analysis",
                    "description": (
                        "Researchers begin using AI systems to help analyse large "
                        "datasets for policy decisions."
                    ),
                    "tags": ["AI", "policy"],
                    "sources": ["p1"],
                    "confidence": 0.9,
                },
                {
                    "when": "2020-01-01",
                    "title": "Public debate over fairness and accountability",
                    "description": (
                        "Concerns about bias and transparency in AI-supported "
                        "decisions move into public and legislative debates."
                    ),
                    "tags": ["fairness"],
                    "sources": [],
                    "confidence": 0.6,
                },
            ],
            "narrative": (
                "Over time, AI moved from a niche research tool to a central part "
                "of policy analysis. As these systems gained influence, questions "
                "about fairness and accountability became harder to ignore."
            ),
        }

        return json.dumps(payload)


def test_history_agent_builds_timeline_and_narrative(monkeypatch: Any) -> None:
    """run_history() should create TimelineEvents and store the narrative.

    Steps
    -----
    1. Seed the blackboard with a small ``parsed_chunks`` list.
    2. Patch :func:`history_agent._get_llm_client` to return :class:`FakeLLMClient`.
    3. Call :func:`history_agent.run_history` and unwrap the :class:`Result`.
    4. Assert that:

       - A non-empty list of :class:`TimelineEvent` is produced.
       - Events are written back under the ``"timeline_events"`` key.
       - The narrative string is written under ``"evolution_narrative"``.
       - Events without sources are tagged with "needs_review".
       - The fake client was called with model alias ``"history"``.
    """
    # 1) Prepare blackboard with parser-style chunks.
    bb = Blackboard()
    bb.put(
        "parsed_chunks",
        [
            {
                "id": "p1",
                "text": "In 2010, early experiments used AI tools to assist with "
                "policy data analysis.",
            },
            {
                "id": "p2",
                "text": "By 2020, debates about fairness and accountability in AI "
                "reached legislators and the public.",
            },
        ],
    )

    # 2) Patch history agent to use our fake LLM client.
    fake_client = FakeLLMClient()

    def fake_get_llm_client() -> FakeLLMClient:
        """Return the singleton fake client for this test run."""
        return fake_client

    monkeypatch.setattr(history_agent, "_get_llm_client", fake_get_llm_client)

    # 3) Run the agent.
    result = history_agent.run_history(bb)
    assert result.is_ok(), f"history agent returned error: {result}"

    events = result.unwrap()
    assert isinstance(events, list)
    assert events, "Expected at least one TimelineEvent."

    # 4) Check TimelineEvent shape and the 'needs_review' tag rule.
    has_needs_review = False
    for event in events:
        assert isinstance(event, TimelineEvent)
        assert event.title.strip()
        assert 0.0 <= event.confidence <= 1.0
        # `when` should be parsed into a date or datetime by Pydantic.
        assert isinstance(event.when, date | datetime)

        if not event.sources:
            # Events without sources must be marked "needs_review".
            assert "needs_review" in event.tags
            has_needs_review = True

    assert has_needs_review, "Expected at least one 'needs_review' event for coverage."

    # Ensure events and narrative were also written back to the blackboard.
    stored_events = bb.get("timeline_events")
    assert isinstance(stored_events, list)
    assert stored_events == events

    narrative = bb.get("evolution_narrative")
    assert isinstance(narrative, str)
    assert narrative.strip()

    # 5) Check that the history agent used the correct model alias.
    assert fake_client.calls, "Fake LLM was never called."
    call = fake_client.calls[0]
    assert call["model"] == "history"

    # Sanity-check that the prompt was built as chat-style messages.
    messages = call["messages"]
    assert isinstance(messages, list)
    assert messages, "History agent should send at least one message."
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
