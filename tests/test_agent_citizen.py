"""E2E tests for the LLM-backed citizen relevance agent.

We mock the LLM client so that:

- No real HTTP calls are made.
- We fully control the JSON payload returned by ``generate()``.
- We can assert that:

  * ``run_citizen()`` reads ``explanations`` from the blackboard.
  * The citizen agent uses the ``"citizen"`` model alias.
  * A list of :class:`RelevanceNote` objects is produced.
  * Notes are written back to the blackboard under ``"relevance_notes"``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from interlines.agents import citizen_agent
from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.explanation import ExplanationCard
from interlines.core.contracts.relevance import RelevanceNote


class FakeLLMClient:
    """Simple in-memory fake for :class:`LLMClient` used in citizen tests.

    The fake records each call to :meth:`generate` and returns a static JSON
    payload shaped like the real citizen-agent response.
    """

    def __init__(self) -> None:
        # Each entry records one call to ``generate()`` with the arguments
        # we care about for assertions.
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Record the call and return a fixed relevance-notes JSON payload.

        Parameters
        ----------
        messages:
            Chat-style messages passed to the LLM client.
        model:
            Logical model alias requested by the citizen agent.
        temperature:
            Sampling temperature (ignored by the fake, but recorded).
        max_tokens:
            Maximum generation length (ignored by the fake, but recorded).

        Returns
        -------
        str
            JSON string emulating the citizen LLM output.
        """
        self.calls.append(
            {
                "messages": list(messages),
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

        payload: dict[str, Any] = {
            "notes": [
                {
                    "target": "local voters",
                    "rationale": (
                        "These findings matter for local voters because they show how "
                        "AI tools can subtly change which voices are heard when "
                        "policies are made."
                    ),
                    "score": 0.9,
                },
                {
                    "target": "public officials",
                    "rationale": (
                        "For public officials, the research highlights concrete risks "
                        "around bias and unequal access when adopting AI in government."
                    ),
                    "score": 0.8,
                },
            ]
        }

        return json.dumps(payload)


def test_citizen_e2e_with_mock_llm(monkeypatch: Any) -> None:
    """run_citizen() should create RelevanceNotes from ExplanationCards.

    This test exercises the citizen agent end-to-end while replacing the real
    LLM client with a deterministic in-memory fake:

    1. Seed the blackboard with a couple of :class:`ExplanationCard` instances.
    2. Patch :func:`citizen_agent._get_llm_client` to return :class:`FakeLLMClient`.
    3. Call :func:`citizen_agent.run_citizen` and unwrap the :class:`Result`.
    4. Assert that:

       - At least one :class:`RelevanceNote` is produced.
       - They are written back to the blackboard under ``"relevance_notes"``.
       - The fake client was called with ``model="citizen"``.
    """
    # 1) Prepare a blackboard with explainer-style output.
    bb = Blackboard()
    explanations = [
        ExplanationCard(
            kind="explanation.v1",
            version="1.0.0",
            confidence=0.9,
            claim="The paper analyses how AI tools affect public policy decisions.",
            rationale=(
                "It explains how automated systems change decision processes "
                "and who gets heard in debates."
            ),
        ),
        ExplanationCard(
            kind="explanation.v1",
            version="1.0.0",
            confidence=0.8,
            claim="It also discusses risks and trade-offs.",
            rationale="It highlights issues such as bias, unequal access, and long-term impact.",
        ),
    ]
    bb.put("explanations", explanations)

    # 2) Patch the citizen agent to use our fake LLM client.
    fake_client = FakeLLMClient()

    def fake_get_llm_client() -> FakeLLMClient:
        """Return the singleton fake client for this test run."""
        return fake_client

    monkeypatch.setattr(citizen_agent, "_get_llm_client", fake_get_llm_client)

    # 3) Run the agent.
    result = citizen_agent.run_citizen(bb)
    assert result.is_ok(), f"citizen agent returned error: {result}"

    notes = result.unwrap()
    assert isinstance(notes, list)
    assert len(notes) >= 1

    # 4) Basic shape checks on the produced RelevanceNotes.
    for note in notes:
        assert isinstance(note, RelevanceNote)
        assert note.target.strip()
        assert note.rationale.strip()
        assert 0.0 <= note.score <= 1.0
        # Confidence should mirror score in the current implementation.
        assert 0.0 <= note.confidence <= 1.0

    # Ensure they were also written back to the blackboard.
    stored = bb.get("relevance_notes")
    assert isinstance(stored, list)
    assert stored == notes

    # 5) Check that the citizen agent used the correct model alias.
    assert fake_client.calls, "Fake LLM was never called."
    call = fake_client.calls[0]
    assert call["model"] == "citizen"

    # Sanity-check that the prompt was built as chat-style messages.
    messages = call["messages"]
    assert isinstance(messages, list)
    assert messages, "Citizen agent should send at least one message."
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
