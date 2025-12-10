"""E2E tests for the LLM-backed explainer agent.

We mock the LLM client so that:
- No real HTTP call is made.
- We fully control the JSON payload returned by `generate()`.
- We can assert that:
    * `run_explainer()` reads `parsed_chunks` from the blackboard.
    * The explainer uses the "explainer" model alias.
    * Three `ExplanationCard` objects are produced.
    * Each card has non-empty `claim` / `rationale`.
    * `claims[]` and provenance IDs from the JSON are encoded into
      `ExplanationCard.evidence` entries (via `EvidenceItem.text` / `source`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from interlines.agents import explainer_agent
from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.explanation import EvidenceItem, ExplanationCard


class FakeLLMClient:
    """Simple in-memory fake for `LLMClient` used in explainer tests.

    The fake records each call to :meth:`generate` and returns a static JSON
    payload shaped like the real explainer response.
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
        """Record the call and return a fixed three-level explanation JSON.

        Parameters
        ----------
        messages:
            Chat-style messages passed to the LLM client.
        model:
            Logical model alias requested by the explainer agent.
        temperature:
            Sampling temperature (ignored by the fake, but recorded).
        max_tokens:
            Maximum generation length (ignored by the fake, but recorded).

        Returns
        -------
        str
            JSON string emulating the explainer LLM output.
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
            "one_sentence": {
                "claim": "The paper studies how AI influences public policy outcomes.",
                "rationale": "It summarises the central finding in a single sentence.",
                "claims": [
                    "AI tools are increasingly embedded in policy workflows.",
                ],
                "provenance_ids": ["p1"],
                "confidence": 0.9,
            },
            "three_paragraph": {
                "claim": "AI reshapes decision-making, data analysis, and citizen interaction.",
                "rationale": "Three short paragraphs expand on mechanisms and examples.",
                "claims": [
                    "AI supports large-scale text and data analysis.",
                    "Automation changes who participates in policy debates.",
                ],
                "provenance_ids": ["p1", "p2"],
                "confidence": 0.85,
            },
            "deep_dive": {
                "claim": (
                    "The deep explanation walks through methodological choices, "
                    "limits, and implications."
                ),
                "rationale": "It connects methods, results, and normative concerns.",
                "claims": [
                    "The authors rely on a mix of experiments and observational data.",
                    "They highlight risks of bias and unequal access.",
                    "They discuss how institutions can adapt over time.",
                ],
                "provenance_ids": ["p2"],
                "confidence": 0.8,
            },
        }

        return json.dumps(payload)


def test_explainer_e2e_with_mock_llm(monkeypatch: Any) -> None:
    """run_explainer() should create three ExplanationCards from parsed_chunks.

    This test exercises the agent end-to-end while replacing the real LLM
    client with a deterministic in-memory fake:

    1. Seed the blackboard with two parsed paragraphs (p1, p2).
    2. Patch `explainer_agent._get_llm_client` to return FakeLLMClient.
    3. Call `run_explainer(bb)` and unwrap the `Result`.
    4. Assert that:
       - Three ExplanationCard instances are produced.
       - They are written back to the blackboard under "explanations".
       - The fake client was called with model="explainer".
       - Evidence items encode claims[] and provenance IDs.
    """
    # 1) Prepare a blackboard with parser output.
    bb = Blackboard()
    bb.put(
        "parsed_chunks",
        [
            {"id": "p1", "text": "The paper analyses how AI tools affect policy decisions."},
            {
                "id": "p2",
                "text": ("It combines experiments with observational data and discusses risks."),
            },
        ],
    )

    # 2) Patch the explainer to use our fake LLM client.
    fake_client = FakeLLMClient()

    def fake_get_llm_client() -> FakeLLMClient:
        """Return the singleton fake client for this test run."""
        return fake_client

    monkeypatch.setattr(explainer_agent, "_get_llm_client", fake_get_llm_client)

    # 3) Run the agent.
    result = explainer_agent.run_explainer(bb)
    assert result.is_ok(), f"explainer returned error: {result}"

    cards = result.unwrap()
    assert isinstance(cards, list)
    assert len(cards) == 3

    # 4) Basic shape checks on the produced ExplanationCards.
    for card in cards:
        assert isinstance(card, ExplanationCard)
        assert card.claim.strip()
        assert card.rationale.strip()

    # Ensure they were also written back to the blackboard.
    stored = bb.get("explanations")
    assert isinstance(stored, list)
    assert stored == cards

    # 5) Check that the explainer used the correct model alias.
    assert fake_client.calls, "Fake LLM was never called."
    call = fake_client.calls[0]
    assert call["model"] == "explainer"

    # 6) DoD-specific checks: claims[] + provenance IDs should show up in evidence.
    # We encoded each sub-claim as an EvidenceItem(text=..., source="paragraphs: ...").
    all_sources: list[str] = []
    for card in cards:
        assert card.evidence, "Each explanation card should carry evidence items."
        for ev in card.evidence:
            assert isinstance(ev, EvidenceItem)
            assert ev.text.strip(), "EvidenceItem.text must be non-empty."
            src = ev.source or ""
            all_sources.append(src)
            # Source may be None in theory, but in our mapping it should mention paragraphs.
            assert "paragraphs:" in src

    joined_sources = " | ".join(all_sources)
    assert "p1" in joined_sources and "p2" in joined_sources
