"""Citizen-facing relevance agent.

This agent reads expert-facing :class:`ExplanationCard` objects from the
blackboard and asks an LLM (in a journalist/civic role) to produce
colloquial "why it matters" notes for different audiences.

It writes a list of :class:`RelevanceNote` instances back to the
blackboard under the ``"relevance_notes"`` key and returns them wrapped
in a :class:`Result`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.explanation import ExplanationCard
from interlines.core.contracts.relevance import RelevanceNote
from interlines.core.result import Result, err, ok
from interlines.llm.client import LLMClient

# Blackboard keys
_EXPLANATIONS_KEY = "explanations"
_RELEVANCE_NOTES_KEY = "relevance_notes"

# Model alias for the citizen agent.
_CITIZEN_MODEL_ALIAS = "citizen"


def _get_llm_client() -> LLMClient:
    """Return the shared LLM client for the citizen agent.

    Separated into a tiny helper so tests can monkeypatch this function and
    inject a fake client.
    """
    return LLMClient.from_env()


def _build_citizen_messages(
    explanations: Sequence[ExplanationCard],
) -> list[Mapping[str, str]]:
    """Build chat messages for the citizen-facing LLM call.

    Parameters
    ----------
    explanations:
        The list of expert-facing :class:`ExplanationCard` instances produced
        by the explainer agent.

    Returns
    -------
    list[Mapping[str, str]]
        Chat messages in the OpenAI / Gemini style:
        ``[{"role": "system", ...}, {"role": "user", ...}]``.
    """
    bullet_lines: list[str] = []
    for idx, card in enumerate(explanations, start=1):
        claim = (card.claim or "").strip()
        rationale = (card.rationale or "").strip()
        bullet_lines.append(f"{idx}. Claim: {claim}\n" f"   Expert view: {rationale}")

    expert_recape = "\n\n".join(bullet_lines) if bullet_lines else "No explanations available."

    system_content = """You are a civic-minded journalist explaining research and policy
to everyday citizens.

Your task is to turn expert explanations into a small set of
“why this matters to me” notes for different audiences.

Style guidelines:
- Use everyday language, as if explaining to a curious friend.
- Do NOT introduce any new facts, numbers, or events that are not already
  implied by the explanations. You may rephrase or slightly generalise,
  but stay within their content.
- Avoid heavy jargon; if a technical term appears, briefly gloss it.
- Each note should focus on a concrete audience or life situation
  (e.g., local voters, parents, workers, students).

Output format (VERY IMPORTANT):
Return only a single JSON object with this structure:

{
  "notes": [
    {
      "target": "short label for who/what this matters for",
      "rationale": "1-3 colloquial sentences explaining why it matters",
      "score": 0.0
    }
    // 2-4 items total ...
  ]
}

The "score" field must be between 0.0 and 1.0 and reflects how central
this note is to the overall topic.
"""

    user_content = (
        "Here are expert-style explanations of a research paper or policy.\n\n"
        f"{expert_recape}\n\n"
        "Based ONLY on the content above, write 2-4 citizen-facing notes about "
        "why this topic matters. Follow the JSON schema from the system message "
        "and do not add any new facts or events."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _read_explanations(
    bb: Blackboard,
    source_key: str,
) -> Result[list[ExplanationCard], str]:
    """Load and validate explanations from the blackboard.

    Returns an ``Err`` if the key is missing, not a list, or the list is empty.
    """
    raw: Any = bb.get(source_key)
    if raw is None:
        return err("Citizen agent requires 'explanations' on the blackboard.")

    if not isinstance(raw, list) or not all(isinstance(c, ExplanationCard) for c in raw):
        return err("Citizen agent expected 'explanations' to be a list[ExplanationCard].")

    if not raw:
        return err("Citizen agent received an empty list of explanations.")

    return ok(cast(list[ExplanationCard], raw))


def _call_citizen_llm(
    client: LLMClient,
    messages: Sequence[Mapping[str, str]],
) -> Result[str, str]:
    """Call the underlying LLM and wrap any transport errors in a Result."""
    try:
        raw = client.generate(
            messages,
            model=_CITIZEN_MODEL_ALIAS,
            temperature=0.4,
            max_tokens=800,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return err(f"Citizen agent LLM error: {exc}")

    return ok(raw)


def _note_from_json(node: Any) -> RelevanceNote | None:
    """Convert a single JSON note object into a :class:`RelevanceNote`.

    This helper keeps :func:`_parse_notes_json` simple enough to satisfy the
    linter's complexity limit, while keeping all field-munging logic in one
    place.
    """
    if not isinstance(node, Mapping):
        return None

    target = str(node.get("target", "")).strip()
    rationale = str(node.get("rationale", "")).strip()

    raw_score = node.get("score", 0.0)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0

    if score < 0.0:
        score = 0.0
    elif score > 1.0:
        score = 1.0

    if not target or not rationale:
        return None

    return RelevanceNote(
        kind="relevance.v1",
        version="1.0.0",
        confidence=score,
        target=target,
        rationale=rationale,
        score=score,
    )


def _parse_notes_json(raw: str) -> Result[list[RelevanceNote], str]:
    """Parse the LLM JSON payload into :class:`RelevanceNote` objects."""
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        return err(f"Citizen agent could not parse LLM JSON: {exc}")

    if not isinstance(payload, Mapping):
        return err("Citizen agent expected JSON object as root.")

    notes_field = payload.get("notes", [])
    if not isinstance(notes_field, list):
        return err("Citizen agent expected 'notes' to be a list in the JSON output.")

    notes: list[RelevanceNote] = []
    for item in notes_field:
        note = _note_from_json(item)
        if note is not None:
            notes.append(note)

    if not notes:
        return err("Citizen agent produced no usable relevance notes.")

    return ok(notes)


def run_citizen(
    bb: Blackboard,
    *,
    source_key: str = _EXPLANATIONS_KEY,
    target_key: str = _RELEVANCE_NOTES_KEY,
) -> Result[list[RelevanceNote], str]:
    """Run the citizen-facing relevance agent.

    This is the main entry point used by the pipeline.
    """
    # 1) Explanations from blackboard.
    explain_result = _read_explanations(bb, source_key)
    if explain_result.is_err():
        return err(explain_result.unwrap_err())
    explanations = explain_result.unwrap()

    # 2) LLM call.
    client = _get_llm_client()
    messages = _build_citizen_messages(explanations)
    llm_result = _call_citizen_llm(client, messages)
    if llm_result.is_err():
        return err(llm_result.unwrap_err())
    raw = llm_result.unwrap()

    # 3) Parse notes JSON.
    notes_result = _parse_notes_json(raw)
    if notes_result.is_err():
        return err(notes_result.unwrap_err())
    notes = notes_result.unwrap()

    # 4) Persist and return.
    bb.put(target_key, notes)
    return ok(notes)


__all__ = ["run_citizen", "_RELEVANCE_NOTES_KEY"]
