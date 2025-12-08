"""
Jargon agent: detect key technical terms and build TermCard objects.

This agent looks at the parsed source text (``parsed_chunks`` on the
blackboard) and asks an LLM to:

- Identify important domain-specific or technical terms.
- Provide plain-language definitions.
- Suggest aliases (other ways people might refer to the same idea).
- Provide one or more short usage examples.

The agent writes a list of :class:`TermCard` instances back to the
blackboard under the ``"terms"`` key and returns them wrapped in a
:class:`Result`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.term import TermCard
from interlines.core.result import Result, err, ok
from interlines.llm.client import LLMClient

# Blackboard keys
_PARSED_CHUNKS_KEY = "parsed_chunks"
_TERMS_KEY = "terms"

# Logical model alias for jargon detection.
_JARGON_MODEL_ALIAS = "jargon"


def _get_llm_client() -> LLMClient:
    """Return the shared LLM client for the jargon agent.

    Split into a helper so tests can monkeypatch this function and inject
    a fake client.
    """
    return LLMClient.from_env()


def _normalise_chunks(raw: Any) -> list[dict[str, str]]:
    """Coerce whatever is stored under ``parsed_chunks`` into a clean list.

    Expected shape (from the parser agent)::

        [{"id": "p1", "text": "First paragraph..."}, {"id": "p2", "text": "Second par..."}, ...]

    The parser is free to use int IDs; we convert everything to strings
    and drop empty texts.
    """
    if not isinstance(raw, list):
        return []

    out: list[dict[str, str]] = []
    for idx, item in enumerate(raw):
        if isinstance(item, Mapping):
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            pid = item.get("id")
            if pid is None:
                pid = f"p{idx}"
            out.append({"id": str(pid), "text": text})
        elif isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            out.append({"id": f"p{idx}", "text": text})

    return out


def _build_jargon_messages(
    chunks: Sequence[Mapping[str, str]],
) -> list[Mapping[str, str]]:
    """Build chat messages for the jargon-detection LLM call.

    Parameters
    ----------
    chunks:
        Cleaned paragraph chunks with ``id`` and ``text`` keys.

    Returns
    -------
    list[Mapping[str, str]]
        Chat messages in the OpenAI / Gemini style:
        ``[{"role": "system", ...}, {"role": "user", ...}]``.
    """
    para_lines: list[str] = []
    for chunk in chunks:
        pid = str(chunk.get("id", "")).strip()
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        para_lines.append(f"[{pid}] {text}")

    paragraphs_block = "\n".join(para_lines) if para_lines else "No text available."

    system_content = """You are a glossary writer for a public-interest website.

Given some paragraphs from a research or policy text, you will:

- Detect the most important technical or domain-specific terms that a
  non-expert reader might not recognise.
- For each term, write a short, plain-language definition.
- Suggest a few aliases (other names or phrases people might use).
- Provide one or two short usage examples in natural language.

Constraints:
- Base your definitions only on what is implied by the text. Do not add
  specific new facts, statistics, or external stories.
- Prefer terms that are central to understanding the text, not generic
  words like "policy", "people", or "data".

Output format (VERY IMPORTANT):
Return only a single JSON object with this structure:

{
  "terms": [
    {
      "term": "machine learning",
      "definition": "Short plain-language definition...",
      "aliases": ["ML"],
      "examples": [
        "For example, a system that learns from past cases to help officials rank applications."
      ],
      "confidence": 0.0,
      "sources": ["p1", "p3"]
    }
    // 3-7 items total ...
  ]
}

Rules:
- Use double quotes for all JSON strings.
- "aliases", "examples", and "sources" must be JSON arrays (can be empty).
- "confidence" must be a number between 0.0 and 1.0.
- Do not wrap the JSON in backticks or Markdown.
"""

    user_content = (
        "Here are the source paragraphs with IDs:\n\n"
        f"{paragraphs_block}\n\n"
        "Identify 3-7 key terms and return them in the JSON format described "
        "in the system message."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _parse_confidence(raw: Any, default: float = 0.8) -> float:
    """Parse a confidence value and clamp it into [0.0, 1.0]."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = default

    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _parse_str_list(raw: Any) -> list[str]:
    """Parse a sequence-like value into a cleaned list of strings."""
    items: list[str] = []
    # Reject non-sequences and plain strings/bytes early.
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        return items

    for entry in raw:
        text = str(entry).strip()
        if text:
            items.append(text)

    return items


def _term_from_json(node: Any) -> TermCard | None:
    """Convert a single JSON term object into a :class:`TermCard`.

    This helper keeps the main JSON parsing function simple while
    centralising all the field-level munging logic.
    """
    if not isinstance(node, Mapping):
        return None

    term = str(node.get("term", "")).strip()
    definition = str(node.get("definition", "")).strip()
    if not term or not definition:
        return None

    confidence = _parse_confidence(node.get("confidence", 0.8))
    aliases = _parse_str_list(node.get("aliases", []))
    examples = _parse_str_list(node.get("examples", []))
    sources = _parse_str_list(node.get("sources", []))

    return TermCard(
        kind="term.v1",
        version="1.0.0",
        confidence=confidence,
        term=term,
        definition=definition,
        aliases=aliases,
        examples=examples,
        sources=sources,
    )


def _parse_terms_json(raw: str) -> Result[list[TermCard], str]:
    """Parse the LLM JSON payload into :class:`TermCard` objects."""
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        return err(f"Jargon agent could not parse LLM JSON: {exc}")

    if not isinstance(payload, Mapping):
        return err("Jargon agent expected JSON object as root.")

    terms_field = payload.get("terms", [])
    if not isinstance(terms_field, list):
        return err("Jargon agent expected 'terms' to be a list in the JSON output.")

    terms: list[TermCard] = []
    for item in terms_field:
        card = _term_from_json(item)
        if card is not None:
            terms.append(card)

    if not terms:
        return err("Jargon agent produced no usable term cards.")

    return ok(terms)


def run_jargon(
    bb: Blackboard,
    *,
    source_key: str = _PARSED_CHUNKS_KEY,
    target_key: str = _TERMS_KEY,
) -> Result[list[TermCard], str]:
    """Run the jargon-detection agent.

    Parameters
    ----------
    bb:
        Shared :class:`Blackboard` instance for the current pipeline run.
    source_key:
        Blackboard key containing the parsed text chunks. Expected to be a
        list of mappings with ``id`` and ``text``.
    target_key:
        Blackboard key under which :class:`TermCard` objects will be written.

    Returns
    -------
    Result[list[TermCard], str]
        ``Ok(list[TermCard])`` on success, or ``Err(str)`` with a
        human-readable error message.
    """
    raw_chunks: Any = bb.get(source_key)
    chunks = _normalise_chunks(raw_chunks)
    if not chunks:
        return err("Jargon agent requires non-empty 'parsed_chunks' on the blackboard.")

    client = _get_llm_client()
    messages = _build_jargon_messages(chunks)

    try:
        raw = client.generate(
            messages,
            model=_JARGON_MODEL_ALIAS,
            temperature=0.3,
            max_tokens=900,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return err(f"Jargon agent LLM error: {exc}")

    terms_result = _parse_terms_json(raw)
    if terms_result.is_err():
        return err(terms_result.unwrap_err())
    terms = terms_result.unwrap()

    bb.put(target_key, terms)
    return ok(terms)


__all__ = ["run_jargon", "_TERMS_KEY"]
