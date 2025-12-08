"""
History agent: build a timeline and evolution narrative from parsed text.

Responsibilities
----------------
- Read ``parsed_chunks`` from the blackboard (output of the parser agent).
- Ask an LLM (model alias: "history") to:
  * Extract key dated events as a timeline.
  * Write a short evolution narrative describing how the topic changes
    over time.
- Convert the JSON response into a list of :class:`TimelineEvent` objects.
- Mark any event that has no ``sources`` as needing verification by
  adding a "needs_review" tag.
- Write results back to the blackboard:
  * "timeline_events": list[TimelineEvent]
  * "evolution_narrative": str
- Return the timeline as a :class:`Result`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any, cast

from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.timeline import TimelineEvent
from interlines.core.result import Result, err, ok
from interlines.llm.client import LLMClient

# Blackboard keys
_PARSED_CHUNKS_KEY = "parsed_chunks"
_TIMELINE_EVENTS_KEY = "timeline_events"
_EVOLUTION_NARRATIVE_KEY = "evolution_narrative"

# Logical model alias for the history agent.
_HISTORY_MODEL_ALIAS = "history"


def _get_llm_client() -> LLMClient:
    """Return the shared LLM client for the history agent.

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


def _build_history_messages(
    chunks: Sequence[Mapping[str, str]],
) -> list[Mapping[str, str]]:
    """Build chat messages for the history / timeline LLM call.

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

    system_content = """You are a historian summarising how a policy, technology,
or idea has evolved over time.

Given some paragraphs from a research or policy text, you will:

1. Extract a small timeline of key events.
2. Write a short narrative (one or two paragraphs) describing how the
   topic develops over time.

Guidelines:
- Use only what is implied in the text. If the timing is vague, use
  approximate dates or years (e.g., "2010", "2020-01-01").
- Prefer events that change how people think or act (e.g., major laws,
  crises, technological shifts) rather than tiny details.
- For each event, include a list of paragraph IDs from the provided
  text that support it. If you cannot confidently link an event to
  specific paragraphs, leave the sources list empty.

Output format (VERY IMPORTANT):
Return only a single JSON object with this structure:

{
  "events": [
    {
      "when": "2010" or "2010-05-01",
      "title": "Short event title",
      "description": "1-3 sentences describing the event and why it matters.",
      "tags": ["optional", "keywords"],
      "sources": ["p1", "p3"],
      "confidence": 0.0
    }
    // 3-8 items total ...
  ],
  "narrative": "One or two paragraphs summarising the evolution over time."
}

Rules:
- Use double quotes for all JSON strings.
- "tags" and "sources" must be JSON arrays (can be empty).
- "confidence" must be a number between 0.0 and 1.0.
- Do not wrap the JSON in backticks or Markdown.
"""

    user_content = (
        "Here are the source paragraphs with IDs:\n\n"
        f"{paragraphs_block}\n\n"
        "Based only on this text, construct a concise timeline and an evolution "
        "narrative. Follow the JSON schema from the system message exactly."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _parse_confidence(raw: Any, default: float = 0.7) -> float:
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


def _parse_sources(raw: Any) -> list[str]:
    """Parse the sources field into a cleaned list of paragraph IDs."""
    sources: list[str] = []
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        return sources

    for entry in raw:
        text = str(entry).strip()
        if text:
            sources.append(text)

    return sources


def _parse_tags(raw: Any) -> list[str]:
    """Parse the tags field into a cleaned list of strings."""
    tags: list[str] = []
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        return tags

    for entry in raw:
        text = str(entry).strip()
        if text:
            tags.append(text)

    return tags


def _parse_when(raw: Any) -> str | None:
    """Parse the 'when' field into a normalised string representation.

    We accept:
    - String dates or years (e.g., "2010", "2010-05-01").
    - Integers representing years (e.g., 2010).
    - ``date`` or ``datetime`` instances (converted with ``isoformat()``).

    The returned value is always a string; :class:`TimelineEvent`
    will parse it as ``date | datetime`` at runtime.
    """
    if isinstance(raw, date | datetime):
        return raw.isoformat()

    if isinstance(raw, int | float):
        year = int(raw)
        return str(year)

    if isinstance(raw, str):
        value = raw.strip()
        return value or None

    return None


def _event_from_json(node: Any) -> TimelineEvent | None:
    """Convert a single JSON event object into a :class:`TimelineEvent`.

    If the event is missing mandatory fields (``when`` or ``title``), the
    function returns ``None`` and the caller will skip it.
    """
    if not isinstance(node, Mapping):
        return None

    when_raw = node.get("when")
    when_str = _parse_when(when_raw)
    if not when_str:
        return None

    title = str(node.get("title", "")).strip()
    if not title:
        return None

    description_raw = node.get("description")
    description = str(description_raw).strip() if description_raw is not None else None
    if description == "":
        description = None

    confidence = _parse_confidence(node.get("confidence", 0.7))
    sources = _parse_sources(node.get("sources", []))
    tags = _parse_tags(node.get("tags", []))

    # If the event has no explicit sources, mark it as "needs_review" for review.
    if not sources and "needs_review" not in tags:
        tags.append("needs_review")

    return TimelineEvent(
        kind="timeline_event.v1",
        version="1.0.0",
        confidence=confidence,
        # Pydantic accepts string input for date/datetime; the cast is for type checkers.
        when=cast("date | datetime", when_str),
        title=title,
        description=description,
        tags=tags,
        sources=sources,
    )


def _parse_timeline_json(raw: str) -> Result[tuple[list[TimelineEvent], str], str]:
    """Parse the LLM JSON payload into events and an evolution narrative.

    Returns
    -------
    Result[tuple[list[TimelineEvent], str], str]
        On success, an ``Ok`` wrapping a pair ``(events, narrative)``.
        On failure, an ``Err`` with a human-readable message.
    """
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        return err(f"History agent could not parse LLM JSON: {exc}")

    if not isinstance(payload, Mapping):
        return err("History agent expected JSON object as root.")

    events_raw = payload.get("events", [])
    if not isinstance(events_raw, list):
        return err("History agent expected 'events' to be a list in the JSON output.")

    events: list[TimelineEvent] = []
    for item in events_raw:
        event = _event_from_json(item)
        if event is not None:
            events.append(event)

    if not events:
        return err("History agent produced no usable timeline events.")

    narrative_raw = payload.get("narrative", "")
    narrative = str(narrative_raw).strip()

    return ok((events, narrative))


def run_history(
    bb: Blackboard,
    *,
    source_key: str = _PARSED_CHUNKS_KEY,
    events_key: str = _TIMELINE_EVENTS_KEY,
    narrative_key: str = _EVOLUTION_NARRATIVE_KEY,
) -> Result[list[TimelineEvent], str]:
    """Run the history / timeline agent.

    Parameters
    ----------
    bb:
        Shared :class:`Blackboard` instance for the current pipeline run.
    source_key:
        Blackboard key containing the parsed text chunks, expected to be
        a list of mappings with ``id`` and ``text``.
    events_key:
        Blackboard key under which :class:`TimelineEvent` objects will be
        written.
    narrative_key:
        Blackboard key under which the evolution narrative (str) will be
        written.

    Returns
    -------
    Result[list[TimelineEvent], str]
        ``Ok(list[TimelineEvent])`` on success, or ``Err(str)`` with a
        human-readable error message.
    """
    raw_chunks: Any = bb.get(source_key)
    chunks = _normalise_chunks(raw_chunks)
    if not chunks:
        return err("History agent requires non-empty 'parsed_chunks' on the blackboard.")

    client = _get_llm_client()
    messages = _build_history_messages(chunks)

    try:
        raw = client.generate(
            messages,
            model=_HISTORY_MODEL_ALIAS,
            temperature=0.4,
            max_tokens=1200,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return err(f"History agent LLM error: {exc}")

    parsed = _parse_timeline_json(raw)
    if parsed.is_err():
        return err(parsed.unwrap_err())
    events, narrative = parsed.unwrap()

    bb.put(events_key, events)
    bb.put(narrative_key, narrative)

    return ok(events)


__all__ = ["run_history", "_TIMELINE_EVENTS_KEY", "_EVOLUTION_NARRATIVE_KEY"]
