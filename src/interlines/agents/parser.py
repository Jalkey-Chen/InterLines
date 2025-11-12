"""Parser Agent (placeholder): split input text into `parsed_chunks`.

This lightweight "agent" is intentionally minimal for Step 2.1. It:
1) Normalizes line endings.
2) Splits the input into paragraphs using blank lines as delimiters.
3) Strips whitespace and drops empty paragraphs.
4) Writes the resulting `list[str]` to the in-memory blackboard under a key
   (default: ``"parsed_chunks"``), and optionally records a trace snapshot.

Why paragraphs?
---------------
For early pipeline prototyping, paragraphs are a sensible unit: they are human-
legible, stable across formatting tweaks, and easy to reassemble downstream.

API
---
- `parser_agent(text, bb, key="parsed_chunks", min_chars=1, make_trace=True)`
    Returns the list of chunks written to the blackboard (pure data; easy to test).

Notes
-----
- This is a *placeholder* agent; later steps may enrich chunks with metadata,
  token counts, provenance, or sentence-level segmentation.
- We avoid extra dependencies here; `re` is sufficient.

Examples
--------
>>> from interlines.core.blackboard.memory import Blackboard
>>> bb = Blackboard()
>>> chunks = parser_agent("A\\n\\nB", bb)
>>> chunks
['A', 'B']
>>> bb.get("parsed_chunks")
['A', 'B']
"""

from __future__ import annotations

import re

from interlines.core.blackboard.memory import Blackboard

_BLANKS = re.compile(r"\n\s*\n+", flags=re.MULTILINE)


def _split_paragraphs(text: str, *, min_chars: int) -> list[str]:
    """Split `text` by blank lines into non-empty paragraphs.

    Parameters
    ----------
    text : str
        Raw input text to split.
    min_chars : int
        Minimum number of non-whitespace characters to keep a chunk.

    Returns
    -------
    list[str]
        Cleaned paragraph chunks.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    parts = [p.strip() for p in _BLANKS.split(text)]
    out: list[str] = []
    for p in parts:
        # Collapse internal whitespace for stability, but keep sentence spacing.
        cleaned = re.sub(r"[ \t]+", " ", p).strip()
        if len(cleaned.replace(" ", "")) >= max(0, min_chars):
            out.append(cleaned)
    return out


def parser_agent(
    text: str,
    bb: Blackboard,
    *,
    key: str = "parsed_chunks",
    min_chars: int = 1,
    make_trace: bool = True,
) -> list[str]:
    """Split `text` into paragraph chunks and write them to the blackboard.

    Parameters
    ----------
    text : str
        Input text blob.
    bb : Blackboard
        In-memory blackboard where the result will be stored.
    key : str, default "parsed_chunks"
        Blackboard key used to store the chunks.
    min_chars : int, default 1
        Minimum number of non-space characters for a paragraph to be kept.
    make_trace : bool, default True
        If True, record a trace snapshot after writing the chunks.

    Returns
    -------
    list[str]
        The list of parsed chunks (also stored in the blackboard under `key`).
    """
    chunks = _split_paragraphs(text, min_chars=min_chars)
    bb.put(key, chunks)
    if make_trace:
        bb.trace(f"parser_agent: {len(chunks)} chunks -> {key}")
    return chunks


__all__ = ["parser_agent"]
