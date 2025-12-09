"""LLM-backed Parser Agent: from raw documents to `parsed_chunks`.

This module implements the **semantic Parser Agent** for InterLines.

It replaces the earlier placeholder that only split plain text into
paragraphs and instead introduces a two-stage design:

1. **Extraction stage (see ``parser_extractor.py``)**
   - Handles I/O and filesystem paths.
   - Normalises TXT/PDF/DOCX/image inputs into *coarse* page- or
     file-level blocks.
   - Avoids fine-grained semantics: it only says “this is a chunk of
     text on page 3” or “this is an image at path X”.

2. **Semantic parsing stage (this module)**
   - Optionally calls an LLM to segment and label the extracted blocks.
   - Produces a list of paragraph-like segments with identifiers and
     text, written to the blackboard under ``"parsed_chunks"``.
   - Remains strictly JSON-only in its LLM interaction (no free-form
     prose), so downstream components receive predictable data.

Backwards compatibility
-----------------------
Historically, :func:`parser_agent` accepted **plain text** and wrote a
list of strings to the blackboard under ``"parsed_chunks"``:

.. code-block:: python

    ["Paragraph 1", "Paragraph 2", ...]

For now, we preserve compatibility with the explainer and pipeline by
still writing a **list of mappings** shaped like:

.. code-block:: python

    [{"id": "p0", "text": "Paragraph 1"}, {"id": "p1", "text": "Paragraph 2"}, ...]

This is exactly the shape that :func:`_normalise_chunks` in
``explainer_agent.py`` expects and will continue to work even as the
Parser's internal implementation evolves.

Usage
-----
High-level entry point (recommended):

.. code-block:: python

    from interlines.core.blackboard.memory import Blackboard
    from interlines.llm.client import LLMClient
    from interlines.agents.parser_agent import parser_agent

    bb = Blackboard()
    llm = LLMClient.from_env(default_model_alias="balanced")

    # Inline text (no files)
    chunks = parser_agent("Some long text...", bb, llm=llm)

    # Or a path to a local file (TXT/PDF/DOCX/Image)
    chunks = parser_agent("paper_example.pdf", bb, llm=llm)

    bb.get("parsed_chunks")  # -> list[dict[id, text, ...]]

Design goals
------------
- **Hybrid**: reuse format-specific extractors, avoid duplicating I/O.
- **LLM-optional**: if the LLM call fails or is unavailable, fall back
  to a deterministic, paragraph-based splitter.
- **Contract-first**: always produce a stable shape for
  ``parsed_chunks`` so that other agents (Explainer, Jargon, Citizen)
  do not need to change with every Parser iteration.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from interlines.agents.parser_extractor import (
    extract_from_docx,
    extract_from_image,
    extract_from_pdf,
    extract_from_text,
)
from interlines.core.blackboard.memory import Blackboard
from interlines.llm.client import LLMClient

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Blackboard key under which the parsed segments are stored.
_PARSED_KEY: str = "parsed_chunks"

#: Default logical model alias for the parser's LLM calls.
#: We intentionally reuse the generic "balanced" alias so that the
#: parser works even before a dedicated "parser" entry is added to the
#: model registry.
_PARSER_MODEL_ALIAS: str = "balanced"

#: Internal regex for the legacy paragraph splitter fallback.
_BLANKS = re.compile(r"\n\s*\n+", flags=re.MULTILINE)


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #


def _normalise_path(path: str | Path) -> Path:
    """Normalise a user-provided path into an absolute :class:`Path`.

    Parameters
    ----------
    path:
        A filesystem path as a string or :class:`Path`.

    Returns
    -------
    Path
        A normalised, absolute :class:`Path` instance.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist on disk.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")
    return p


def _looks_like_path(value: str) -> bool:
    """Heuristic to decide whether a string is more likely a path or inline text.

    The heuristic is intentionally simple:

    - Strings with newlines are considered *inline text*.
    - Strings containing typical path separators (``/`` or ``\\``) and no
      newlines are considered *paths*.

    Parameters
    ----------
    value:
        Raw string passed by the caller.

    Returns
    -------
    bool
        ``True`` if this looks like a filesystem path, ``False`` otherwise.
    """
    if "\n" in value or "\r" in value:
        return False
    return ("/" in value) or ("\\" in value)


def _split_paragraphs(text: str, *, min_chars: int = 1) -> list[str]:
    """Legacy paragraph splitter used as a deterministic fallback.

    This implements the original stub behaviour: normalise line endings,
    split on blank lines, compress internal whitespace, and drop short or
    empty paragraphs.

    Parameters
    ----------
    text:
        Raw input text blob.
    min_chars:
        Minimum number of non-whitespace characters required for a
        paragraph to be kept.

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
        cleaned = re.sub(r"[ \t]+", " ", p).strip()
        if len(cleaned.replace(" ", "")) >= max(0, min_chars):
            out.append(cleaned)
    return out


def _serialise_blocks_to_chunks(blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert internal block-like records into ``parsed_chunks`` shape.

    The explainer and other agents expect ``parsed_chunks`` to be a list
    of dictionaries with at least:

    - ``id``: stable string identifier
    - ``text``: non-empty paragraph text

    Any additional fields are preserved but ignored by existing
    consumers.

    Parameters
    ----------
    blocks:
        Sequence of mapping-like objects, each representing a block with
        at least ``id`` and ``text`` keys where possible.

    Returns
    -------
    list[dict[str, Any]]
        Normalised list suitable for storing in the blackboard.
    """
    chunks: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        if not isinstance(block, Mapping):
            continue

        raw_text = block.get("text", "")
        text = str(raw_text).strip()
        if not text:
            # Skip empty segments; they only add noise downstream.
            continue

        bid = block.get("id")
        if bid is None:
            bid = f"p{index}"

        node: dict[str, Any] = dict(block)
        node["id"] = str(bid)
        node["text"] = text
        chunks.append(node)

    return chunks


def _build_llm_messages(blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Build chat messages instructing the LLM to return JSON-only segments.

    The system prompt positions the model as a careful parser whose job
    is to segment an academic or policy text into semantically meaningful
    pieces, *without* changing the underlying meaning or adding new
    facts.

    The user prompt lists the input blocks with their IDs and page
    numbers, followed by explicit instructions to return JSON of the
    following form (no extra keys):

    .. code-block:: json

        {
          "segments": [
            {
              "id": "p0",
              "text": "First paragraph...",
              "page": 1,
              "type": "paragraph"
            },
            ...
          ]
        }

    Parameters
    ----------
    blocks:
        Coarse block dictionaries with (at least) ``id``, ``text`` and
        optionally ``page``.

    Returns
    -------
    list[dict[str, str]]
        Messages ready to be passed into :class:`LLMClient.generate`.
    """
    system_msg = {
        "role": "system",
        "content": (
            "You are a careful document parser for an academic or policy text. "
            "Your job is to split the source into semantically meaningful "
            "segments (paragraphs, headings, bullet points) and to return "
            "ONLY JSON. Do not add new facts. Do not write explanations."
        ),
    }

    lines: list[str] = []
    for block in blocks:
        bid = str(block.get("id", "b0"))
        page = block.get("page")
        text = str(block.get("text", "")).strip()
        page_label = f"(page {page}) " if page is not None else ""
        lines.append(f"[{bid}] {page_label}{text}")

    numbered_source = "\n\n".join(lines)

    user_msg = {
        "role": "user",
        "content": (
            "You are given coarse document blocks with IDs and optional page numbers.\n"
            "Split them into finer segments, preserving order.\n\n"
            "Source blocks:\n\n"
            f"{numbered_source}\n\n"
            "Return ONLY JSON with this exact shape:\n\n"
            "{\n"
            '  "segments": [\n'
            "    {\n"
            '      "id": "p0",\n'
            '      "text": "...",\n'
            '      "page": 1,\n'
            '      "type": "paragraph"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- Copy text from the input; you may merge or split, but never invent content.\n"
            "- `id` values must be unique strings.\n"
            "- `page` should be inherited from the source block where possible.\n"
            "- `type` must be one of: paragraph, heading, bullet, figure, table.\n"
            "- Do not include any keys other than `segments`, `id`, `text`, `page`, `type`."
        ),
    }

    return [system_msg, user_msg]


def _parse_llm_segments(
    raw: str,
    *,
    fallback_blocks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Parse the LLM JSON payload into a list of segment dictionaries.

    This helper is intentionally defensive:

    - If the payload is not valid JSON, we fall back to the coarse blocks.
    - If the JSON does not match the expected structure, we also fall
      back to the coarse blocks.

    Parameters
    ----------
    raw:
        Raw string returned in :attr:`LLMResponse.text`.
    fallback_blocks:
        Coarse blocks to fall back to when parsing fails.

    Returns
    -------
    list[dict[str, Any]]
        Segment dictionaries suitable for normalisation via
        :func:`_serialise_blocks_to_chunks`.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return [dict(b) for b in fallback_blocks]

    if isinstance(payload, Mapping) and "segments" in payload:
        segments = payload["segments"]
    else:
        # Allow bare list as a lenient alternative.
        segments = payload

    if not isinstance(segments, list):
        return [dict(b) for b in fallback_blocks]

    out: list[dict[str, Any]] = []
    for index, item in enumerate(segments):
        if not isinstance(item, Mapping):
            continue
        node = dict(item)
        # Normalise fields with conservative defaults.
        if "id" not in node:
            node["id"] = f"p{index}"
        if "text" not in node:
            node["text"] = ""
        out.append(node)

    if not out:
        return [dict(b) for b in fallback_blocks]

    return out


# --------------------------------------------------------------------------- #
# Core Agent implementation
# --------------------------------------------------------------------------- #


class ParserAgent:
    """Hybrid Parser Agent orchestrating extraction + LLM segmentation.

    The :class:`ParserAgent` is a small, testable wrapper around:

    - **Format-specific extraction** — delegated to functions in
      :mod:`interlines.agents.parser_extractor`.
    - **LLM-backed segmentation** — a JSON-only call to
      :class:`~interlines.llm.client.LLMClient`.

    It does *not* know about blackboards or pipelines directly; those
    concerns live in the :func:`parser_agent` convenience wrapper.

    Parameters
    ----------
    llm:
        LLM client used for semantic segmentation. In unit tests this
        can be a fake or stub that returns a known JSON payload.
    model_alias:
        Logical alias passed into :meth:`LLMClient.generate`. By default
        we reuse ``"balanced"`` so the parser works without any extra
        configuration in the model registry.
    """

    def __init__(self, llm: LLMClient, model_alias: str = _PARSER_MODEL_ALIAS) -> None:
        self._llm = llm
        self._model_alias = model_alias

    # ----------------------- extraction phase ----------------------- #

    def _extract_blocks(self, input_data: str | Path) -> list[dict[str, Any]]:
        """Extract coarse blocks from inline text or a local file.

        Parameters
        ----------
        input_data:
            Either a raw text blob or a filesystem path. If the value
            looks like a path *and* the file exists, we dispatch based
            on file extension; otherwise it is treated as inline text.

        Returns
        -------
        list[dict[str, Any]]
            Coarse blocks represented as dictionaries with at least
            ``id`` and ``text`` fields where possible.
        """
        # Inline text case (most common in early usage and tests)
        if isinstance(input_data, str) and not _looks_like_path(input_data):
            blocks = extract_from_text(input_data)
            return [b.model_dump() for b in blocks]

        # File path case
        path = _normalise_path(input_data)
        suffix = path.suffix.lower()

        if suffix == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
            blocks = extract_from_text(text)
        elif suffix == ".pdf":
            blocks = extract_from_pdf(path)
        elif suffix in {".docx", ".doc"}:
            blocks = extract_from_docx(path)
        elif suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            blocks = extract_from_image(path)
        else:
            # Conservatively treat unknown formats as raw text.
            text = path.read_text(encoding="utf-8", errors="ignore")
            blocks = extract_from_text(text)

        return [b.model_dump() for b in blocks]

    # ----------------------- semantic phase ------------------------ #

    def semantic_parse(self, input_data: str | Path) -> list[dict[str, Any]]:
        """Perform full semantic parsing and return normalised segments.

        The method combines both stages:

        1. Extract coarse blocks from ``input_data``.
        2. Call the LLM with these blocks and parse the returned JSON
           segments.
        3. Fall back to deterministic paragraph splitting if anything
           goes wrong (missing API key, invalid JSON, etc.).
        4. Normalise the result into the ``parsed_chunks`` shape.

        Parameters
        ----------
        input_data:
            Either inline text or a file path supported by the extractor
            layer (TXT, PDF, DOCX, image formats).

        Returns
        -------
        list[dict[str, Any]]
            Parsed segments suitable for writing into the blackboard
            under :data:`_PARSED_KEY`.
        """
        coarse_blocks = self._extract_blocks(input_data)

        # If we have no coarse blocks at all, return early.
        if not coarse_blocks:
            return []

        # Attempt LLM-based segmentation; on failure we fall back to a
        # deterministic paragraph splitter applied to the concatenated
        # text of the coarse blocks.
        try:
            messages = _build_llm_messages(coarse_blocks)
            raw = self._llm.generate(messages, model=self._model_alias)
            segments = _parse_llm_segments(raw, fallback_blocks=coarse_blocks)
        except Exception:
            # Fallback: treat the entire content as one big text blob and
            # re-use the legacy paragraph logic to at least produce
            # stable paragraphs.
            joined_text = "\n\n".join(str(b.get("text", "")) for b in coarse_blocks)
            paras = _split_paragraphs(joined_text, min_chars=1)
            segments = [
                {"id": f"p{idx}", "text": para, "page": None, "type": "paragraph"}
                for idx, para in enumerate(paras)
            ]

        return _serialise_blocks_to_chunks(segments)


# --------------------------------------------------------------------------- #
# Public entry point (backwards-compatible wrapper)
# --------------------------------------------------------------------------- #


def parser_agent(
    input_data: str | Path,
    bb: Blackboard,
    *,
    key: str = _PARSED_KEY,
    min_chars: int = 1,
    make_trace: bool = True,
    llm: LLMClient | None = None,
) -> list[dict[str, Any]]:
    """Parse raw input into semantic segments and write them to the blackboard.

    This is the **public, backwards-compatible** entry point used by the
    pipeline and tests. It supports two modes:

    1. **LLM mode** (recommended)
       If ``llm`` is provided, the function uses :class:`ParserAgent` to
       run the full extraction + LLM segmentation pipeline and writes
       the resulting list of segment dictionaries to the blackboard
       under ``key`` (default: ``"parsed_chunks"``).

    2. **Legacy stub mode**
       If ``llm`` is ``None``, the function falls back to the original
       behaviour of splitting *inline text* into paragraphs (ignoring
       file paths) and emitting a list of mappings with ``id`` and
       ``text``. This ensures that existing tests and the explainer
       agent continue to work even before the LLM-backed parser is fully
       wired into every call site.

    Parameters
    ----------
    input_data:
        Either raw text or a filesystem path. In legacy stub mode,
        non-text inputs are treated as plain strings and **not** opened
        as files.
    bb:
        In-memory blackboard used to store the parsed segments.
    key:
        Blackboard key used to store the result. Defaults to
        :data:`_PARSED_KEY` (``"parsed_chunks"``).
    min_chars:
        Minimum number of non-space characters to keep a paragraph in
        legacy mode. Ignored in LLM mode (the model controls segmentation).
    make_trace:
        If ``True``, append a human-readable trace line to the
        blackboard after writing the chunks.
    llm:
        Optional LLM client. If provided, enables the full hybrid parser
        behaviour; if omitted, the function behaves like the original
        paragraph-based stub.

    Returns
    -------
    list[dict[str, Any]]
        The list of normalised segment dictionaries written into the
        blackboard under ``key``.
    """
    # ------------------------ LLM-backed path ------------------------ #
    if llm is not None:
        agent = ParserAgent(llm)
        chunks = agent.semantic_parse(input_data)
        bb.put(key, chunks)
        if make_trace:
            bb.trace(f"parser_agent (LLM): {len(chunks)} segments -> {key}")
        return chunks

    # ------------------------ legacy stub path ----------------------- #
    # Keep the behaviour of the original parser: only operate on plain
    # text, ignoring any notion of paths or formats.
    if not isinstance(input_data, str):
        text = str(input_data)
    else:
        text = input_data

    paras = _split_paragraphs(text, min_chars=min_chars)
    chunks = [{"id": f"p{idx}", "text": para} for idx, para in enumerate(paras)]

    bb.put(key, chunks)
    if make_trace:
        bb.trace(f"parser_agent (stub): {len(chunks)} chunks -> {key}")

    return chunks


__all__ = ["ParserAgent", "parser_agent"]
