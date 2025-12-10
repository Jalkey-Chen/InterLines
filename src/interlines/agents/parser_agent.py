"""LLM-backed Parser Agent with semantic refinement.

This module implements the *semantic* Parser Agent for InterLines.

It builds on the previous foundation (hybrid extractor + basic JSON
segments) and extends it with **LLM-only refinement**:

- The LLM is now instructed to return *richer* segment objects with:
  - ``id``: stable identifier
  - ``text``: full segment text
  - ``page``: page number (when known)
  - ``type``: ``paragraph | heading | bullet | table | figure``
  - ``sentences``: list of sentence strings
  - ``metadata``:
      - ``numbers``: numeric expressions (years, percentages, etc.)
      - ``entities``: important named entities or institutions
      - ``markers``: discourse markers (however, therefore, for example…)

- The Python side remains conservative:
  - It validates the JSON shape.
  - It fills in defaults when fields are missing.
  - It avoids re-interpreting content: semantics are delegated to the LLM.

Backwards compatibility
-----------------------
- The *public* entry point :func:`parser_agent` keeps the same signature.
- Stub mode (``llm=None``) is preserved, but now emits segments conforming
  to the richer schema (with sensible defaults).
- Downstream consumers such as Explainer / Jargon / Citizen can continue
  to rely on the ``parsed_chunks`` key, now with more useful metadata.

Usage
-----
Typical usage (LLM mode):

.. code-block:: python

    from interlines.core.blackboard.memory import Blackboard
    from interlines.llm.client import LLMClient
    from interlines.agents.parser_agent import parser_agent

    bb = Blackboard()
    llm = LLMClient.from_env(default_model_alias="balanced")

    segments = parser_agent("paper.pdf", bb, llm=llm)
    # bb.get("parsed_chunks") -> list of segment dicts with id/text/page/type/sentences/metadata

The Parser Agent itself does not deal with HTTP, CLI, or API schemas; it
is a pure, in-process component that other layers orchestrate.
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

#: Default model alias used by the parser when calling the LLM.
_PARSER_MODEL_ALIAS: str = "balanced"

#: Legacy paragraph splitter used in stub/fallback mode.
_BLANKS = re.compile(r"\n\s*\n+", flags=re.MULTILINE)

#: Simple sentence splitter pattern for fallback defaults.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #


def _normalise_path(path: str | Path) -> Path:
    """Normalise a user-provided path into an absolute :class:`Path`.

    Parameters
    ----------
    path:
        Raw path as string or :class:`Path`.

    Returns
    -------
    Path
        Absolute, expanded :class:`Path`.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")
    return p


def _looks_like_path(value: str) -> bool:
    """Heuristic to decide whether a string is a path or inline text.

    Rules
    -----
    - Contains newline -> treat as text.
    - Contains '/' or '\\' and no newlines -> treat as path.
    """
    if "\n" in value or "\r" in value:
        return False
    return ("/" in value) or ("\\" in value)


def _split_paragraphs(text: str, *, min_chars: int = 1) -> list[str]:
    """Split text into paragraphs by blank lines for stub/fallback mode.

    The paragraphs are:
    - Normalised to LF line endings.
    - Trimmed of leading/trailing whitespace.
    - Filtered to keep only chunks with at least ``min_chars`` non-space
      characters.

    Parameters
    ----------
    text:
        Raw text blob.
    min_chars:
        Minimum non-space characters required to keep a paragraph.

    Returns
    -------
    list[str]
        Cleaned paragraph strings.
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


def _fallback_sentences(text: str) -> list[str]:
    """Derive sentence list from a text blob in a conservative way.

    This is only used as a *default* when the LLM does not supply
    ``sentences`` explicitly. It is deliberately simple and language-
    agnostic to avoid overfitting to any particular style.
    """
    text = (text or "").strip()
    if not text:
        return []
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    return parts or [text]


def _ensure_metadata(node: dict[str, Any]) -> dict[str, Any]:
    """Ensure a segment node has a well-formed ``metadata`` structure.

    The LLM is instructed to return a ``metadata`` object with keys:
    ``numbers``, ``entities``, and ``markers``. This helper fills in
    missing keys with empty lists to simplify downstream consumption.
    """
    raw_meta = node.get("metadata")
    if not isinstance(raw_meta, Mapping):
        meta: dict[str, Any] = {}
    else:
        meta = dict(raw_meta)

    for key in ("numbers", "entities", "markers"):
        value = meta.get(key)
        if not isinstance(value, list):
            meta[key] = []
    node["metadata"] = meta
    return node


def _serialise_blocks_to_chunks(blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalise block-like mappings into the canonical segment dict shape.

    The canonical shape for segments is:

    .. code-block:: json

        {
          "id": "p0",
          "text": "Some text...",
          "page": 1,
          "type": "paragraph",
          "sentences": ["Some text..."],
          "metadata": {
            "numbers": [],
            "entities": [],
            "markers": []
          }
        }

    Any missing fields are filled with conservative defaults to keep the
    contract stable for downstream agents.
    """
    chunks: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        if not isinstance(block, Mapping):
            continue

        node: dict[str, Any] = dict(block)

        # id
        bid = node.get("id", f"p{index}")
        node["id"] = str(bid)

        # text
        raw_text = node.get("text", "")
        text = str(raw_text).strip()
        if not text:
            # Empty segments are ignored entirely.
            continue
        node["text"] = text

        # type
        seg_type = node.get("type", "paragraph")
        node["type"] = str(seg_type) if seg_type is not None else "paragraph"

        # page
        page_val = node.get("page", None)
        if page_val is not None:
            try:
                page_int = int(page_val)
            except (TypeError, ValueError):
                page_int = None
        else:
            page_int = None
        node["page"] = page_int

        # sentences
        sentences = node.get("sentences")
        if not isinstance(sentences, list):
            node["sentences"] = _fallback_sentences(text)
        else:
            node["sentences"] = [str(s).strip() for s in sentences if str(s).strip()]

        # metadata
        node = _ensure_metadata(node)

        chunks.append(node)

    return chunks


def _build_llm_messages(blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Build chat messages instructing the LLM to return refined segments.

    The prompt asks the LLM to:

    - Split the input blocks into semantically coherent segments.
    - Label each segment with a type (paragraph / heading / bullet / etc.).
    - Provide a list of sentences and simple metadata (numbers, entities,
      discourse markers).
    - Return **JSON only**, in a stable, contract-first shape.
    """
    system_msg = {
        "role": "system",
        "content": (
            "You are a careful document parser for academic and policy texts. "
            "Your job is to split the source into semantically meaningful segments "
            "and return ONLY JSON, with no explanations or commentary."
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
            "Split them into refined segments, preserving logical order.\n\n"
            "Source blocks:\n\n"
            f"{numbered_source}\n\n"
            "Return ONLY JSON with this exact shape:\n\n"
            "{\n"
            '  "segments": [\n'
            "    {\n"
            '      "id": "p0",\n'
            '      "text": "Segment text...",\n'
            '      "page": 1,\n'
            '      "type": "paragraph",\n'
            '      "sentences": ["Sentence 1.", "Sentence 2."],\n'
            '      "metadata": {\n'
            '        "numbers": ["2020", "35%"],\n'
            '        "entities": ["United Nations"],\n'
            '        "markers": ["however"]\n'
            "      }\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- Do not invent facts that are not in the source.\n"
            "- `id` values must be unique strings.\n"
            "- `page` should be inherited from the source block whenever possible.\n"
            "- `type` must be one of: paragraph, heading, bullet, table, figure.\n"
            "- `sentences` must be a list of sentence strings.\n"
            "- `metadata.numbers` should list years, percentages, and other numeric expressions.\n"
            "- `metadata.entities` should list important named entities and institutions.\n"
            "- `metadata.markers` should list discourse markers (e.g., however, therefore).\n"
            "- Do not include any keys other than: segments/id/text/page/type/sentences/metadata."
        ),
    }

    return [system_msg, user_msg]


def _parse_llm_segments(
    raw: str,
    *,
    fallback_blocks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Parse the LLM JSON string into a list of segment dictionaries.

    The function is intentionally defensive:

    - If the payload is not valid JSON, the coarse blocks are used
      instead.
    - If the JSON does not contain a ``segments`` list, the coarse blocks
      are used instead.
    - If the resulting segments are empty or malformed, the coarse blocks
      are used instead.

    The output is **not** yet normalised; it still needs to go through
    :func:`_serialise_blocks_to_chunks`.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return [dict(b) for b in fallback_blocks]

    if isinstance(payload, Mapping) and "segments" in payload:
        segments = payload["segments"]
    else:
        # Allow a bare list as a lenient alternative.
        segments = payload

    if not isinstance(segments, list):
        return [dict(b) for b in fallback_blocks]

    out: list[dict[str, Any]] = []
    for index, item in enumerate(segments):
        if not isinstance(item, Mapping):
            continue
        node = dict(item)
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
    """Hybrid Parser Agent that orchestrates extraction + LLM refinement.

    The ParserAgent performs three conceptual steps:

    1. **Extraction** via :mod:`parser_extractor`
       TXT/PDF/DOCX/image sources are converted into coarse page-level
       Blocks.

    2. **LLM segmentation + refinement**
       The Blocks are passed to the LLM, which returns a JSON payload of
       refined segments enriched with type, sentences, and metadata.

    3. **Normalisation**
       The segments are normalised into a stable contract shape for
       storage under ``parsed_chunks`` in the blackboard.

    Parameters
    ----------
    llm:
        LLM client used for semantic segmentation and refinement.
    model_alias:
        Logical model alias used when calling :meth:`LLMClient.generate`.
    """

    def __init__(self, llm: LLMClient, model_alias: str = _PARSER_MODEL_ALIAS) -> None:
        self._llm = llm
        self._model_alias = model_alias

    # ----------------------- extraction phase ----------------------- #

    def _extract_blocks(self, input_data: str | Path) -> list[dict[str, Any]]:
        """Extract coarse Blocks from inline text or local files.

        If the input is a string and does not look like a path, it is
        treated as raw text. Otherwise, the file extension determines
        which extractor to use.
        """
        if isinstance(input_data, str) and not _looks_like_path(input_data):
            blocks = extract_from_text(input_data)
            return [b.model_dump() for b in blocks]

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
            # Fallback to treating the file as raw text.
            text = path.read_text(encoding="utf-8", errors="ignore")
            blocks = extract_from_text(text)

        return [b.model_dump() for b in blocks]

    # ----------------------- semantic phase ------------------------ #

    def semantic_parse(self, input_data: str | Path) -> list[dict[str, Any]]:
        """Perform full semantic parsing and refinement.

        Steps
        -----
        1. Extract coarse Blocks from ``input_data``.
        2. Call the LLM to obtain refined segments (id/text/page/type/
           sentences/metadata).
        3. Fall back to deterministic paragraph splitting if the LLM
           call fails for any reason.
        4. Normalise the output into the canonical segment shape.
        """
        coarse_blocks = self._extract_blocks(input_data)
        if not coarse_blocks:
            return []

        try:
            messages = _build_llm_messages(coarse_blocks)
            raw = self._llm.generate(messages, model=self._model_alias)
            segments = _parse_llm_segments(raw, fallback_blocks=coarse_blocks)
        except Exception:
            # Fallback: reuse the legacy paragraph splitter on concatenated text.
            joined_text = "\n\n".join(str(b.get("text", "")) for b in coarse_blocks)
            paras = _split_paragraphs(joined_text, min_chars=1)
            segments = [
                {
                    "id": f"p{idx}",
                    "text": para,
                    "page": None,
                    "type": "paragraph",
                    "sentences": [para],
                    "metadata": {
                        "numbers": [],
                        "entities": [],
                        "markers": [],
                    },
                }
                for idx, para in enumerate(paras)
            ]

        return _serialise_blocks_to_chunks(segments)


# --------------------------------------------------------------------------- #
# Public entry point
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

    Modes
    -----
    - **LLM mode** (``llm`` is provided):
        Uses :class:`ParserAgent` to perform extraction + LLM refinement
        into fully structured segments.

    - **Stub mode** (``llm`` is ``None``):
        Falls back to the legacy paragraph splitter but emits segments
        that still conform to the richer schema (with default metadata).

    Parameters
    ----------
    input_data:
        Raw text or a local file path.
    bb:
        Blackboard instance where the result will be stored.
    key:
        Blackboard key to store the segments under (default:
        ``"parsed_chunks"``).
    min_chars:
        Minimum non-space characters required for a paragraph in stub
        mode. Ignored in LLM mode.
    make_trace:
        Whether to append a trace snapshot after writing to the
        blackboard.
    llm:
        Optional LLM client. If provided, enables full semantic parsing
        and refinement; otherwise, stub mode is used.

    Returns
    -------
    list[dict[str, Any]]
        Normalised segment dictionaries written to the blackboard.
    """
    # LLM-backed mode
    if llm is not None:
        agent = ParserAgent(llm)
        chunks = agent.semantic_parse(input_data)
        bb.put(key, chunks)
        if make_trace:
            bb.trace(f"parser_agent (LLM): {len(chunks)} segments -> {key}")
        return chunks

    # Stub mode: text-only paragraph splitting with rich schema defaults.
    if not isinstance(input_data, str):
        text = str(input_data)
    else:
        text = input_data

    paras = _split_paragraphs(text, min_chars=min_chars)
    segments: list[dict[str, Any]] = []
    for idx, para in enumerate(paras):
        para_text = para.strip()
        segments.append(
            {
                "id": f"p{idx}",
                "text": para_text,
                "page": None,
                "type": "paragraph",
                "sentences": _fallback_sentences(para_text),
                "metadata": {
                    "numbers": [],
                    "entities": [],
                    "markers": [],
                },
            }
        )

    chunks = _serialise_blocks_to_chunks(segments)
    bb.put(key, chunks)
    if make_trace:
        bb.trace(f"parser_agent (stub): {len(chunks)} segments -> {key}")
    return chunks


__all__ = ["ParserAgent", "parser_agent"]
