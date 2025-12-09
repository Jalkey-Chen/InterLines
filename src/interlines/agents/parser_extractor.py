"""
Hybrid document extractor for the Parser Agent.

This module provides *format-specific* extractors that convert raw inputs
(text blobs, PDF files, Word documents, and images) into coarse-grained
:class:`Block` objects. These “coarse blocks” act as the foundation for the
LLM-backed semantic parsing stage introduced.

Rationale
---------
The parser is designed as a two-stage system:

1. **Extraction stage (this module)**
   - Handles I/O.
   - Normalizes filesystem paths.
   - Converts heterogeneous formats into unified coarse Blocks.
   - Avoids fine-grained segmentation or semantic labeling.

2. **Semantic parsing stage**
   - Powered by an LLM with (optional) vision support.
   - Splits long text into meaningful paragraphs.
   - Recognizes headings, bullets, lists.
   - Interprets tables and figures.
   - Extracts captions and key points.

This separation ensures that:

- File-format complexity is isolated from LLM reasoning.
- Downstream agents (explainer, history, citizen) work with consistent,
  structured units.
- The pipeline remains deterministic and testable before LLMs enter.

Extraction Design
-----------------
TXT
    Loaded as a *single* paragraph block bound to ``page=1``.
PDF
    Each page is emitted as one coarse paragraph block using ``pdfplumber``.
DOCX
    All non-empty paragraphs are concatenated and returned as a single block.
IMAGE
    Represented as a figure block with ``image_path``, leaving interpretation
    to the LLM.

Implementation Notes
--------------------
- This version uses *top-level imports* for format libraries
  (``pdfplumber``, ``python-docx``, ``Pillow``), because the InterLines
  runtime environment guarantees they are installed.
- For editor/IDE compatibility, a type stub for python-docx is placed in
  ``src/interlines/_stubs/docx.pyi`` so that Pylance recognizes
  :class:`Document`.
- All extractors are pure functions: they do not mutate blackboards, write
  artifacts, or trigger LLM calls. They simply return ``List[Block]`` for
  the semantic parser to consume.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pdfplumber
from docx import Document
from PIL import Image  # noqa: F401  # Imported to validate the environment.

from interlines.core.contracts.block import Block

# ===========================================================================
# Helpers
# ===========================================================================


def _normalize_path(path: str | Path) -> Path:
    """Normalize a filesystem path and ensure it exists.

    Parameters
    ----------
    path:
        A filesystem path pointing to a text, PDF, DOCX, or image file.

    Returns
    -------
    Path
        A normalized, absolute :class:`Path` instance.

    Raises
    ------
    FileNotFoundError
        If the provided path does not exist.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")
    return p


def _make_block_id(index: int, prefix: str = "b") -> str:
    """Generate a simple monotonic block identifier.

    Examples
    --------
    >>> _make_block_id(1)
    'b1'
    >>> _make_block_id(2, prefix="fig")
    'fig2'
    """
    return f"{prefix}{index}"


# ===========================================================================
# TXT extractor
# ===========================================================================


def extract_from_text(text: str, *, doc_id_prefix: str = "b") -> list[Block]:
    """Extract a single coarse text block from raw text.

    The entire input text is returned as one ``paragraph``-type block.
    Fine-grained splitting (headings, bullets, paragraph boundaries) is
    performed by the LLM semantic parser in Step 6.1.3.

    Parameters
    ----------
    text:
        Raw text content, typically loaded from a ``.txt`` file.
    doc_id_prefix:
        Prefix used when generating the block identifier (default ``"b"``).

    Returns
    -------
    list[Block]
        Zero or one block depending on whether the input is empty.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    block = Block(
        id=_make_block_id(1, prefix=doc_id_prefix),
        type="paragraph",
        page=1,
        text=cleaned,
        caption=None,
        key_points=None,
        image_path=None,
        table_cells=None,
        bbox=None,
        provenance=None,
    )
    return [block]


# ===========================================================================
# PDF extractor
# ===========================================================================


def extract_from_pdf(
    path: str | Path,
    *,
    doc_id_prefix: str = "b",
) -> list[Block]:
    """Extract coarse paragraph blocks from a PDF document.

    PDF extraction uses :mod:`pdfplumber` to obtain page text. This
    extractor:

    - Emits at most one block per page.
    - Ignores empty pages.
    - Does *not* attempt layout analysis or table extraction.

    Parameters
    ----------
    path:
        Path to the PDF file.
    doc_id_prefix:
        Prefix used for block identifiers.

    Returns
    -------
    list[Block]
        One block per non-empty PDF page.

    Notes
    -----
    Complex PDFs may require enhanced extraction strategies (tables,
    figures, multi-column layout) which can be layered onto this module
    in later steps.
    """
    pdf_path = _normalize_path(path)
    blocks: list[Block] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue

            block = Block(
                id=_make_block_id(page_index, prefix=doc_id_prefix),
                type="paragraph",
                page=page_index,
                text=text,
                caption=None,
                key_points=None,
                image_path=None,
                table_cells=None,
                bbox=None,
                provenance=None,
            )
            blocks.append(block)

    return blocks


# ===========================================================================
# DOCX extractor
# ===========================================================================


def extract_from_docx(
    path: str | Path,
    *,
    doc_id_prefix: str = "b",
) -> list[Block]:
    """Extract coarse text blocks from a Word (``.docx``) document.

    Strategy
    --------
    - All paragraphs are read from the :class:`Document` object.
    - Empty paragraphs are ignored.
    - Non-empty paragraphs are concatenated with double newlines.
    - The result is emitted as one ``paragraph`` block.

    Parameters
    ----------
    path:
        Path to the DOCX file.
    doc_id_prefix:
        Prefix for block identifiers.

    Returns
    -------
    list[Block]
        A list containing zero or one block.

    Notes
    -----
    Precise segmentation (headings, subheadings, lists, captions) should
    be delegated to the LLM semantic parser.
    """
    docx_path = _normalize_path(path)
    document = Document(str(docx_path))

    paragraphs: Iterable[str] = ((p.text or "").strip() for p in document.paragraphs)
    non_empty = [p for p in paragraphs if p]

    if not non_empty:
        return []

    joined = "\n\n".join(non_empty)
    block = Block(
        id=_make_block_id(1, prefix=doc_id_prefix),
        type="paragraph",
        page=1,
        text=joined,
        caption=None,
        key_points=None,
        image_path=None,
        table_cells=None,
        bbox=None,
        provenance=None,
    )
    return [block]


# ===========================================================================
# Image extractor
# ===========================================================================


def extract_from_image(
    path: str | Path,
    *,
    doc_id_prefix: str = "fig",
) -> list[Block]:
    """Wrap an image as a figure-type block.

    No decoding or OCR is performed. The block simply records the absolute
    path to the image. Any semantic interpretation (caption generation,
    table detection, diagram understanding) is deferred.

    Parameters
    ----------
    path:
        Path to the image file.
    doc_id_prefix:
        Prefix used when generating block identifiers (default ``"fig"``).

    Returns
    -------
    list[Block]
        A list containing exactly one ``figure`` block.

    Raises
    ------
    FileNotFoundError
        If the image path does not exist.
    """
    img_path = _normalize_path(path)

    block = Block(
        id=_make_block_id(1, prefix=doc_id_prefix),
        type="figure",
        page=1,
        text=None,
        caption=None,
        key_points=None,
        image_path=str(img_path),
        table_cells=None,
        bbox=None,
        provenance=None,
    )
    return [block]


__all__ = [
    "extract_from_text",
    "extract_from_pdf",
    "extract_from_docx",
    "extract_from_image",
]
