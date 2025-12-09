"""
Hybrid document extractor for the Parser Agent.

This module provides *format-specific* extractors that convert raw
inputs (inline text, file paths) into coarse-grained :class:`Block`
objects. These coarse Blocks are later refined by the LLM-backed
semantic parser.

Design
------
- TXT: treated as a single page-1 block.
- PDF: best-effort extraction; each page becomes one paragraph block.
  Importantly, the optional dependency ``pdfplumber`` is loaded lazily.
  The module must remain importable even if ``pdfplumber`` is missing.
- DOCX: paragraph concatenation; ``python-docx`` also loaded lazily.
- Image: wrapped as a figure-type block; semantic interpretation (OCR,
  captioning, metadata recognition) is deferred to the LLM parser.

None of the functions in this module mutate global state—they only
return structured Block objects. Higher-level parser logic orchestrates
the extraction + semantic parsing pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from interlines.core.contracts.block import Block

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _normalize_path(path: str | Path) -> Path:
    """Normalize a user path into an absolute :class:`Path`.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")
    return p


def _make_block_id(index: int, prefix: str = "b") -> str:
    """Generate a stable block identifier.

    Examples
    --------
    >>> _make_block_id(1)
    'b1'
    >>> _make_block_id(3, prefix="fig")
    'fig3'
    """
    return f"{prefix}{index}"


# ---------------------------------------------------------------------------
# TXT extractor
# ---------------------------------------------------------------------------


def extract_from_text(text: str, *, doc_id_prefix: str = "b") -> list[Block]:
    """Extract a single coarse block from plain text.

    The entire text blob becomes a single page-1 paragraph block.
    Semantic segmentation is delegated to the LLM parser.
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


# ---------------------------------------------------------------------------
# PDF extractor — *lazy import*, avoids CI breakage
# ---------------------------------------------------------------------------


def extract_from_pdf(
    path: str | Path,
    *,
    doc_id_prefix: str = "b",
) -> list[Block]:
    """Extract coarse text blocks from a PDF file.

    Notes
    -----
    - ``pdfplumber`` is imported lazily inside this function.
    - If the optional dependency is not installed, a clear RuntimeError
      is raised, but importing this module remains safe.
    """
    pdf_path = _normalize_path(path)

    try:
        import pdfplumber  # lazy import
    except Exception as exc:  # pragma: no cover (CI stability)
        raise RuntimeError(
            "PDF extraction requires the optional 'pdfplumber' dependency. "
            "Install it with: pip install pdfplumber"
        ) from exc

    blocks: list[Block] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            cleaned = text.strip()
            if not cleaned:
                continue

            block_id = _make_block_id(page_idx, prefix=doc_id_prefix)
            blocks.append(
                Block(
                    id=block_id,
                    type="paragraph",
                    page=page_idx,
                    text=cleaned,
                    caption=None,
                    key_points=None,
                    image_path=None,
                    table_cells=None,
                    bbox=None,
                    provenance=None,
                )
            )

    return blocks


# ---------------------------------------------------------------------------
# DOCX extractor — *lazy import*, avoids hard dependency
# ---------------------------------------------------------------------------


def extract_from_docx(
    path: str | Path,
    *,
    doc_id_prefix: str = "b",
) -> list[Block]:
    """Extract coarse text from a DOCX file.

    All non-empty paragraphs are concatenated into a single page-1 block.
    The optional ``python-docx`` library is imported lazily.
    """
    docx_path = _normalize_path(path)

    try:
        import docx  # lazy import
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "DOCX extraction requires the optional 'python-docx' dependency. "
            "Install it with: pip install python-docx"
        ) from exc

    document = docx.Document(docx_path)
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


# ---------------------------------------------------------------------------
# Image extractor
# ---------------------------------------------------------------------------


def extract_from_image(
    path: str | Path,
    *,
    doc_id_prefix: str = "fig",
) -> list[Block]:
    """Wrap an image file as a figure-type block.

    No OCR or visual interpretation is attempted here. That work is left
    to the semantic parser. This function only records the file path.
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
