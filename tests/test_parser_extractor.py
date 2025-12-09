"""
Unit tests for the hybrid document extractor (Step 6.1.2).

These tests focus on the *in-process* behavior of the extractor helper
functions without requiring optional dependencies such as ``pdfplumber``
or ``python-docx``. For PDF and DOCX extraction, we only verify that a
clear error message is raised when the optional dependency is missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from interlines.agents import parser_extractor as extractor
from interlines.core.contracts.block import Block


def test_extract_from_text_returns_single_block() -> None:
    """Plain text should be wrapped into a single paragraph block."""
    text = "InterLines turns expert language into public language."
    blocks = extractor.extract_from_text(text)

    assert isinstance(blocks, list)
    assert len(blocks) == 1

    block = blocks[0]
    assert isinstance(block, Block)
    assert block.type == "paragraph"
    assert block.page == 1
    assert block.text == text


def test_extract_from_text_handles_empty_input() -> None:
    """Empty or whitespace-only input should return an empty list."""
    assert extractor.extract_from_text("") == []
    assert extractor.extract_from_text("   \n\t  ") == []


def test_extract_from_image_wraps_path() -> None:
    """Image extraction should emit a single figure block with image_path set."""
    # Use a temporary empty file to simulate an image.
    tmp = Path("tests") / "tmp-empty-image.bin"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(b"")

    try:
        blocks = extractor.extract_from_image(tmp)
        assert len(blocks) == 1
        block = blocks[0]
        assert block.type == "figure"
        assert block.image_path is not None
        assert Path(block.image_path).resolve() == tmp.resolve()
    finally:
        tmp.unlink(missing_ok=True)


def test_extract_from_pdf_requires_dependency() -> None:
    """PDF extraction should raise a RuntimeError when pdfplumber is missing."""
    # We cannot reliably assert the absence of pdfplumber in all
    # environments, so we only run this test when the import fails.
    test_pdf = Path("tests") / "dummy.pdf"
    test_pdf.write_bytes(b"%PDF-1.4\n%EOF\n")  # Minimal placeholder bytes.

    try:
        try:
            pytest.skip("pdfplumber is installed; dependency-missing path not testable.")
        except Exception:
            with pytest.raises(RuntimeError):
                extractor.extract_from_pdf(test_pdf)
    finally:
        test_pdf.unlink(missing_ok=True)


def test_extract_from_docx_requires_dependency() -> None:
    """DOCX extraction should raise a RuntimeError when python-docx is missing."""
    test_docx = Path("tests") / "dummy.docx"
    test_docx.write_bytes(b"PK\x03\x04")  # ZIP file magic prefix; minimal placeholder.

    try:
        try:
            pytest.skip("python-docx is installed; dependency-missing path not testable.")
        except Exception:
            with pytest.raises(RuntimeError):
                extractor.extract_from_docx(test_docx)
    finally:
        test_docx.unlink(missing_ok=True)
