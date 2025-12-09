"""
Block Contract (Step 6.1.1)

This Pydantic model defines the structured unit produced by the new Parser
Agent. Each Block corresponds to a paragraph, heading, table, figure, or
image extracted from the source document.

Downstream agents (explainer, jargon, history, citizen) will consume lists
of Blocks rather than raw strings.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Block(BaseModel):
    """A structured block extracted from a document."""

    id: str = Field(..., description="Unique identifier (e.g. 'b1', 'fig2').")
    type: str = Field(
        ...,
        description="Block type.",
        pattern="^(paragraph|heading|table|figure|image)$",
    )
    page: int = Field(..., ge=1, description="1-indexed page number.")

    text: str | None = Field(
        None,
        description="Text content (or LLM-generated summary for non-text blocks).",
    )
    caption: str | None = Field(None, description="Caption extracted from PDF or provided by LLM.")
    key_points: list[str] | None = Field(default=None, description="Key findings extracted by LLM.")

    image_path: str | None = Field(
        default=None, description="Filesystem path for figure/image artifacts."
    )

    table_cells: list[list[str | None]] | None = Field(
        default=None, description="Optional 2D table cell values."
    )

    bbox: list[float] | None = Field(
        default=None, description="Bounding box [x1,y1,x2,y2] in PDF coordinate space."
    )

    provenance: list[str] | None = Field(
        default=None, description="Source references for downstream cards."
    )
