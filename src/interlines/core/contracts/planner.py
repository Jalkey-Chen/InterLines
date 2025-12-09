"""
PlannerPlanSpec — the structured, model-driven representation of a
pipeline plan for the public-translation workflow.

This dataclass formalizes the planner's output before it is converted
into a DAG. It is intentionally minimal in Commit 1, but will grow in
later milestones (e.g., LLM-backed planning, re-planning, retry rules).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PlannerPlanSpec(BaseModel):
    """
    A high-level, ordered plan for executing the public-translation pipeline.

    Parameters
    ----------
    steps:
        Ordered list of logical step names such as:
        ["parse", "explain", "narrate", "review", "brief"].

        These names are *logical* step identifiers, not agent function
        names. The pipeline will translate them into agent calls in later
        commits once DAG-driven execution is introduced.

    enable_history:
        Whether the history/timeline branch should be included.

    notes:
        Optional human-readable note (filled later by LLM planner).

    Notes
    -----
    - In Commit 1 (Step 5.1), this object simply mirrors the existing
      rule-based planner's linear order. It will become the primary
      "contract" for the planner in M5 Step 5.2+.
    """

    steps: list[str] = Field(
        default_factory=list,
        description="Ordered logical steps for the pipeline.",
    )
    enable_history: bool = Field(
        default=False,
        description="Whether the history branch is enabled.",
    )
    notes: str | None = Field(
        default=None,
        description="Optional natural-language annotation for the plan.",
    )


__all__ = ["PlannerPlanSpec"]
