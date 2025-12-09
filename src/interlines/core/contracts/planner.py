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
    strategy:
        The planner's chosen execution strategy. Examples:
        - "no_history"
        - "with_history"
        - "rule_planner.v1"
        - "llm_planner.v1"

        This is the semantic label representing how the planner decided
        to configure the workflow. All downstream components (DAG,
        trace, blackboard) inherit this value for explainability.

    steps:
        Ordered list of logical step names such as:
        ["parse", "explain", "narrate", "review", "brief"].
        These are logical pipeline phase identifiers.

    enable_history:
        Whether the history/timeline branch should be included.

    notes:
        Optional human-readable commentary (later filled by LLM planner).
    """

    # NEW FIELD — absolutely required for DAG + tests + future LLM planners
    strategy: str = Field(description="Planner's chosen strategy label (semantic).")

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
