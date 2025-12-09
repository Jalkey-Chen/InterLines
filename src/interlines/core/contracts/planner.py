"""
Planner contracts: Specifications and Reports for pipeline orchestration.

This module defines the data structures used by the Planner Agent to control
and report on the pipeline execution flow.

- :class:`PlannerPlanSpec`: The "forward-looking" instruction set (what to do).
- :class:`PlanReport`: The "backward-looking" summary (what was decided).
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Set of logical step names allowed during the replan/refine phase.
#: The planner must only choose steps from this allowlist when populating
#: ``replan_steps``.
ALLOWED_REFINE_STEPS: frozenset[str] = frozenset(
    {
        "explainer_refine",
        "citizen_refine",
        "jargon_refine",
        "history_refine",
        "editor",  # The editor usually runs again to verify the fix.
    }
)


# --------------------------------------------------------------------------- #
# Data Model: Instruction Spec (Forward-looking)
# --------------------------------------------------------------------------- #


class PlannerPlanSpec(BaseModel):
    """
    A high-level, ordered plan for executing the public-translation pipeline.

    This model supports a two-stage execution:
    1. **Initial execution**: Defined by ``steps``.
    2. **Optional replan**: Defined by ``should_replan`` and ``replan_steps``.

    Parameters
    ----------
    strategy:
        The planner's chosen execution strategy (e.g., "llm_planner.v1").
    steps:
        Ordered list of logical step names for the *initial* pass.
    enable_history:
        Whether the history branch is enabled in the initial pass.
    notes:
        Optional natural-language rationale for the plan.
    should_replan:
        Flag indicating if a refinement pass is triggered based on review.
    replan_steps:
        Ordered list of steps to execute during the refinement pass.
    replan_reason:
        Rationale for the replan decision.
    """

    # --- Initial Plan Configuration ---
    strategy: str = Field(description="Planner's chosen strategy label (semantic).")

    steps: list[str] = Field(
        default_factory=list,
        description="Ordered logical steps for the pipeline's first pass.",
    )

    enable_history: bool = Field(
        default=False,
        description="Whether the history branch is enabled.",
    )

    notes: str | None = Field(
        default=None,
        description="Optional natural-language annotation for the plan.",
    )

    # --- Replan / Refine Configuration (Step 5.3) ---
    should_replan: bool = Field(
        default=False,
        description="Whether a refinement pass is triggered based on review.",
    )

    replan_steps: list[str] | None = Field(
        default=None,
        description=(
            "Ordered steps for the refinement pass. " "Ignored if should_replan is False."
        ),
    )

    replan_reason: str | None = Field(
        default=None,
        description="Rationale for the replan decision (or lack thereof).",
    )

    # Expose the constant on the class for convenient access in validators
    ALLOWED_STEPS: ClassVar[frozenset[str]] = ALLOWED_REFINE_STEPS


# --------------------------------------------------------------------------- #
# Data Model: Executive Summary (Backward-looking)
# --------------------------------------------------------------------------- #


class PlanReport(BaseModel):
    """
    Structured summary of the Planner's decisions for a pipeline run.

    Unlike :class:`PlannerPlanSpec`, which drives execution, this model is
    intended for observability, debugging, and final reporting. It aggregates
    decisions from both the initial phase and any subsequent refinement loops.

    Fields
    ------
    strategy:
        The strategy label used (e.g., "rule_planner.v1", "llm_planner.v1").
    enable_history:
        Final status of the history branch.
    initial_steps:
        The steps executed in Phase 1.
    replan_steps:
        The steps executed in Phase 2 (if any).
    refine_used:
        Boolean flag indicating if a refinement loop actually occurred.
    replan_reason:
        The rationale provided for triggering (or skipping) refinement.
    notes:
        General planning notes or rationale.
    """

    strategy: str
    enable_history: bool
    initial_steps: list[str]
    replan_steps: list[str] | None = None
    refine_used: bool = False
    replan_reason: str | None = None
    notes: str | None = None


__all__ = ["PlannerPlanSpec", "PlanReport", "ALLOWED_REFINE_STEPS"]
