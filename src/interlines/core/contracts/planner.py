# src/interlines/core/contracts/planner.py
"""
PlannerPlanSpec — the structured, model-driven representation of a
pipeline plan for the public-translation workflow.

This module formalizes the planner's output. It now includes support for
single-round replanning (Step 5.3), allowing the planner to inspect a
ReviewReport and trigger a targeted refinement pass.
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
# Data Model
# --------------------------------------------------------------------------- #


class PlannerPlanSpec(BaseModel):
    """
    A high-level, ordered plan for executing the public-translation pipeline.

    This model now supports a two-stage execution:
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
        Flag indicating if a second pass is required. Defaults to False.
    replan_steps:
        Ordered list of steps to execute during the refinement pass.
        Must be a subset of :data:`ALLOWED_REFINE_STEPS`.
    replan_reason:
        Rationale for triggering (or not triggering) a replan, usually based
        on specific issues found in the ReviewReport.
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
    # or downstream logic.
    ALLOWED_STEPS: ClassVar[frozenset[str]] = ALLOWED_REFINE_STEPS


__all__ = ["PlannerPlanSpec", "ALLOWED_REFINE_STEPS"]
