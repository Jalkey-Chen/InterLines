"""
Rule-based planning strategy for the public-translation pipeline.

Commit 1 (Step 5.1):
- Introduces PlannerPlanSpec as a structured representation
- Keeps the legacy DAG construction unchanged so tests continue to pass
"""

from __future__ import annotations

from interlines.core.contracts.planner import PlannerPlanSpec

from .dag import DAG


def expected_path(enable_history: bool) -> tuple[str, ...]:
    """
    Return the expected topological order for the rule-based planner.
    Tests assert this exact ordering.
    """
    if enable_history:
        return ("parse", "translate", "timeline", "narrate", "review", "brief")
    return ("parse", "translate", "narrate", "review", "brief")


def build_plan(enable_history: bool) -> tuple[PlannerPlanSpec, DAG]:
    """
    Construct the default rule-based PlannerPlanSpec *and* the legacy DAG.

    Returns
    -------
    (planner_plan_spec, dag)

    Notes
    -----
    - The pipeline still uses the DAG in Commit 1.
    - PlannerPlanSpec is introduced now to prepare for DAG-driven
      execution in Commit 2 and for LLM planning in Step 5.2+.
    """
    if enable_history:
        steps = ["parse", "translate", "timeline", "narrate", "review", "brief"]
    else:
        steps = ["parse", "translate", "narrate", "review", "brief"]

    plan_spec = PlannerPlanSpec(
        steps=steps,
        enable_history=enable_history,
        notes=None,
    )

    # legacy DAG-compatible structure (unchanged)
    dag = DAG(strategy="with_history" if enable_history else "no_history")

    dag.add("parse", "translate")
    if enable_history:
        dag.add("translate", "timeline")
        dag.add("timeline", "narrate")
    else:
        dag.add("translate", "narrate")

    dag.add("narrate", "review")
    dag.add("review", "brief")

    return plan_spec, dag


__all__ = ["build_plan", "expected_path"]
