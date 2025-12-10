"""
Tests for default planner ordering and DAG generation.

This verifies three invariants:

1. The step order returned by build_default_plan(...) matches the
   legacy hand-written expected path for enable_history = {True, False}.

2. DAG.from_plan_spec(plan_spec).topological_order() is identical to
   plan_spec.steps.

3. No cycles or missing nodes exist in the generated DAG.
"""

from interlines.core.planner.dag import DAG
from interlines.core.planner.strategy import build_plan

# Legacy expected execution paths
EXPECTED_NO_HISTORY = ["parse", "translate", "narrate", "review", "brief"]
EXPECTED_WITH_HISTORY = ["parse", "translate", "timeline", "narrate", "review", "brief"]


def test_default_plan_matches_expected_path_no_history() -> None:
    plan_spec, _ = build_plan(enable_history=False)
    assert plan_spec.steps == EXPECTED_NO_HISTORY


def test_default_plan_matches_expected_path_with_history() -> None:
    plan_spec, _ = build_plan(enable_history=True)
    assert plan_spec.steps == EXPECTED_WITH_HISTORY


def test_dag_matches_plan_spec_no_history() -> None:
    plan_spec, _ = build_plan(enable_history=False)
    dag = DAG.from_plan_spec(plan_spec)
    assert dag.topological_order() == plan_spec.steps


def test_dag_matches_plan_spec_with_history() -> None:
    plan_spec, _ = build_plan(enable_history=True)
    dag = DAG.from_plan_spec(plan_spec)
    assert dag.topological_order() == plan_spec.steps


def test_dag_no_cycles() -> None:
    plan_spec, _ = build_plan(enable_history=True)
    dag = DAG.from_plan_spec(plan_spec)

    order = dag.topological_order()
    assert len(order) == len(set(order))  # no duplicates
    assert set(order) == set(plan_spec.steps)
