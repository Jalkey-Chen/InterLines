# tests/test_pipeline_llm_planner.py
"""
Integration tests for DAG-driven pipeline execution with LLM planner stubs.

These tests verify that `run_pipeline(use_llm_planner=True)` correctly:
1. Calls the PlannerAgent to get a custom plan.
2. Converts that plan into a DAG.
3. Executes agents in the EXACT order dictated by the plan (even if weird).
4. Persists the plan spec to the blackboard.

Updates
-------
- Updated `mock_parse` signature to accept `llm` argument (Semantic Parsing support).
- Renamed `input_text` to `input_data` for consistency.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.planner import PlannerPlanSpec
from interlines.core.result import ok
from interlines.pipelines.public_translation import run_pipeline


def _stub_ok_result(data: Any = None) -> Any:
    """Helper to return an Ok result with minimal dummy data."""
    if data is None:
        data = []
    return ok(data)


def test_pipeline_follows_custom_llm_plan() -> None:
    """
    Verify that the pipeline follows a custom, non-standard execution order
    provided by a stubbed PlannerAgent.

    Scenario:
    - The planner decides to skip 'review' and 'narrate'.
    - The planner decides to run 'timeline' BEFORE 'translate' (unusual, but
      valid for testing DAG adherence).
    - Custom Order: parse -> timeline -> translate -> brief
    """
    input_data = "Test input for custom planning."  # Renamed variable

    # 1. Define the custom plan (The "Contract")
    custom_steps = ["parse", "timeline", "translate", "brief"]
    custom_plan = PlannerPlanSpec(
        strategy="custom_test_strategy",
        steps=custom_steps,
        enable_history=True,
        notes="Testing custom routing logic.",
    )

    # 2. Prepare a execution log to track who gets called
    execution_log: list[str] = []

    # 3. Define side-effect functions for the agents
    #    These agents don't need to do real work, just log their presence
    #    and return valid empty artifacts to keep the pipeline from crashing.

    # Fixed: Updated signature to accept `llm` kwarg passed by the pipeline
    # Also renamed first arg to `input_data` to match new signature types
    def mock_parse(input_data: Any, bb: Blackboard, **kwargs: Any) -> Any:
        execution_log.append("parse")
        return [{"id": "p1", "text": "stub"}]

    def mock_history(bb: Blackboard) -> Any:
        execution_log.append("timeline")
        return ok([])  # Return empty Result[list[TimelineEvent]]

    def mock_explainer(bb: Blackboard) -> Any:
        execution_log.append("translate")
        return ok([])  # Return empty Result[list[ExplanationCard]]

    def mock_brief(bb: Blackboard) -> Any:
        execution_log.append("brief")
        # brief_builder returns Result[Path, str]
        return ok("artifacts/reports/stub.md")

    # 4. Patch everything using a context manager
    #    We patch where the objects are USED (in the pipeline module).
    with (
        patch("interlines.pipelines.public_translation.PlannerAgent") as MockPlannerCls,
        patch(
            "interlines.pipelines.public_translation.parser_agent",
            side_effect=mock_parse,
        ),
        patch(
            "interlines.pipelines.public_translation.run_history",
            side_effect=mock_history,
        ),
        patch(
            "interlines.pipelines.public_translation.run_explainer",
            side_effect=mock_explainer,
        ),
        patch(
            "interlines.pipelines.public_translation.run_brief_builder",
            side_effect=mock_brief,
        ),
    ):
        # Setup the Planner Stub
        mock_planner_instance = MockPlannerCls.return_value
        # When planner.plan(...) is called, return our custom_plan
        mock_planner_instance.plan.return_value = custom_plan

        # 5. Run the pipeline
        result = run_pipeline(
            input_data,
            enable_history=True,  # This is a hint, but our stub planner forces True
            use_llm_planner=True,  # CRITICAL: This enables the logic we are testing
        )

        bb = result["blackboard"]

        # 6. Assertions

        # A) Assert execution order matches the plan exactly
        assert execution_log == custom_steps, (
            f"Pipeline did not follow the plan! Expected {custom_steps}, " f"got {execution_log}"
        )

        # B) Assert the plan spec was stored on the blackboard
        stored_plan_raw = bb.get("planner_plan_spec.initial")
        assert stored_plan_raw is not None

        # Verify content matches (comparing dict representations)
        assert stored_plan_raw == custom_plan.model_dump()
        assert stored_plan_raw["strategy"] == "custom_test_strategy"
        assert stored_plan_raw["steps"] == custom_steps

        # C) Verify the strategy label in the DAG payload on blackboard
        dag_payload = bb.get("planner_dag")
        assert dag_payload is not None
        assert dag_payload["strategy"] == "custom_test_strategy"
        assert dag_payload["topo"] == custom_steps
