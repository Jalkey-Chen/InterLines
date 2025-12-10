"""
Integration tests for Planner Reporting and Observability (Step 5.4).

This module verifies that the pipeline produces a structured :class:`PlanReport`
and meaningful trace logs, allowing developers to inspect *why* the system
made certain decisions (e.g., strategy selection, replan triggers).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.planner import PlannerPlanSpec, PlanReport
from interlines.core.contracts.review import ReviewCriteria, ReviewReport
from interlines.core.result import Result, ok
from interlines.pipelines.public_translation import run_pipeline


def _stub_agents_and_run(
    input_data: str,
    *,
    use_llm_planner: bool,
    mock_planner_instance: MagicMock | None = None,
    mock_editor_return: Any = None,
) -> Blackboard:
    """Helper to run the pipeline with all agents stubbed out."""
    mock_parser = MagicMock(return_value=[{"id": "p1", "text": "stub"}])
    mock_explainer = MagicMock(return_value=ok([]))
    mock_citizen = MagicMock(return_value=ok([]))
    mock_jargon = MagicMock(return_value=ok([]))
    mock_history = MagicMock(return_value=ok([]))
    mock_brief = MagicMock(return_value=ok("stub.md"))

    editor_result: Result[ReviewReport, str]

    if mock_editor_return:
        editor_result = ok(mock_editor_return)
    else:
        passing_report = ReviewReport(
            kind="review_report.v1",
            version="1.0.0",
            confidence=1.0,
            overall=0.9,
            criteria=ReviewCriteria(
                kind="review_criteria.v1",
                version="1.0.0",
                confidence=1.0,
                accuracy=0.9,
                clarity=0.9,
                completeness=0.9,
                safety=1.0,
            ),
            comments=["Good job."],
            actions=[],
        )
        editor_result = ok(passing_report)

    # Note: run_editor must write to blackboard AND return the result
    def side_effect_editor(bb: Blackboard) -> Any:
        report = editor_result.unwrap()
        bb.put("review_report", report.model_dump())
        return editor_result

    # 3. Patch the pipeline module
    with (
        patch("interlines.pipelines.public_translation.PlannerAgent") as MockPlannerCls,
        patch("interlines.pipelines.public_translation.parser_agent", mock_parser),
        patch("interlines.pipelines.public_translation.run_explainer", mock_explainer),
        patch("interlines.pipelines.public_translation.run_citizen", mock_citizen),
        patch("interlines.pipelines.public_translation.run_jargon", mock_jargon),
        patch("interlines.pipelines.public_translation.run_history", mock_history),
        patch("interlines.pipelines.public_translation.run_brief_builder", mock_brief),
        patch(
            "interlines.pipelines.public_translation.run_editor",
            side_effect=side_effect_editor,
        ),
    ):
        if mock_planner_instance:
            MockPlannerCls.return_value = mock_planner_instance

        # 4. Execute
        res = run_pipeline(
            input_data=input_data,
            enable_history=False,
            use_llm_planner=use_llm_planner,
        )
        return res["blackboard"]


def test_report_generated_for_rule_based_plan() -> None:
    """
    Verify observability for the legacy Rule-Based path.

    Expectations:
    - The :class:`PlanReport` should exist on the blackboard.
    - ``strategy`` should match the rule-based label (e.g. "no_history").
    - ``refine_used`` should be False.
    - Trace logs should confirm the report was written.
    """
    bb = _stub_agents_and_run(
        input_data="Rule based test",
        use_llm_planner=False,
    )

    # 1. Verify Report Existence
    report_raw = bb.get("planner_report")
    assert report_raw is not None, "Pipeline must write 'planner_report' to blackboard."

    report = PlanReport(**report_raw)

    assert report.strategy == "no_history"
    assert report.enable_history is False
    assert report.refine_used is False
    assert report.replan_steps is None
    assert report.replan_reason is None

    # 3. Verify Trace
    traces = [t.note for t in bb.traces() if t.note]
    assert "planner: report written" in traces


def test_report_generated_for_llm_replan() -> None:
    """
    Verify observability for the LLM Replan path.

    Scenario:
    - Initial plan runs standard steps.
    - Editor returns a poor report.
    - Planner triggers a replan with specific steps and a reason.
    """
    # --- Setup Mocks ---
    initial_steps = ["parse", "translate", "review", "brief"]
    replan_steps = ["explainer_refine", "editor"]
    replan_reason = "Readability too low."

    # 1. Mock Planner Behavior
    mock_planner = MagicMock()

    # Phase 1: Initial Plan
    mock_planner.plan.return_value = PlannerPlanSpec(
        strategy="llm_planner.test",
        steps=initial_steps,
        enable_history=False,
        should_replan=False,
    )

    # Phase 2: Replan Decision
    mock_planner.replan.return_value = PlannerPlanSpec(
        strategy="llm_planner.test",
        steps=initial_steps,
        should_replan=True,
        replan_steps=replan_steps,
        replan_reason=replan_reason,
    )

    # 2. Mock Editor Behavior (Low Score to justify the replan)
    low_score_report = ReviewReport(
        kind="review_report.v1",
        version="1.0.0",
        confidence=1.0,
        overall=0.4,
        criteria=ReviewCriteria(
            kind="review_criteria.v1",
            version="1.0.0",
            confidence=1.0,
            accuracy=0.8,
            clarity=0.3,
            completeness=0.9,
            safety=1.0,
        ),
    )

    # --- Run Pipeline ---
    bb = _stub_agents_and_run(
        input_data="LLM replan test",
        use_llm_planner=True,
        mock_planner_instance=mock_planner,
        mock_editor_return=low_score_report,
    )

    # --- Assertions ---
    report_raw = bb.get("planner_report")
    assert report_raw is not None
    report = PlanReport(**report_raw)

    assert report.strategy == "llm_planner.test"
    assert report.initial_steps == initial_steps
    assert report.refine_used is True
    assert report.replan_steps == replan_steps
    assert report.replan_reason == replan_reason

    # 3. Verify Trace Provenance
    traces = [t.note for t in bb.traces() if t.note]
    assert any("planner: triggering replan" in t for t in traces)
    assert "planner: report written" in traces
