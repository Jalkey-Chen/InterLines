# tests/test_pipeline_report.py
"""
Integration tests for Planner Reporting and Observability (Step 5.4).

This module verifies that the pipeline produces a structured :class:`PlanReport`
and meaningful trace logs, allowing developers to inspect *why* the system
made certain decisions (e.g., strategy selection, replan triggers).

Scenarios Covered
-----------------
1. **Rule-Based Execution**:
   - Verify that the report correctly identifies the "rule-based" strategy.
   - Confirm that no replan steps are recorded.

2. **LLM-Based Execution with Replan**:
   - Simulate a low-quality draft triggering a refinement loop.
   - Verify that the report captures the `replan_reason`, `refine_used` flag,
     and the specific `replan_steps` taken.
   - Confirm that trace logs contain the "planner: report written" marker.

Test Architecture
-----------------
We use ``unittest.mock`` to stub out the actual Agents. This ensures:
- **Determinism**: We control the "decisions" (plan specs) and "quality"
  (review reports) explicitly.
- **Speed**: No actual LLM calls or network I/O.
- **Isolation**: We test the *reporting logic* in the pipeline, not the agents.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.planner import PlannerPlanSpec, PlanReport
from interlines.core.contracts.review import ReviewCriteria, ReviewReport

# Fixed (MyPy): Imported Result for type annotation
from interlines.core.result import Result, ok
from interlines.pipelines.public_translation import run_pipeline


def _stub_agents_and_run(
    input_data: str,  # Renamed from input_text
    *,
    use_llm_planner: bool,
    mock_planner_instance: MagicMock | None = None,
    mock_editor_return: Any = None,
) -> Blackboard:
    """
    Helper to run the pipeline with all agents stubbed out.

    Parameters
    ----------
    input_data:
        The raw input text for the pipeline.
    use_llm_planner:
        Passed through to ``run_pipeline``.
    mock_planner_instance:
        If provided, this Mock object is injected as the return value of
        ``PlannerAgent()``. Callers should configure its ``.plan()`` and
        ``.replan()`` methods beforehand.
    mock_editor_return:
        If provided, the ``run_editor`` stub will return this value (usually
        a :class:`ReviewReport`). If None, a default passing report is returned.

    Returns
    -------
    Blackboard
        The blackboard state after pipeline execution, ready for assertions.
    """
    # 1. Define default stubs for non-decision agents
    #    These agents just need to "work" so the pipeline keeps moving.
    mock_parser = MagicMock(return_value=[{"id": "p1", "text": "stub"}])
    mock_explainer = MagicMock(return_value=ok([]))
    mock_citizen = MagicMock(return_value=ok([]))
    mock_jargon = MagicMock(return_value=ok([]))
    mock_history = MagicMock(return_value=ok([]))
    # run_brief_builder must return (Path, str) or Result[Path, str] depending on contract
    # In the pipeline, it returns Result[Path, str].
    mock_brief = MagicMock(return_value=ok("stub.md"))

    # 2. Define the Editor stub (critical for replan logic)
    # Fixed (MyPy): Added explicit type annotation for the result variable
    editor_result: Result[ReviewReport, str]

    if mock_editor_return:
        editor_result = ok(mock_editor_return)
    else:
        # Default: High quality report (no replan needed)
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
    # Fixed: We must define a side_effect function to simulate the agent's write action
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
            input_data=input_data,  # Updated parameter name
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
    - ``refine_used`` should be False (rules don't support replan yet).
    - Trace logs should confirm the report was written.
    """
    bb = _stub_agents_and_run(
        input_data="Rule based test",
        use_llm_planner=False,
    )

    # 1. Verify Report Existence
    report_raw = bb.get("planner_report")
    assert report_raw is not None, "Pipeline must write 'planner_report' to blackboard."

    # Validate structure using the Pydantic model
    report = PlanReport(**report_raw)

    # 2. Verify Content
    # The default rule-based strategy for enable_history=False is "no_history"
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

    Expectations:
    - The :class:`PlanReport` must capture the *entire* story.
    - ``strategy`` should match the initial plan's strategy.
    - ``refine_used`` should be True.
    - ``replan_reason`` and ``replan_steps`` must match the Planner's decision.
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
        steps=initial_steps,  # History of what ran first
        should_replan=True,  # Trigger!
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
            clarity=0.3,  # <--- The problem
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

    # 1. Verify Report Integrity
    report_raw = bb.get("planner_report")
    assert report_raw is not None
    report = PlanReport(**report_raw)

    # 2. Verify Deep Fields
    assert report.strategy == "llm_planner.test"
    assert report.initial_steps == initial_steps

    # Crucial: Did it capture the replan details?
    assert report.refine_used is True
    assert report.replan_steps == replan_steps
    assert report.replan_reason == replan_reason

    # 3. Verify Trace Provenance
    # We expect to see both the "triggering replan" trace and the "report written" trace
    traces = [t.note for t in bb.traces() if t.note]
    assert any("planner: triggering replan" in t for t in traces)
    assert "planner: report written" in traces
