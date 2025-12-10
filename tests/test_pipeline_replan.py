# tests/test_pipeline_replan.py
"""
Integration tests for the Single-Round Replan logic (Step 5.3).

This module verifies that the pipeline correctly handles the feedback loop:
1.  Pipeline runs initial plan.
2.  Editor produces a low-quality report.
3.  Planner inspects the report and triggers a replan.
4.  Pipeline constructs a transient DAG and executes refinement steps.
5.  Artifacts are updated and the Editor runs a second time.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.planner import PlannerPlanSpec
from interlines.core.contracts.review import ReviewCriteria, ReviewReport
from interlines.core.result import ok
from interlines.pipelines.public_translation import run_pipeline


def test_replan_triggered_on_low_readability() -> None:
    """
    Scenario: Editor reports low readability (0.5), Planner triggers refinement.

    Expected Behavior:
    - Initial steps (parse, translate, narrate, review) run once.
    - Replan steps (explainer_refine, citizen_refine, editor) run once.
    - Total calls: Explainer=2, Citizen=2, Editor=2.
    - Blackboard contains the 'planner_plan_spec.replan' artifact.
    """
    input_data = "Complex text needing refinement."

    # -----------------------------------------------------------------------
    # 1. Define Contracts (Plans & Reports)
    # -----------------------------------------------------------------------

    # A) Initial Plan (Standard)
    initial_steps = ["parse", "translate", "narrate", "review", "brief"]
    initial_plan = PlannerPlanSpec(
        strategy="test_replan_strategy",
        steps=initial_steps,
        enable_history=False,
        should_replan=False,
    )

    # B) The Low-Quality Report returned by the Editor (Trigger)
    low_quality_report = ReviewReport(
        kind="review_report.v1",
        version="1.0.0",
        confidence=1.0,
        overall=0.5,
        criteria=ReviewCriteria(
            kind="review_criteria.v1",
            version="1.0.0",
            confidence=1.0,
            accuracy=0.9,
            clarity=0.4,  # <--- Low clarity triggers replan
            completeness=0.8,
            safety=1.0,
        ),
        comments=["Text is too dense."],
        actions=["Simplify sentence structure."],
    )

    # C) The Replan Spec returned by the Planner (Decision)
    refine_steps = ["explainer_refine", "citizen_refine", "editor"]
    replan_spec = PlannerPlanSpec(
        strategy="test_replan_strategy",
        steps=initial_steps,  # Keeps history of initial steps
        should_replan=True,  # <--- DECISION: YES
        replan_steps=refine_steps,
        replan_reason="Clarity is too low (0.4).",
    )

    # -----------------------------------------------------------------------
    # 2. Setup Mocks & Side Effects
    # -----------------------------------------------------------------------

    # Track how many times each agent is called
    call_counts = {
        "explainer": 0,
        "citizen": 0,
        "editor": 0,
    }

    def mock_explainer(bb: Blackboard) -> Any:
        call_counts["explainer"] += 1
        return ok([])  # Return empty list of cards

    def mock_citizen(bb: Blackboard) -> Any:
        call_counts["citizen"] += 1
        return ok([])  # Return empty list of notes

    def mock_editor(bb: Blackboard) -> Any:
        call_counts["editor"] += 1
        # Fixed: Explicitly write the report to Blackboard so Pipeline sees it!
        # The real agent does this; our mock must simulate it.
        bb.put("review_report", low_quality_report.model_dump())
        return ok(low_quality_report)

    # FIX: mock_brief should write to BB like the real agent (Side Effect)
    # The real BriefBuilder writes the file path to 'public_brief_md_path'.
    def mock_brief(bb: Blackboard, **kwargs: Any) -> Any:
        bb.put("public_brief_md_path", "path/to/stub.md")
        return ok("path/to/stub.md")

    # Stub other agents to keep pipeline happy
    mock_parser = MagicMock(return_value=[{"id": "p1", "text": "stub"}])
    mock_jargon = MagicMock(return_value=ok([]))
    mock_history = MagicMock(return_value=ok([]))

    # -----------------------------------------------------------------------
    # 3. Execution with Patches
    # -----------------------------------------------------------------------

    # We patch the agents where they are USED (in public_translation.py)
    with (
        patch("interlines.pipelines.public_translation.PlannerAgent") as MockPlannerCls,
        patch("interlines.pipelines.public_translation.parser_agent", mock_parser),
        patch(
            "interlines.pipelines.public_translation.run_explainer",
            side_effect=mock_explainer,
        ),
        patch(
            "interlines.pipelines.public_translation.run_citizen",
            side_effect=mock_citizen,
        ),
        patch(
            "interlines.pipelines.public_translation.run_editor",
            side_effect=mock_editor,
        ),
        patch("interlines.pipelines.public_translation.run_jargon", mock_jargon),
        patch("interlines.pipelines.public_translation.run_history", mock_history),
        # Patch the BriefBuilder with our side-effect mock
        patch("interlines.pipelines.public_translation.run_brief_builder", side_effect=mock_brief),
    ):
        # Configure the Planner Mock
        planner_instance = MockPlannerCls.return_value
        planner_instance.plan.return_value = initial_plan  # Phase 1
        planner_instance.replan.return_value = replan_spec  # Phase 2

        # Run Pipeline
        result = run_pipeline(input_data=input_data, use_llm_planner=True)
        bb = result["blackboard"]

    # -----------------------------------------------------------------------
    # 4. Assertions
    # -----------------------------------------------------------------------

    # A) Verify call counts match the two-phase execution
    # Explainer: 1 (translate) + 1 (explainer_refine) = 2
    assert (
        call_counts["explainer"] == 2
    ), f"Explainer should run twice. Got {call_counts['explainer']}"

    # Citizen: 1 (narrate) + 1 (citizen_refine) = 2
    assert call_counts["citizen"] == 2, f"Citizen should run twice. Got {call_counts['citizen']}"

    # Editor: 1 (review) + 1 (editor replan) = 2
    assert call_counts["editor"] == 2, f"Editor should run twice. Got {call_counts['editor']}"

    # B) Verify Blackboard State
    # Ensure the replan decision was persisted
    stored_replan = bb.get("planner_plan_spec.replan")
    assert stored_replan is not None
    assert stored_replan["should_replan"] is True
    assert stored_replan["replan_reason"] == "Clarity is too low (0.4)."
    assert stored_replan["replan_steps"] == refine_steps

    # C) Verify Trace Log
    # We should see the trace emitted when the planner triggers the replan
    snaps = bb.traces()
    notes = [s.note for s in snaps if s.note]

    # Look for the specific trace message defined in public_translation.py
    replan_traces = [note for note in notes if "planner: triggering replan" in note]
    assert replan_traces, "Trace log should record the replan trigger event."
    assert str(refine_steps) in replan_traces[0]
