"""
PlannerAgent v1 — LLM-backed semantic routing and refinement for InterLines.

Milestone
---------
M5 | Planner Agent
Step 5.2 | LLM-backed Initial Planning
Step 5.3 | Single-round Replan based on Review Report

This module implements the central "brain" of the pipeline. It is responsible
for two distinct phases of execution:

1. **Initial Planning** (Step 5.2):
   - Analyzes a document preview and execution context.
   - Determines the optimal sequence of analysis steps (e.g., whether to
     include a historical timeline).
   - Sets quality thresholds (readability, factuality).

2. **Refinement / Re-planning** (Step 5.3):
   - Inspects the :class:`ReviewReport` produced by the Editor Agent.
   - Decides if the output meets quality standards.
   - If not, generates a targeted refinement plan (e.g., "explainer_refine",
     "citizen_refine") to fix specific issues before final delivery.

Design Notes
------------
- The planner defines the *contract* (JSON schema) for the LLM but relies
  on the core :class:`PlannerPlanSpec` for internal state.
- It is stateless between calls; context is passed in via arguments.
- It strictly adheres to :data:`ALLOWED_REFINE_STEPS` during the replan
  phase to prevent hallucinated step names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.planner import ALLOWED_REFINE_STEPS, PlannerPlanSpec
from interlines.core.contracts.review import ReviewReport
from interlines.llm.client import LLMClient

# --------------------------------------------------------------------------- #
# PlannerContext — lightweight metadata passed into the planner
# --------------------------------------------------------------------------- #


@dataclass
class PlannerContext:
    """
    Lightweight metadata describing the planning situation.

    This object is intentionally small and serialisation-friendly. It lets
    the planner reason about *how* to route a document without requiring
    access to the entire raw text.

    Parameters
    ----------
    task_type:
        High-level task category. For the current PKI / InterLines project
        this will almost always be ``"public_translation"``.
    document_kind:
        Optional descriptor of the document genre, such as ``"policy"``,
        ``"research"``, ``"news"``, etc.
    approx_char_count:
        Rough estimate of the input document length in characters.
    language:
        Two-letter language code for the *original* document (e.g. "en").
    enable_history_requested:
        Whether the *user* explicitly requested a historical / timeline
        perspective. The planner may override this, but treats it as a hint.
    """

    task_type: str = "public_translation"
    document_kind: str | None = None
    approx_char_count: int = 0
    language: str = "en"
    enable_history_requested: bool = False


# --------------------------------------------------------------------------- #
# LLM Output Schemas
# --------------------------------------------------------------------------- #


class PlannerLLMOutput(BaseModel):
    """
    Validated JSON output schema for the initial planning phase.

    Fields
    ------
    steps:
        Ordered list of logical pipeline step names for the first pass.
    enable_history:
        Boolean indicating whether the history / timeline branch is active.
    readability_threshold:
        Optional target score for readability (0.0 - 1.0).
    factuality_threshold:
        Optional target score for factuality (0.0 - 1.0).
    max_refine_rounds:
        Optional upper bound on how many refinement passes to attempt.
    notes:
        Optional natural-language justification of the plan.
    """

    steps: list[str] = Field(
        default_factory=list,
        description="Ordered list of logical pipeline steps.",
    )
    enable_history: bool = Field(
        default=False,
        description="Whether the history / timeline branch should be used.",
    )
    readability_threshold: float | None = Field(
        default=None,
        description="Optional readability target in [0, 1].",
    )
    factuality_threshold: float | None = Field(
        default=None,
        description="Optional factuality target in [0, 1].",
    )
    max_refine_rounds: int | None = Field(
        default=None,
        description="Optional upper bound on refinement iterations.",
    )
    notes: str | None = Field(
        default=None,
        description="Optional natural-language planner rationale.",
    )


class ReplanLLMOutput(BaseModel):
    """
    Validated JSON output schema for the replanning/refinement phase.

    This schema forces the LLM to make a binary decision (`should_replan`)
    and provide specific steps if affirmative.

    Fields
    ------
    should_replan:
        True if the review report indicates issues that require fixing.
    replan_steps:
        List of refinement steps to execute. The PlannerAgent will validate
        that these are subsets of `ALLOWED_REFINE_STEPS`.
    replan_reason:
        Explanation of why replanning is (or isn't) necessary, often
        referencing specific scores or comments from the review.
    """

    should_replan: bool = Field(
        description="Whether to trigger a refinement loop based on the review.",
    )
    replan_steps: list[str] = Field(
        default_factory=list,
        description=(
            "List of refinement steps to execute. " "Must be a subset of allowed refine steps."
        ),
    )
    replan_reason: str = Field(
        default="",
        description="Reasoning for the decision, referencing specific review issues.",
    )


# --------------------------------------------------------------------------- #
# PlannerAgent Implementation
# --------------------------------------------------------------------------- #


class PlannerAgent:
    """
    LLM-backed planner for semantic routing and refinement.

    This agent is the decision-making engine of the pipeline. It operates in
    two modes:

    1. **Initial Plan**: Called at the start of ``run_pipeline``.
       Constructs the primary DAG based on document characteristics.

    2. **Replan**: Called after the Editor Agent runs.
       Inspects quality metrics and decides if a second pass is needed.
    """

    def __init__(self, llm: LLMClient, *, model_alias: str = "planner") -> None:
        """
        Initialize the PlannerAgent.

        Parameters
        ----------
        llm:
            Shared :class:`LLMClient` instance for provider API calls.
        model_alias:
            Logical model alias to use (default: "planner").
            Resolved via :mod:`interlines.llm.models`.
        """
        self.llm = llm
        self.model_alias = model_alias

    # ------------------------------------------------------------------ #
    # Initial Planning
    # ------------------------------------------------------------------ #

    def _initial_system_prompt(self) -> str:
        """
        Construct the System Prompt for the Initial Planning phase.

        Updates (Fixing Over-optimization):
        - Added strict guidelines to enforce 'translate' for technical docs.
        - Clarified the dependencies between steps (e.g. brief requires translate).
        """
        return (
            "You are the PlannerAgent in the InterLines (PKI) system. "
            "Your job is to design an efficient sequence of analysis steps "
            "that transforms an expert-facing document into a clear, "
            "public-friendly brief.\n\n"
            "**Allowed Logical Steps:**\n"
            "  - parse: Extract text (Mandatory start).\n"
            "  - translate: Core explanation engine (produces ExplanationCards).\n"
            "  - citizen: Audience relevance analysis.\n"
            "  - jargon: Terminology extraction.\n"
            "  - timeline: Historical event extraction.\n"
            "  - narrate: (Legacy) groups citizen/jargon.\n"
            "  - review: Editor quality check.\n"
            "  - brief: Markdown assembly (Mandatory end).\n\n"
            "**Critical Guidelines:**\n"
            "1. **Research Papers & Technical Reports:** You MUST include the 'translate' step. "
            "   Skipping 'translate' leads to empty summaries. Do not optimize it away.\n"
            "2. **Simple/Short Texts:** You may skip 'timeline' or 'jargon' if irrelevant.\n"
            "3. **User Requests:** If `history_requested` is true, you MUST include 'timeline'.\n"
            "4. **Standard flow:** parse -> translate -> [jargon/citizen/timeline] -> "
            "review -> brief.\n\n"
            "Return ONLY a single JSON object with this shape:\n"
            "{\n"
            '  "steps": ["parse", "translate", ...],\n'
            '  "enable_history": false,\n'
            '  "readability_threshold": 0.75,\n'
            '  "factuality_threshold": 0.80,\n'
            '  "max_refine_rounds": 1,\n'
            '  "notes": "Rationale for the plan..."\n'
            "}"
        )

    def _build_initial_user_prompt(self, bb: Blackboard, ctx: PlannerContext) -> str:
        parsed: list[Any] = bb.get("parsed_chunks") or []
        preview = ""
        if parsed:
            first = parsed[0]
            if isinstance(first, dict):
                preview = str(first.get("text", ""))
            else:
                preview = str(first)

        # FIX: Increased context window from 800 -> 2000 chars.
        # This ensures the Planner sees the Abstract, Intro, and Metadata of research papers,
        # preventing it from incorrectly classifying them as "simple text".
        preview = preview.strip()[:2000]

        return (
            "Document Preview (First 2000 chars):\n"
            f"{preview or '[no preview available]'}\n\n"
            "Execution Context:\n"
            f"- Task Type: {ctx.task_type}\n"
            f"- Document Kind: {ctx.document_kind}\n"
            f"- Language: {ctx.language}\n"
            f"- History Requested: {ctx.enable_history_requested}\n\n"
            "Instructions:\n"
            "Based on the preview, decide the necessary steps. "
            "If this looks like a research paper (e.g. has Abstract, Introduction), "
            "ensure 'translate' and 'jargon' are included."
        )

    def plan(self, bb: Blackboard, ctx: PlannerContext) -> PlannerPlanSpec:
        """
        Generate the initial execution plan based on document preview.

        Parameters
        ----------
        bb:
            Blackboard containing ``parsed_chunks``.
        ctx:
            Contextual metadata (user intent, document type).

        Returns
        -------
        PlannerPlanSpec
            The initial execution plan to be converted into a DAG.
        """
        messages: Sequence[Mapping[str, str]] = [
            {"role": "system", "content": self._initial_system_prompt()},
            {"role": "user", "content": self._build_initial_user_prompt(bb, ctx)},
        ]

        response_text = self.llm.generate(
            messages=messages,
            model=self.model_alias,
            temperature=0.2,
            max_tokens=512,
        )

        try:
            llm_plan = PlannerLLMOutput.model_validate_json(response_text)
        except (ValidationError, TypeError) as exc:
            raise RuntimeError(
                f"PlannerAgent: Invalid JSON in initial plan: {exc}\n{response_text}"
            ) from exc

        plan_spec = PlannerPlanSpec(
            strategy="llm_planner.v1",
            steps=llm_plan.steps,
            enable_history=llm_plan.enable_history,
            notes=llm_plan.notes,
            # Initialize replan fields as empty (Step 5.3)
            should_replan=False,
            replan_steps=None,
            replan_reason=None,
        )

        bb.put("planner_plan_spec.initial", plan_spec.model_dump())
        return plan_spec

    # ------------------------------------------------------------------ #
    # Re-planning / Refinement (Step 5.3)
    # ------------------------------------------------------------------ #

    def _replan_system_prompt(self) -> str:
        allowed_list = ", ".join(sorted(ALLOWED_REFINE_STEPS))
        return (
            "You are the Re-Planner for the InterLines system. "
            "You review the quality report from the Editor and decide if "
            "refinement is needed.\n\n"
            f"Allowed refinement steps: {allowed_list}\n"
            "  - *_refine steps allow agents to improve their output.\n"
            "  - 'editor' must usually be included at the end to verify fixes.\n\n"
            "Return ONLY a single JSON object:\n"
            "{\n"
            '  "should_replan": true,\n'
            '  "replan_steps": ["explainer_refine", "citizen_refine", "editor"],\n'
            '  "replan_reason": "Readability is too low (0.4 < 0.7)."\n'
            "}"
        )

    def _build_replan_user_prompt(
        self,
        previous_plan: PlannerPlanSpec,
        report: ReviewReport,
    ) -> str:
        # Extract key metrics for the LLM
        readability = report.criteria.clarity  # Map clarity -> readability
        factuality = report.criteria.accuracy
        completeness = report.criteria.completeness
        overall = report.overall

        issues = "\n".join(f"- {c}" for c in report.comments[:10])
        actions = "\n".join(f"- {a}" for a in report.actions[:5])

        return (
            "Previous Plan Summary:\n"
            f"- Steps: {previous_plan.steps}\n"
            f"- Notes: {previous_plan.notes}\n\n"
            "Editor Review Report:\n"
            f"- Overall Score: {overall:.2f}\n"
            f"- Readability (Clarity): {readability:.2f}\n"
            f"- Factuality (Accuracy): {factuality:.2f}\n"
            f"- Completeness: {completeness:.2f}\n\n"
            f"Key Issues Identified:\n{issues or '(none)'}\n\n"
            f"Suggested Actions:\n{actions or '(none)'}\n\n"
            "Decision:\n"
            "If scores are low (< 0.7) or critical issues exist, trigger a replan "
            "with appropriate *_refine steps and the editor. Otherwise, set "
            "should_replan to false."
        )

    def replan(
        self,
        bb: Blackboard,
        ctx: PlannerContext,
        previous_plan: PlannerPlanSpec,
        review_report: ReviewReport,
    ) -> PlannerPlanSpec:
        """
        Evaluate the review report and decide whether to trigger a refinement loop.

        This method corresponds to Step 5.3 of the Planner Agent roadmap.
        It calls the LLM with the review report to determine if the pipeline
        output meets quality standards.

        Parameters
        ----------
        bb:
            The shared :class:`Blackboard` instance. While not heavily used
            for input (data comes via arguments), it is used for tracing.
        ctx:
            Planner context containing task metadata.
        previous_plan:
            The :class:`PlannerPlanSpec` from the *initial* pass. Used to
            understand what was already attempted.
        review_report:
            The :class:`ReviewReport` produced by the Editor Agent, containing
            scores (accuracy, clarity) and specific comments.

        Returns
        -------
        PlannerPlanSpec
            A **new** plan specification. If ``should_replan`` is True, the
            ``replan_steps`` field will contain the sequence for the
            second pass.

        Side Effects
        ------------
        - Writes the decision to blackboard under ``"planner_plan_spec.replan"``.
        - Emits a trace log if JSON parsing fails.
        """
        messages: Sequence[Mapping[str, str]] = [
            {"role": "system", "content": self._replan_system_prompt()},
            {
                "role": "user",
                "content": self._build_replan_user_prompt(previous_plan, review_report),
            },
        ]

        response_text = self.llm.generate(
            messages=messages,
            model=self.model_alias,
            temperature=0.2,
            max_tokens=512,
        )

        try:
            llm_output = ReplanLLMOutput.model_validate_json(response_text)
        except (ValidationError, TypeError) as exc:
            # Graceful degradation: if the LLM flubs the JSON, we log it
            # and default to NOT replanning (safety first).
            bb.trace(f"planner: replan JSON error, defaulting to stop. {exc}")
            llm_output = ReplanLLMOutput(
                should_replan=False,
                replan_steps=[],
                replan_reason=f"JSON parsing failed: {exc}",
            )

        # Validate suggested steps against the strict allowlist.
        valid_steps: list[str] = []
        if llm_output.should_replan:
            for step in llm_output.replan_steps:
                if step in ALLOWED_REFINE_STEPS:
                    valid_steps.append(step)

            # If the LLM wanted to replan but gave invalid steps (or empty),
            # we enforce a minimal "check again" fallback.
            if not valid_steps and llm_output.replan_steps:
                valid_steps = ["editor"]

        # Construct the updated plan spec.
        # Note: We preserve the *initial* steps for record-keeping, but
        # set the *replan* fields for the next execution phase.
        new_plan = PlannerPlanSpec(
            strategy=previous_plan.strategy,
            steps=previous_plan.steps,
            enable_history=previous_plan.enable_history,
            notes=previous_plan.notes,
            should_replan=llm_output.should_replan and bool(valid_steps),
            replan_steps=valid_steps if llm_output.should_replan else None,
            replan_reason=llm_output.replan_reason,
        )

        bb.put("planner_plan_spec.replan", new_plan.model_dump())
        return new_plan


__all__ = [
    "PlannerAgent",
    "PlannerContext",
    "PlannerLLMOutput",
    "ReplanLLMOutput",
]
