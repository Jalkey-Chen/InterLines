"""
PlannerAgent v1 — LLM-backed semantic routing for the InterLines pipeline.

Milestone
---------
M5 | Planner Agent
Step 5.2 | LLM-backed Planner v1

This module introduces the *first* LLM-backed planner for the
public-translation pipeline. It does **not** yet modify the pipeline
execution order by itself; instead, it focuses on:

1. Defining a clear *planning contract* between the LLM and the
   orchestration layer (``PlannerLLMOutput`` → ``PlannerPlanSpec``).
2. Providing a small, explicit ``PlannerContext`` object so that the
   planner can reason about:
      - task type (currently always "public_translation"),
      - document type,
      - approximate length,
      - language,
      - whether the user requested a historical lens.
3. Implementing ``PlannerAgent.plan(...)`` which:
      - builds a prompt using document preview + context metadata,
      - calls the LLM via :class:`LLMClient`,
      - expects a strict JSON object with routing + thresholds,
      - validates it with Pydantic,
      - converts it into :class:`PlannerPlanSpec`,
      - stores it on the :class:`Blackboard` under
        ``"planner_plan_spec.initial"``.

Later commits in M5 will:

- Connect this planner into ``run_pipeline`` via a ``use_llm_planner``
  flag (defaulting to ``True``).
- Use the returned ``PlannerPlanSpec`` to construct a DAG and drive
  execution (including retries and refinement).
- Extend the schema with more advanced strategies (multi-round refine,
  step-specific options, etc.).

For now, this module is deliberately self-contained: importing it does
not alter any pipeline behaviour. It is safe to land as a first commit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.planner import PlannerPlanSpec
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
    access to the entire raw text (only a preview is passed via the
    blackboard).

    Parameters
    ----------
    task_type:
        High-level task category. For the current PKI / InterLines project
        this will almost always be ``"public_translation"`` but the field is
        kept generic for future reuse.
    document_kind:
        Optional descriptor of the document genre, such as ``"policy"``,
        ``"research"``, ``"news"``, ``"legal"``, etc. Downstream callers are
        free to leave this as ``None`` if they have no better guess.
    approx_char_count:
        Rough estimate of the input document length in characters. This is
        intended as a coarse-grained signal (e.g. short memo vs. 80-page
        report), not an exact value.
    language:
        Two-letter language code for the *original* document, e.g. ``"en"``,
        ``"zh"``, ``"es"``. The planner can use this to reason about which
        steps matter (e.g. jargon vs. citizen).
    enable_history_requested:
        Whether the *user* explicitly requested a historical / timeline
        perspective. The planner may override this depending on the document
        type and length, but this flag should still be surfaced as a strong
        hint in the prompt.

    Notes
    -----
    - The planner only uses :class:`PlannerContext` as *input*; it does not
      mutate or store it. All persistent decisions are recorded in
      :class:`PlannerPlanSpec` and on the :class:`Blackboard`.
    """

    task_type: str = "public_translation"
    document_kind: str | None = None
    approx_char_count: int = 0
    language: str = "en"
    enable_history_requested: bool = False


# --------------------------------------------------------------------------- #
# PlannerLLMOutput — JSON schema expected from the planner model
# --------------------------------------------------------------------------- #


class PlannerLLMOutput(BaseModel):
    """
    Validated JSON output schema for the LLM-backed planner.

    This model defines the *external* contract between the planner model
    (e.g. Grok / GPT / Gemini) and the PKI / InterLines orchestration layer.

    Fields
    ------
    steps:
        Ordered list of logical pipeline step names. Each element must be one
        of::

            "parse"

            "translate"
            "citizen"
            "jargon"
            "timeline"
            "narrate"
            "review"
            "brief"

        The list may omit steps that are unnecessary for the current
        document, but it should always include ``"parse"`` and ``"brief"``.
        Enforcement is currently left to downstream validation (future work).
    enable_history:
        Boolean indicating whether the history / timeline branch should be
        activated. This will typically correlate with the presence of the
        ``"timeline"`` step in ``steps``.
    readability_threshold:
        Optional readability score threshold in the range ``[0, 1]`` above
        which the planner is satisfied with the current draft. This is
        **advisory** in v1; later commits can wire this into the refinement
        loop.
    factuality_threshold:
        Optional factuality / hallucination threshold, also in ``[0, 1]``.
        Similarly advisory in v1.
    max_refine_rounds:
        Optional upper bound on how many refinement passes the pipeline
        should attempt if quality goals are not met.
    notes:
        Optional natural-language justification of the plan. This is meant
        for developers and future "planner report" UI surfaces rather than
        for end-users.

    The planner agent will map this model into an internal
    :class:`PlannerPlanSpec` record, which is the *authoritative* planning
    representation within the core layer.
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


# --------------------------------------------------------------------------- #
# PlannerAgent — LLM-backed semantic routing (v1)
# --------------------------------------------------------------------------- #


class PlannerAgent:
    """
    LLM-backed planner for semantic routing in the public-translation pipeline.

    This v1 implementation focuses on **initial planning only**:

    - It does not perform re-planning or dynamic retries.
    - It does not yet encode per-step options or multi-branch flows.
    - It simply asks the LLM to choose:

        - which steps to run,
        - whether to enable history,
        - optional quality thresholds,
        - an explanatory note.

    Integration points
    ------------------
    - The agent expects a :class:`Blackboard` with a ``"parsed_chunks"`` key
      holding the parser output. It uses the first chunk as a preview.
    - It uses :class:`LLMClient` with a *model alias* (default: ``"planner"``)
      that is resolved by :mod:`interlines.llm.models`.
    - It produces a :class:`PlannerPlanSpec` instance and writes its dict
      representation to the blackboard under ``"planner_plan_spec.initial"``.

    This class does **not** modify ``run_pipeline`` directly; that is handled
    by a subsequent commit which introduces a ``use_llm_planner`` flag and
    DAG construction from :class:`PlannerPlanSpec`.
    """

    def __init__(self, llm: LLMClient, *, model_alias: str = "planner") -> None:
        """
        Create a new PlannerAgent.

        Parameters
        ----------
        llm:
            Shared :class:`LLMClient` instance used to talk to provider APIs.
        model_alias:
            Logical model alias to use for planning (default: ``"planner"``).
            The alias is resolved by :func:`interlines.llm.models.get_model`
            inside the client.
        """
        self.llm = llm
        self.model_alias = model_alias

    # ------------------------------------------------------------------ #
    # Prompt construction helpers
    # ------------------------------------------------------------------ #

    def _system_prompt(self) -> str:
        """
        Build the system message for the planner model.

        The system prompt explains:

        - The high-level goal of the InterLines pipeline.
        - The list of allowed logical steps.
        - The required JSON schema to output.

        The model is *not* asked to generate natural-language commentary
        around the JSON; only a JSON object is expected in the final answer.
        """
        return (
            "You are the PlannerAgent in the InterLines (PKI) system. "
            "Your job is to design an efficient sequence of analysis steps "
            "that transforms an expert-facing document into a clear, "
            "public-friendly brief.\n\n"
            "You are NOT writing the brief itself. Instead, you decide which "
            "pipeline steps should run and in what order.\n\n"
            "Allowed logical steps (use these exact strings):\n"
            "  - parse      : segment and structure the raw document\n"
            "  - translate  : produce semantic explanation cards\n"
            "  - citizen    : reason about public relevance and 'why it matters'\n"
            "  - jargon     : extract and explain key technical terms\n"
            "  - timeline   : add a historical / temporal lens\n"
            "  - narrate    : assemble a coherent narrative from explanations\n"
            "  - review     : run a critical review pass (quality, bias, errors)\n"
            "  - brief      : assemble a public-facing brief / report\n\n"
            "You MUST return a single JSON object with this exact shape:\n"
            "{\n"
            '  "steps": ["parse", "translate", ...],\n'
            '  "enable_history": false,\n'
            '  "readability_threshold": 0.75,\n'
            '  "factuality_threshold": 0.80,\n'
            '  "max_refine_rounds": 1,\n'
            '  "notes": "Optional planning rationale."\n'
            "}\n\n"
            "Do not include any extra keys. Do not wrap the JSON in Markdown. "
            "Return ONLY the JSON object."
        )

    def _build_user_prompt(self, bb: Blackboard, ctx: PlannerContext) -> str:
        """
        Build the user message with a short document preview and metadata.

        The planner should see a *small* excerpt of the parsed document plus
        contextual signals that influence the routing decision.

        Preview strategy
        ----------------
        - We read the ``\"parsed_chunks\"`` key from the blackboard.
        - If present and non-empty, we take the first element.
        - If that element is a mapping, we use its ``\"text\"`` field;
          otherwise we cast it to ``str``.
        - The preview is truncated to ~500 characters to keep the prompt
          compact and to encourage general planning rather than local editing.
        """
        parsed: list[Any] = bb.get("parsed_chunks") or []
        preview = ""

        if parsed:
            first = parsed[0]
            if isinstance(first, dict):
                preview = str(first.get("text", ""))
            else:
                preview = str(first)

        preview = preview.strip()
        if len(preview) > 500:
            preview = preview[:500] + " ..."

        return (
            "Here is a short preview of the document you need to plan for:\n\n"
            f"{preview or '[no preview available]'}\n\n"
            "Context for planning:\n"
            f"- task_type: {ctx.task_type}\n"
            f"- document_kind: {ctx.document_kind or 'unknown'}\n"
            f"- approx_char_count: {ctx.approx_char_count}\n"
            f"- language: {ctx.language}\n"
            f"- user_requested_history: {ctx.enable_history_requested}\n\n"
            "Based on this preview and context, choose an appropriate set of "
            "steps and whether to enable history. If the document is short or "
            "does not meaningfully benefit from a historical perspective, you "
            "may disable history and omit the 'timeline' step. If the document "
            "is long, policy-heavy, or clearly anchored in events over time, "
            "you should consider enabling history and including 'timeline'.\n\n"
            "Remember: you must respond with a single JSON object only."
        )

    # ------------------------------------------------------------------ #
    # Main entrypoint
    # ------------------------------------------------------------------ #

    def plan(self, bb: Blackboard, ctx: PlannerContext) -> PlannerPlanSpec:
        """
        Generate a :class:`PlannerPlanSpec` by calling the planner model.

        Parameters
        ----------
        bb:
            The shared :class:`Blackboard` instance. The planner reads
            ``\"parsed_chunks\"`` from it to construct a document preview.
        ctx:
            :class:`PlannerContext` containing structured metadata about the
            current planning request.

        Returns
        -------
        PlannerPlanSpec
            The internal planning representation, ready to be converted into
            a DAG by the core planner in later commits.

        Side Effects
        ------------
        - Writes the JSON-serialised plan spec to the blackboard under the
          key ``\"planner_plan_spec.initial\"``. This is primarily for
          debugging, tracing, and future planner-report features.

        Raises
        ------
        RuntimeError
            If the planner model does not return valid JSON conforming to
            :class:`PlannerLLMOutput`, or if validation fails.
        """
        messages: Sequence[Mapping[str, str]] = [
            {
                "role": "system",
                "content": self._system_prompt(),
            },
            {
                "role": "user",
                "content": self._build_user_prompt(bb, ctx),
            },
        ]

        # Delegate to the shared LLM client. The model alias is resolved by
        # the client into a concrete provider + model ID.
        response_text = self.llm.generate(
            messages=messages,
            model=self.model_alias,
            temperature=0.2,
            max_tokens=512,
        )

        # The model is instructed to return *only* a JSON object. We still
        # wrap parsing in a try/except to provide a clear error message in
        # case of deviation.
        try:
            llm_plan = PlannerLLMOutput.model_validate_json(response_text)
        except (ValidationError, TypeError) as exc:
            raise RuntimeError(
                "PlannerAgent: LLM returned invalid JSON; cannot "
                "construct planning specification.\n\n"
                f"Raw response:\n{response_text}"
            ) from exc

        # Map LLM-facing schema onto the internal core contract.
        plan_spec = PlannerPlanSpec(
            strategy="llm_planner.v1",
            steps=llm_plan.steps,
            enable_history=llm_plan.enable_history,
            notes=llm_plan.notes,
        )

        # Store a serialised copy on the blackboard for provenance. In later
        # steps, additional planner reports / trace snapshots may refer to
        # this key.
        bb.put("planner_plan_spec.initial", plan_spec.model_dump())

        return plan_spec


__all__ = [
    "PlannerAgent",
    "PlannerContext",
    "PlannerLLMOutput",
]
