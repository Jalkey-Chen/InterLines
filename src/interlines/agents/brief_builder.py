"""
AI-powered Brief Builder Agent (The "Chief Editor").

This agent acts as the final synthesizer. Instead of using a rigid template,
it reads all high-level artifacts from the blackboard (Explanations, History,
Glossary, Relevance Notes, QA Report) and uses an LLM to compose a cohesive,
well-structured Markdown public brief.

Responsibilities
----------------
- Context Assembly: Serialize all artifacts into a structured context for the LLM.
- AI Composition: Ask the LLM (Role: Science Communicator/Editor) to write
  the brief, dynamically deciding the best flow and layout based on the content.
- Persistence: Save the generated Markdown to `artifacts/reports/<run_id>.md`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from interlines.core.blackboard.memory import Blackboard
from interlines.core.result import Result, err, ok
from interlines.llm.client import LLMClient

# Blackboard keys
_EXPLANATIONS_KEY = "explanations"
_TERMS_KEY = "terms"
_TIMELINE_KEY = "timeline_events"
_RELEVANCE_NOTES_KEY = "relevance_notes"
_REVIEW_REPORT_KEY = "review_report"

# Output key
_PUBLIC_BRIEF_MD_KEY = "public_brief_md_path"

# Model configuration
# Aligned with models.py definition for "brief_builder"
_BUILDER_MODEL_ALIAS = "brief_builder"
_DEFAULT_REPORTS_DIR = Path("artifacts") / "reports"


def _get_llm_client() -> LLMClient:
    """Return the shared LLM client."""
    return LLMClient.from_env()


def _serialize_artifacts(bb: Blackboard) -> str:
    """
    Serialize all relevant blackboard artifacts into a JSON string context.

    This provides the raw material for the AI Editor to work with.
    """

    # We use explicit Pydantic dumping if available, or simple dict access
    def _dump(val: Any) -> Any:
        if hasattr(val, "model_dump"):
            return val.model_dump()
        if isinstance(val, list):
            return [_dump(v) for v in val]
        return val

    context = {
        "explanations": _dump(bb.get(_EXPLANATIONS_KEY)),
        "timeline": _dump(bb.get(_TIMELINE_KEY)),
        "glossary": _dump(bb.get(_TERMS_KEY)),
        "relevance_notes": _dump(bb.get(_RELEVANCE_NOTES_KEY)),
        "quality_report": _dump(bb.get(_REVIEW_REPORT_KEY)),
    }

    # Prune empty keys to save tokens and reduce noise
    clean_context = {k: v for k, v in context.items() if v}

    return json.dumps(clean_context, indent=2, ensure_ascii=False)


def _build_editor_prompt(context_json: str) -> list[dict[str, str]]:
    """Construct the prompt for the AI Chief Editor."""
    # Use implicit string concatenation to satisfy line-length limits (ruff E501)
    # while preserving the exact formatting of the prompt.
    system_content = (
        "You are the Chief Editor of 'InterLines', a publication that translates "
        "complex topics for the general public.\n\n"
        "Your task is to synthesize a set of raw research notes into a beautiful, "
        "cohesive, and readable Markdown report.\n\n"
        "**Input Data:**\n"
        "You will be provided with a JSON object containing:\n"
        "1. `explanations`: The core logic and claims (Expert layer).\n"
        "2. `relevance_notes`: Why this matters to specific people (Citizen layer).\n"
        "3. `timeline`: Historical events (History layer).\n"
        "4. `glossary`: Technical term definitions.\n"
        "5. `quality_report`: An internal quality score.\n\n"
        "**Composition Guidelines:**\n"
        "- **Structure**: Do NOT just list the inputs. Organize them into a narrative flow.\n"
        "    - Start with a catchy Title and a strong Introduction "
        "(mix the Summary with Relevance Notes).\n"
        "    - Use the `explanations` to build the main body. Use clear headings.\n"
        "    - Weave the `glossary` definitions naturally into the text OR create a "
        '"Key Concepts" sidebar section if there are many terms.\n'
        "    - If a `timeline` exists, present it where it adds the most context "
        "(usually after the intro or at the end).\n"
        "    - **Formatting**: Use Markdown heavily. Use bolding for emphasis, "
        "bullet points for readability, and blockquotes for key takeaways.\n"
        "    - **Tone**: Engaging, clear, objective, but accessible "
        "(New York Times Science section style).\n\n"
        "**Critical Rules:**\n"
        "- **Truthfulness**: Do NOT invent new facts. Use only the information "
        "provided in the JSON.\n"
        "- **Completeness**: You MUST include the content from the `explanations` "
        "and `relevance_notes`.\n"
        "- **Transparency**: At the very bottom of the report, add a small "
        '"About this Brief" footer. Include the "Overall Score" from the '
        "`quality_report` if available, to show the reader we verify our work.\n\n"
        "Output ONLY the raw Markdown content. Do not output markdown fences (```markdown)."
    )

    user_content = (
        f"Here are the raw artifacts for this run:\n\n{context_json}\n\n"
        "Please compose the final Public Brief in Markdown."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def run_brief_builder(
    bb: Blackboard,
    *,
    run_id: str = "run",
    reports_dir: Path | str | None = None,
) -> Result[Path, str]:
    """
    Run the AI Brief Builder to compose a Markdown report.

    Parameters
    ----------
    bb:
        Shared blackboard instance.
    run_id:
        Identifier for the filename.
    reports_dir:
        Directory to save the file.

    Returns
    -------
    Result[Path, str]
        Path to the generated file on success.
    """
    # 1. Prepare Data
    context_str = _serialize_artifacts(bb)

    # If the context is essentially empty (just "{}"), we should probably fail
    # or return a warning, but let's see if the LLM can handle "No data".
    # For safety, let's enforce at least some content.
    if len(context_str) < 10:
        return err("brief_builder: No artifacts found on blackboard to compose.")

    # 2. Call LLM
    client = _get_llm_client()
    messages = _build_editor_prompt(context_str)

    try:
        markdown_content = client.generate(
            messages,
            model=_BUILDER_MODEL_ALIAS,
            # We use a slightly higher temperature for "Creative Layout" tasks,
            # but model registry defaults (0.4) will override this if we don't set it.
            # Let's set it explicitly to encourage better writing flow.
            temperature=0.7,
            max_tokens=2000,
        )
    except Exception as exc:
        return err(f"brief_builder LLM error: {exc}")

    # 3. Clean and Save
    # Remove markdown fences if the model included them despite instructions
    clean_md = markdown_content.strip()
    if clean_md.startswith("```markdown"):
        clean_md = clean_md.replace("```markdown", "", 1)
    if clean_md.startswith("```"):
        clean_md = clean_md.replace("```", "", 1)
    if clean_md.endswith("```"):
        clean_md = clean_md[:-3]

    clean_md = clean_md.strip()

    reports_path = Path(reports_dir) if reports_dir else _DEFAULT_REPORTS_DIR
    reports_path.mkdir(parents=True, exist_ok=True)

    filename = f"{run_id}.md" if run_id else "brief.md"
    output_path = reports_path / filename

    try:
        output_path.write_text(clean_md, encoding="utf-8")
    except OSError as exc:
        return err(f"brief_builder I/O error: {exc}")

    bb.put(_PUBLIC_BRIEF_MD_KEY, str(output_path))

    return ok(output_path)


__all__ = ["run_brief_builder", "_PUBLIC_BRIEF_MD_KEY"]
