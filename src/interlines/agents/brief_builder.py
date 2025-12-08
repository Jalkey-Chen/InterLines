"""
Markdown brief builder agent.

This agent reads higher-level artifacts from the blackboard
(explanations, terms, timeline events) and compiles them into a
single human-facing Markdown file.

Responsibilities
----------------
- Collect:
  - ExplanationCard instances under "explanations"
  - TermCard instances under "terms"
  - TimelineEvent instances under "timeline_events"
- Render a Markdown document with three main sections:
  1. Overview (based on explanations)
  2. Key terms (glossary-style)
  3. Timeline (historical evolution)
- Write the Markdown file under ``artifacts/reports/<run_id>.md``
  and store the resulting path on the blackboard under
  ``"public_brief_md_path"``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

from interlines.core.blackboard.memory import Blackboard
from interlines.core.result import Result, err, ok

# Blackboard keys (kept consistent with other agents).
_PARSED_CHUNKS_KEY = "parsed_chunks"
_EXPLANATIONS_KEY = "explanations"
_TERMS_KEY = "terms"
_TIMELINE_KEY = "timeline_events"

# Where we store the generated Markdown path on the blackboard.
_PUBLIC_BRIEF_MD_KEY = "public_brief_md_path"

# Default relative directory for Markdown briefs.
_DEFAULT_REPORTS_DIR = Path("artifacts") / "reports"


def _as_list(value: Any) -> list[Any]:
    """Normalize an arbitrary value into a list for iteration.

    - None           -> []
    - list           -> itself
    - other Sequence -> list(value)
    - everything else -> [value]
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return list(value)
    return [value]


def _get_field(obj: Any, key: str) -> Any:
    """Retrieve a field from a Mapping or an object via attribute access."""
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _get_str_field(obj: Any, key: str) -> str:
    """Return a stripped string representation of obj[key] (or empty string)."""
    value = _get_field(obj, key)
    if value is None:
        return ""
    return str(value).strip()


def _normalise_date(value: Any) -> str:
    """Convert a TimelineEvent.when-like value into a readable string."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if value is None:
        return ""
    return str(value)


def _ensure_reports_dir(base: Path | str | None = None) -> Path:
    """Ensure the reports directory exists and return it."""
    if base is None:
        base_path = _DEFAULT_REPORTS_DIR
    else:
        base_path = Path(base)
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


def _build_overview_section(explanations: Sequence[Any]) -> tuple[str, str]:
    """Build the document title and the Overview section.

    Returns
    -------
    title: str
        Top-level document title.
    section_md: str
        Markdown for the overview section (without the leading '# ' title).
    """
    lines: list[str] = []

    primary_claim = ""
    if explanations:
        primary_claim = _get_str_field(explanations[0], "claim")

    if primary_claim:
        title = primary_claim
    else:
        title = "InterLines Public Brief"

    lines.append("## Overview")
    lines.append("")

    if explanations:
        for idx, card in enumerate(explanations, start=1):
            claim = _get_str_field(card, "claim")
            rationale = _get_str_field(card, "rationale")
            if not claim and not rationale:
                continue

            if claim:
                lines.append(f"### Explanation {idx}")
                lines.append("")
                lines.append(f"**Claim:** {claim}")
            else:
                lines.append(f"### Explanation {idx}")

            if rationale:
                lines.append("")
                lines.append(rationale)
            lines.append("")
    else:
        lines.append("_No explanations were available on the blackboard._")
        lines.append("")

    overview_md = "\n".join(lines).rstrip() + "\n"
    return title, overview_md


def _build_terms_section(terms: Sequence[Any]) -> str:
    """Build the 'Key terms' Markdown section."""
    if not terms:
        return ""

    lines: list[str] = []
    lines.append("## Key terms")
    lines.append("")

    for term_obj in terms:
        name = _get_str_field(term_obj, "term")
        if not name:
            continue

        definition = _get_str_field(term_obj, "definition")
        aliases_raw = _get_field(term_obj, "aliases")
        examples_raw = _get_field(term_obj, "examples")

        aliases: list[str] = []
        if isinstance(aliases_raw, Sequence) and not isinstance(
            aliases_raw,
            str | bytes,
        ):
            aliases = [str(a).strip() for a in aliases_raw if str(a).strip()]

        examples: list[str] = []
        if isinstance(examples_raw, Sequence) and not isinstance(
            examples_raw,
            str | bytes,
        ):
            examples = [str(e).strip() for e in examples_raw if str(e).strip()]

        lines.append(f"### {name}")
        lines.append("")
        if definition:
            lines.append(f"**Definition.** {definition}")
        if aliases:
            lines.append(f"**Also called.** {', '.join(aliases)}")
        if examples:
            lines.append("")
            lines.append("**Examples**")
            for ex in examples:
                lines.append(f"- {ex}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _build_timeline_section(events: Sequence[Any]) -> str:
    """Build the 'Timeline' Markdown section."""
    if not events:
        return ""

    # Normalise and sort by date if possible.
    records: list[tuple[str, str, str]] = []
    for ev in events:
        when_raw = _get_field(ev, "when")
        title = _get_str_field(ev, "title")
        description = _get_str_field(ev, "description")
        when_str = _normalise_date(when_raw)
        records.append((when_str, title, description))

    records.sort(key=lambda t: t[0] or "")

    lines: list[str] = []
    lines.append("## Timeline")
    lines.append("")
    for when_str, title, description in records:
        prefix = when_str or "unspecified date"
        if title:
            if description:
                lines.append(f"- {prefix} — **{title}**: {description}")
            else:
                lines.append(f"- {prefix} — **{title}**")
        else:
            if description:
                lines.append(f"- {prefix}: {description}")
            else:
                lines.append(f"- {prefix}")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _build_markdown_document(
    explanations: Sequence[Any],
    terms: Sequence[Any],
    events: Sequence[Any],
) -> str:
    """Compose the full Markdown brief from component sections."""
    title, overview_md = _build_overview_section(explanations)
    terms_md = _build_terms_section(terms)
    timeline_md = _build_timeline_section(events)

    parts: list[str] = []
    parts.append(f"# {title}")
    parts.append("")
    parts.append(overview_md.strip())

    if terms_md.strip():
        parts.append("")
        parts.append(terms_md.strip())

    if timeline_md.strip():
        parts.append("")
        parts.append(timeline_md.strip())

    return "\n".join(parts).rstrip() + "\n"


def run_brief_builder(
    bb: Blackboard,
    *,
    run_id: str = "run",
    reports_dir: Path | str | None = None,
) -> Result[Path, str]:
    """Build a Markdown public brief from artifacts on the blackboard.

    Parameters
    ----------
    bb:
        Shared blackboard instance populated by previous agents.
    run_id:
        Logical identifier for this pipeline run. Used to build the output
        filename as ``<run_id>.md`` under the reports directory.
    reports_dir:
        Optional override for the reports directory. If not provided,
        ``artifacts/reports`` is used.

    Returns
    -------
    Result[Path, str]
        Ok(path) with the generated Markdown file path on success,
        Err(message) with a human-readable error otherwise.
    """
    explanations_raw = bb.get(_EXPLANATIONS_KEY)
    terms_raw = bb.get(_TERMS_KEY)
    timeline_raw = bb.get(_TIMELINE_KEY)

    explanations = _as_list(explanations_raw)
    terms = _as_list(terms_raw)
    events = _as_list(timeline_raw)

    if not explanations and not terms and not events:
        return err(
            "brief_builder requires at least one of "
            "`explanations`, `terms`, or `timeline_events` on the blackboard",
        )

    markdown = _build_markdown_document(explanations, terms, events)

    reports_path = _ensure_reports_dir(reports_dir or _DEFAULT_REPORTS_DIR)
    filename = f"{run_id}.md" if run_id else "brief.md"
    output_path = reports_path / filename
    output_path.write_text(markdown, encoding="utf-8")

    # Store the resulting path back on the blackboard for downstream consumers.
    bb.put(_PUBLIC_BRIEF_MD_KEY, str(output_path))

    return ok(output_path)


__all__ = ["run_brief_builder", "_PUBLIC_BRIEF_MD_KEY"]
