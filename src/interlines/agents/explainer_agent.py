"""
Stub explainer agent for the InterLines pipeline.

This module provides a minimal, fully synchronous "explainer" that generates
three levels of explanation for a given input:

- one_sentence   — a very short headline-style gist
- three_paragraph — a medium-length summary
- deep_dive      — a longer, more detailed commentary

At this stage the agent does NOT call any LLMs. Instead, it produces
structured placeholder objects that mimic the shape of an ExplanationCard
contract. The goal is to unblock early pipeline and UI work while keeping
a clear upgrade path toward real model-backed explanations.

Design notes
------------
* The agent writes its output into the shared Blackboard under a configurable
  key (default: "explanations").
* The input it reads is "parsed_chunks" by default, as produced by the
  parser agent in Step 2.1. If that key is missing, it falls back to a
  generic placeholder seed text.
* We deliberately keep the payload as plain dicts[str, Any] for now, so
  that we do not depend too strongly on the exact Pydantic model details.
  Later, this can be swapped for real ExplanationCard instances.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

# Canonical explanation levels used across the system.
LEVEL_ONE_SENTENCE: Final[str] = "one_sentence"
LEVEL_THREE_PARAGRAPH: Final[str] = "three_paragraph"
LEVEL_DEEP_DIVE: Final[str] = "deep_dive"

DEFAULT_LEVELS: Final[tuple[str, str, str]] = (
    LEVEL_ONE_SENTENCE,
    LEVEL_THREE_PARAGRAPH,
    LEVEL_DEEP_DIVE,
)


def _build_stub_explanation(level: str, seed: str) -> dict[str, Any]:
    """
    Build a stub "ExplanationCard-like" dictionary for a given level.

    Parameters
    ----------
    level:
        The explanation level identifier. Expected values are:
        "one_sentence", "three_paragraph", or "deep_dive".
    seed:
        Short text used as the semantic seed for this explanation, typically
        the first parsed chunk or the raw input text.

    Returns
    -------
    dict[str, Any]
        A dictionary that roughly follows the intended ExplanationCard v1
        contract: it includes `kind`, `version`, `claim`, `rationale`,
        `confidence`, `provenance`, and a `level` field describing the
        explanation depth.
    """
    # Human-readable label for each level, used as a claim headline.
    if level == LEVEL_ONE_SENTENCE:
        label = "One-sentence gist"
    elif level == LEVEL_THREE_PARAGRAPH:
        label = "Three-paragraph summary"
    elif level == LEVEL_DEEP_DIVE:
        label = "Deep-dive commentary"
    else:
        # This should not normally happen; we keep it defensive in case
        # future callers extend the levels list.
        label = f"Explanation ({level})"

    # We keep the rationale text intentionally generic and obviously stubby,
    # so it is easy to spot in logs and UI during early development.
    truncated_seed = seed[:140]
    rationale = (
        f"[stub:{level}] {label} for input starting with: {truncated_seed!r}. "
        "This will be replaced by a real model-backed explanation in a later step."
    )

    return {
        # Contract metadata — loosely mirroring explanation.v1.json
        "kind": "explanation.v1",
        "version": "v1",
        "level": level,
        "claim": label,
        "rationale": rationale,
        "confidence": 0.5,
        # Minimal provenance: we keep a pointer saying that this explanation
        # comes from the primary input text. Later we can attach paragraph IDs
        # or sentence indices.
        "provenance": [
            {
                "source": "blackboard",
                "key": "parsed_chunks",
                "note": "stub explainer used the first available chunk as seed",
            }
        ],
    }


def _extract_seed_text(source: Any) -> str:
    """
    Derive a seed text from the parser output stored on the blackboard.

    The parser agent typically writes a list of chunks (paragraphs or sections)
    under the key "parsed_chunks". We take the first non-empty chunk as the
    seed. If the structure is different, we fall back to a simple string cast.

    Parameters
    ----------
    source:
        Raw value retrieved from the blackboard at the parser output key.

    Returns
    -------
    str
        A short text snippet that can be used as the seed for explanations.
        If nothing sensible is available, a generic placeholder string is
        returned instead.
    """
    # Common case: list of paragraph strings.
    if isinstance(source, list) and source:
        first = source[0]
        if isinstance(first, str) and first.strip():
            return first.strip()

    # Already a string (e.g., raw text directly stored).
    if isinstance(source, str) and source.strip():
        return source.strip()

    # Last-resort fallback: make it obvious that the parser did not provide
    # anything useful.
    return "No parser output available; this is a stub explanation."


def run_explainer_stub(
    bb: Any,
    *,
    source_key: str = "parsed_chunks",
    target_key: str = "explanations",
    levels: Sequence[str] = DEFAULT_LEVELS,
) -> list[dict[str, Any]]:
    """
    Run the stub explainer agent against the given blackboard.

    This function reads the parser output from `source_key`, generates a set
    of three ExplanationCard-like dictionaries (one per level), and writes
    them back to the blackboard under `target_key`.

    Parameters
    ----------
    bb:
        A Blackboard-like object. It is expected to implement `get(key: str)`
        and `put(key: str, value: Any) -> None`. We keep the type generic
        (`Any`) here to avoid import cycles between core modules.
    source_key:
        Blackboard key where the parser agent wrote its output. Defaults
        to "parsed_chunks".
    target_key:
        Blackboard key under which the list of explanation objects will be
        stored. Defaults to "explanations".
    levels:
        Iterable of explanation level identifiers to generate. By default we
        generate the canonical trio: one_sentence, three_paragraph, deep_dive.

    Returns
    -------
    list[dict[str, Any]]
        The list of generated explanation objects. The same list reference is
        also written into the blackboard at `target_key`.

    Notes
    -----
    * This function is deliberately deterministic and side-effect free,
      except for the single `bb.put` call. That makes it very easy to test.
    * In future steps, this stub will be replaced by a real explainer agent
      that calls an LLM and produces semantically rich explanations while
      preserving the same output shape and keys.
    """
    source = bb.get(source_key)
    seed_text = _extract_seed_text(source)

    cards: list[dict[str, Any]] = [_build_stub_explanation(level, seed_text) for level in levels]

    bb.put(target_key, cards)
    return cards
