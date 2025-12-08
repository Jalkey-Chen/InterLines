"""
Explainer agent: upgrade from stub to LLM-backed, scholar-style explanations.

Responsibilities
----------------
- Read `parsed_chunks` from the Blackboard (output of `parser_agent`).
- Call the LLM client (model alias: "explainer", role ~ scholar) with a
  structured prompt that includes numbered paragraphs.
- Parse the JSON response into three `ExplanationCard` instances:

    - level="one_sentence"
    - level="three_paragraph"
    - level="deep_dive"

- Populate:
    - `claim` / `rationale` (main explanation fields)
    - `claims[]`  (finer-grained claims per layer, if provided)
    - `provenance` (list of paragraph IDs from the parser)

- Write the cards back to the Blackboard under the "explanations" key and
  return them wrapped in a `Result`.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from interlines.core.blackboard.memory import Blackboard
from interlines.core.contracts.explanation import EvidenceItem, ExplanationCard
from interlines.core.result import Result, err, ok
from interlines.llm.client import LLMClient

# Blackboard keys
_PARSED_CHUNKS_KEY = "parsed_chunks"
_EXPLANATIONS_KEY = "explanations"

# LLM configuration
_EXPLAINER_MODEL_ALIAS = "explainer"

# Supported explanation levels in the LLM JSON payload.
_LEVELS: tuple[str, ...] = ("one_sentence", "three_paragraph", "deep_dive")


def _get_llm_client() -> LLMClient:
    """Construct an LLM client configured for the explainer agent.

    This helper is intentionally separated so that tests can monkeypatch it
    and inject a fake client without touching :class:`LLMClient` internals.
    """
    return LLMClient.from_env(default_model_alias=_EXPLAINER_MODEL_ALIAS)


def _normalise_chunks(raw: object) -> list[dict[str, str]]:
    """Convert the `parsed_chunks` payload from the blackboard into a clean list.

    Expected shapes
    ---------------
    The parser agent is expected to produce one of the following forms:

    1. A list of mapping objects:

        [
            {"id": "p1", "text": "First paragraph..."},
            {"id": "p2", "text": "Second paragraph..."},
            ...
        ]

    2. A list of strings (in which case we synthesise `id` values):

        ["First paragraph...", "Second paragraph..."]

    This function normalises both cases into a list of dictionaries with
    string ``id`` and ``text`` fields, filtering out empty items.
    """
    out: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return out

    for idx, item in enumerate(raw):
        if isinstance(item, Mapping):
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            pid = item.get("id")
            if pid is None:
                pid = f"p{idx}"
            out.append({"id": str(pid), "text": text})
        elif isinstance(item, str):
            text = item.strip()
            if not text:
                continue
            out.append({"id": f"p{idx}", "text": text})

    return out


def _build_messages(chunks: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Build scholar-style chat messages for the explainer LLM.

    The system prompt positions the model as a careful scholar, and the user
    prompt lists numbered paragraphs together with explicit instructions to
    return JSON of the following shape (no extra keys):

    .. code-block:: json

        {
          "one_sentence": {
            "claim": "...",
            "rationale": "...",
            "claims": ["...", "..."],
            "provenance_ids": ["p1", "p3"],
            "confidence": 0.8
          },
          "three_paragraph": { ... },
          "deep_dive": { ... }
        }
    """
    system_msg = {
        "role": "system",
        "content": (
            "You are a careful scholar explaining an academic or policy text "
            "to an informed but non-expert reader. You care about clarity, "
            "faithfulness to the source, and explicit citation of which "
            "paragraphs support which claims."
        ),
    }

    lines: list[str] = []
    lines.append("Here is the source text, split into numbered paragraphs:\n")
    for chunk in chunks:
        pid = str(chunk.get("id", "")).strip()
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        lines.append(f"[{pid}] {text}")

    lines.append(
        "\n\nPlease analyse the text and produce THREE layers of explanation:\n"
        "1. `one_sentence`: a single-sentence, high-level claim.\n"
        "2. `three_paragraph`: a ~3-paragraph explanation (still concise).\n"
        "3. `deep_dive`: a more detailed, structured explanation.\n\n"
        "Return ONLY a JSON object with exactly these keys:\n\n"
        "{\n"
        '  "one_sentence": {\n'
        '    "claim": "…",            // main claim in one layer\n'
        '    "rationale": "…",        // explanation in that layer\n'
        '    "claims": ["…"],         // list of sub-claims (non-empty)\n'
        '    "provenance_ids": ["…"], // paragraph IDs like "p1", "p2"\n'
        '    "confidence": 0.0-1.0    // optional numeric confidence\n'
        "  },\n"
        '  "three_paragraph": { ... },\n'
        '  "deep_dive": { ... }\n'
        "}\n\n"
        "- `claims` should be a list of shorter bullet-style claims.\n"
        "- `provenance_ids` MUST come from the paragraph IDs shown above.\n"
        "- Do not add keys other than the ones described.\n"
        "- Respond with JSON ONLY, no surrounding commentary."
    )

    user_msg = {
        "role": "user",
        "content": "\n".join(lines),
    }

    return [system_msg, user_msg]


def _parse_llm_payload(raw: str) -> dict[str, MutableMapping[str, Any]]:
    """Parse the raw LLM output string into a structured dictionary.

    The function is intentionally strict: it expects valid JSON and all three
    predefined levels (``one_sentence``, ``three_paragraph``, ``deep_dive``).
    Any deviation raises :class:`RuntimeError`, which is then surfaced as an
    ``Err`` by :func:`run_explainer`.
    """
    import json

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Explainer LLM returned invalid JSON: {exc}") from exc

    if not isinstance(data, Mapping):
        raise RuntimeError("Explainer LLM JSON root must be an object.")

    structured: dict[str, MutableMapping[str, Any]] = {}

    for level in _LEVELS:
        node = data.get(level)
        if not isinstance(node, Mapping):
            raise RuntimeError(f"Explainer LLM JSON missing level: {level!r}")
        structured[level] = dict(node)

    return structured


def _build_card(
    *,
    level: str,
    node: Mapping[str, Any],
    available_ids: set[str],
) -> ExplanationCard:
    """Create a single :class:`ExplanationCard` from one level of JSON.

    Mapping strategy
    ----------------
    - ``claim`` / ``rationale`` map directly to the contract fields.
    - ``claims[]`` (if present) are encoded as :class:`EvidenceItem` entries:
        * ``text``   = sub-claim text
        * ``source`` = human-readable provenance string
          (e.g. ``"paragraphs: p1, p2"``)
    - ``provenance_ids`` is intersected with the set of available paragraph
      IDs to avoid dangling references.

    Note that the current contract does not expose explicit provenance IDs
    as a separate field on :class:`ExplanationCard`; they are embedded into
    ``EvidenceItem.source`` for now.
    """
    claim_raw = node.get("claim") or ""
    rationale_raw = node.get("rationale") or ""

    claim = str(claim_raw).strip()
    rationale = str(rationale_raw).strip()

    # claims[]: finer-grained list from the LLM JSON; if absent, fall back
    # to a single-element list containing the main claim.
    claims_val = node.get("claims")
    claims_list: list[str] = []
    if isinstance(claims_val, list):
        for c in claims_val:
            text = str(c).strip()
            if text:
                claims_list.append(text)
    if not claims_list and claim:
        claims_list.append(claim)

    # provenance_ids: intersect with actually available paragraph IDs.
    prov_val = node.get("provenance_ids")
    provenance_ids: list[str] = []
    if isinstance(prov_val, list):
        for pid in prov_val:
            pid_str = str(pid).strip()
            if pid_str and pid_str in available_ids:
                provenance_ids.append(pid_str)

    if not provenance_ids and available_ids:
        # Fallback: if the model failed to choose, assume all paragraphs.
        provenance_ids = sorted(available_ids)

    provenance_label = f"paragraphs: {', '.join(provenance_ids)}" if provenance_ids else None

    evidence_items: list[EvidenceItem] = [
        EvidenceItem(text=c, source=provenance_label) for c in claims_list
    ]

    # The underlying Artifact base class provides defaults for `kind`,
    # `version`, and `confidence`. Here we populate the Explanation-specific
    # fields plus evidence; a small `type: ignore` keeps static checkers quiet
    # about the extra keyword arguments handled by Pydantic at runtime.
    return ExplanationCard(
        claim=claim,
        rationale=rationale,
        evidence=evidence_items,
        summary=None,
    )  # type: ignore[call-arg]


def run_explainer(bb: Blackboard) -> Result[list[ExplanationCard], str]:
    """Run the LLM-backed explainer agent on the given blackboard.

    Steps
    -----
    1. Read ``parsed_chunks`` from the blackboard (output of the parser agent).
    2. Normalise it into a list of ``{"id": ..., "text": ...}`` dictionaries.
    3. Build scholar-style messages describing the paragraphs and JSON schema.
    4. Invoke the explainer LLM (model alias: ``"explainer"``).
    5. Parse the returned JSON into three :class:`ExplanationCard` instances.
    6. Write the cards back under the ``"explanations"`` key and return them.

    On any failure (missing chunks, LLM error, JSON mismatch), an ``Err`` with
    a human-readable message is returned instead.
    """
    raw_chunks = bb.get(_PARSED_CHUNKS_KEY)
    chunks = _normalise_chunks(raw_chunks)

    if not chunks:
        return err("explainer requires non-empty `parsed_chunks` on the blackboard")

    client = _get_llm_client()
    messages = _build_messages(chunks)

    try:
        raw = client.generate(messages, model=_EXPLAINER_MODEL_ALIAS)
        payload = _parse_llm_payload(raw)
    except Exception as exc:  # pragma: no cover - defensive, but helpful in prod
        return err(f"explainer LLM error: {exc}")

    available_ids = {c["id"] for c in chunks if "id" in c}

    cards: list[ExplanationCard] = []
    for level in _LEVELS:
        node = payload[level]
        card = _build_card(level=level, node=node, available_ids=available_ids)
        cards.append(card)

    bb.put(_EXPLANATIONS_KEY, cards)

    return ok(cards)
