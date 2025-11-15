from __future__ import annotations

from interlines.pipelines.public_translation import (
    PipelineResult,
    run_pipeline,
)


def test_run_pipeline_basic_flow() -> None:
    """End-to-end: parser → explainer → brief, with history enabled."""
    input_text = (
        "InterLines turns expert language into public language.\n"
        "It also provides a historical lens.\n"
        "\n"
        "Agents collaborate through a shared blackboard.\n"
        "This is a stub pipeline test."
    )

    result: PipelineResult = run_pipeline(input_text, enable_history=True)

    bb = result["blackboard"]
    parsed = result["parsed_chunks"]
    cards = result["explanations"]
    brief = result["public_brief"]

    # Blackboard write-through: keys should be present and consistent.
    assert bb.get("parsed_chunks") == parsed
    assert bb.get("explanations") == cards
    assert bb.get("public_brief") == brief

    # Parser output sanity check.
    assert isinstance(parsed, list)
    assert len(parsed) >= 2
    assert all(isinstance(chunk, str) and chunk for chunk in parsed)

    # Explainer output sanity check.
    assert isinstance(cards, list)
    assert len(cards) >= 1
    assert all(isinstance(card, dict) for card in cards)

    # Public brief structure and meta.
    assert brief["kind"].startswith("public_brief.")
    assert brief["meta"]["num_chunks"] == len(parsed)
    assert brief["meta"]["num_cards"] == len(cards)
    assert brief["meta"]["enable_history"] is True
    # Input preview should be a non-empty string, truncated if long.
    assert isinstance(brief["meta"]["input_preview"], str)
    assert brief["meta"]["input_preview"] != ""


def test_run_pipeline_disable_history_flag() -> None:
    """When history is disabled, the brief meta flag reflects that choice."""
    result: PipelineResult = run_pipeline(
        "Short text for a quick pipeline run.",
        enable_history=False,
    )
    brief = result["public_brief"]
    assert brief["meta"]["enable_history"] is False


def test_pipeline_records_final_trace() -> None:
    """Pipeline should record at least one trace snapshot on the blackboard."""
    result: PipelineResult = run_pipeline("Trace test text.", enable_history=False)
    bb = result["blackboard"]

    snaps = bb.traces()
    assert len(snaps) >= 1

    # The last snapshot should correspond to the pipeline completion note.
    last = snaps[-1]
    note: str | None = last.note
    assert note is not None
    assert "pipeline" in note
