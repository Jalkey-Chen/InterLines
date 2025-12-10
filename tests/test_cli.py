"""
Tests for the InterLines command-line interface (CLI).

Milestone
---------
M6 | Interface & Deployment
Step 6.3 | Trace Replay

Scope
-----
These tests verify the interaction layer provided by Typer:
1.  **Command Registration**: Ensuring `--help` works and shows correct usage
    for subcommands (`interpret`, `replay`).
2.  **Argument Validation**: Typer's `exists=True` checks for input files.
3.  **Pipeline Integration**: Mocking the core `run_pipeline` to ensure arguments
    are passed correctly from the CLI to the backend.
4.  **Error Handling**: Verifying graceful exit codes on failures.

Test Strategy
-------------
We use `typer.testing.CliRunner` to invoke the app in-process, avoiding the overhead
of spawning subprocesses. This keeps tests fast and platform-consistent.

Updates
-------
- **Fix**: Added "interpret" subcommand back to invocations.
  Since we added `replay`, the CLI is now a multi-command app, so subcommands
  are mandatory again (e.g. `interlines interpret file.pdf`).
- **Docs**: Restored detailed docstrings and context notes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from interlines.cli import app


# Fixed (MyPy): Added type ignore to silence "Untyped decorator" error.
# This prevents MyPy from complaining that the fixture makes the function untyped.
@pytest.fixture  # type: ignore[misc]
def runner() -> CliRunner:
    """
    Create a fresh CliRunner for each test.

    This ensures that the internal state of Click/Typer logic is isolated
    between tests, preventing 'Exit Code 2' (Usage Error) false positives
    that can occur when reusing a global runner.
    """
    return CliRunner()


def test_cli_help_shows_usage(runner: CliRunner) -> None:
    """Invoking --help should print usage instructions and exit 0."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, f"Help failed: {result.output}"
    assert "InterLines" in result.output
    # Now that we have subcommands (interpret, replay), help shows "Commands"
    assert "Commands" in result.output
    assert "interpret" in result.output
    assert "replay" in result.output


def test_interpret_fails_on_missing_file(runner: CliRunner) -> None:
    """Typer should enforce `exists=True` for the input file argument."""
    # We provide a path that definitely doesn't exist.
    # Updated: Calling WITH 'interpret' subcommand (Multi-Command Mode).
    result = runner.invoke(app, ["interpret", "ghost.pdf"])

    # Typer returns code 2 for usage/validation errors.
    assert result.exit_code != 0
    # FIX: Validation errors go to stderr, so we must check .output (mixed)
    # instead of .stdout, otherwise this assertion fails on an empty string.
    assert "does not exist" in result.output


def test_interpret_happy_path(runner: CliRunner, tmp_path: Path) -> None:
    """
    Verify the happy path where the pipeline runs successfully.

    We mock `run_pipeline` to avoid actual LLM calls and return a
    valid result structure to ensure the rendering logic (Rich) works.
    """
    # 1. Create a dummy file so Typer validation passes
    dummy_file = tmp_path / "paper.pdf"
    dummy_file.write_text("dummy content", encoding="utf-8")

    # 2. Mock the pipeline result (TypedDict structure)
    # Define complex mocks first to satisfy type checkers and prevent
    # "object has no attribute" errors in MyPy when accessing nested items.
    mock_bb = MagicMock()
    # Mock traces to return empty list to avoid iteration errors in CLI inspector
    mock_bb.traces.return_value = []
    # Mock get() to return None (simulating no planner report present)
    mock_bb.get.return_value = None

    mock_result = {
        "blackboard": mock_bb,
        "public_brief": {
            "title": "Mock Brief",
            "summary": "This is a summary.",
            "sections": [
                {
                    "heading": "Key Findings",
                    "bullets": ["Point A", "Point B"],
                }
            ],
        },
        "public_brief_md_path": "/tmp/mock_output.md",
        "parsed_chunks": [],
        "explanations": [],
        "relevance_notes": [],
        "terms": [],
        "timeline_events": [],
    }

    # 3. Patch and Run
    # We patch the core pipeline entrypoint to inject our mock result.
    with patch("interlines.cli.run_pipeline", return_value=mock_result) as mock_run:
        # Note: We pass "n" to the "Show execution trace log?" prompt to skip it.
        # The input="n\n" simulates the user pressing 'n' then Enter.
        # Updated: Calling WITH 'interpret' subcommand (Multi-Command Mode).
        result = runner.invoke(app, ["interpret", str(dummy_file)], input="n\n")

        # 4. Assertions
        # CRITICAL: We print result.output if this assertion fails!
        # This helps debug "Exit Code 2" errors by showing the actual Typer error message.
        assert result.exit_code == 0, f"CLI Failed with Output:\n{result.output}"

        assert "Complete!" in result.output
        assert "Mock Brief" in result.output  # Verify Rich rendering of title
        assert "Key Findings" in result.output  # Verify Rich rendering of sections

        # Verify arguments passed to core logic
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs

        # We passed a path string to CLI, Typer converts to Path, Pipeline gets Path
        assert isinstance(call_kwargs["input_data"], Path)
        assert call_kwargs["input_data"].name == "paper.pdf"
        # Verify default options
        assert call_kwargs["enable_history"] is False
        assert call_kwargs["use_llm_planner"] is True


def test_interpret_handles_pipeline_crash(runner: CliRunner, tmp_path: Path) -> None:
    """Ensure exceptions in the pipeline are caught and displayed nicely."""
    dummy_file = tmp_path / "paper.pdf"
    dummy_file.write_text("dummy")

    with patch("interlines.cli.run_pipeline") as mock_run:
        # Simulate a crash deep in the system (e.g. API quota exceeded)
        mock_run.side_effect = RuntimeError("LLM Out of credits")

        # Updated: Calling WITH 'interpret' subcommand (Multi-Command Mode).
        result = runner.invoke(app, ["interpret", str(dummy_file)])

        # We expect exit code 1 (Application Error), NOT 2 (Usage Error).
        # 2 = Bad arguments, 1 = Runtime exception handled by our try/except block.
        assert (
            result.exit_code == 1
        ), f"Expected crash (1), got {result.exit_code}. Output:\n{result.output}"

        # Verify the user sees a friendly error message, not just a raw traceback.
        assert "Pipeline Error" in result.output
        assert "LLM Out of credits" in result.output
