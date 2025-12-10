# tests/test_cli.py
"""
Tests for the InterLines command-line interface (CLI).

Scope
-----
These tests verify the interaction layer provided by Typer:
1.  **Command Registration**: Ensuring `interpret` and `--help` work.
2.  **Argument Validation**: Typer's `exists=True` checks for input files.
3.  **Pipeline Integration**: Mocking the core `run_pipeline` to ensure arguments
    are passed correctly from the CLI to the backend.
4.  **Error Handling**: Verifying graceful exit codes on failures.

We use `typer.testing.CliRunner` to invoke the app in-process, avoiding the overhead
of spawning subprocesses.

Updates
-------
- Fixed assertions to check `result.output` (combined stdout/stderr) instead of
  `result.stdout`, as Typer writes validation errors to stderr.
- Added `runner` fixture to ensure test isolation (avoiding exit code 2 errors).
- Added debug output to assertions to print CLI output on failure.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from interlines.cli import app


@pytest.fixture  # type: ignore[misc]
def runner() -> CliRunner:
    """
    Create a fresh CliRunner for each test.

    This ensures that the internal state of Click/Typer logic is isolated
    between tests, preventing 'Exit Code 2' (Usage Error) false positives.
    """
    return CliRunner()


def test_cli_help_shows_usage(runner: CliRunner) -> None:
    """Invoking --help should print usage instructions and exit 0."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, f"Help failed: {result.output}"
    assert "InterLines" in result.output
    assert "interpret" in result.output


def test_interpret_fails_on_missing_file(runner: CliRunner) -> None:
    """Typer should enforce `exists=True` for the input file argument."""
    # We provide a path that definitely doesn't exist.
    result = runner.invoke(app, ["interpret", "ghost.pdf"])

    # Typer returns code 2 for usage/validation errors.
    assert result.exit_code != 0
    # FIX: Validation errors go to stderr, so we must check .output (mixed)
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
    # "object has no attribute" errors in MyPy.
    mock_bb = MagicMock()
    # Mock traces to return empty list to avoid iteration errors in CLI inspector
    mock_bb.traces.return_value = []
    # Mock get() to return None (simulating no planner report)
    mock_bb.get.return_value = None

    mock_result = {
        "blackboard": mock_bb,
        "public_brief": {
            "title": "Mock Brief",
            "summary": "This is a summary.",
            "sections": [{"heading": "Key Findings", "bullets": ["Point A", "Point B"]}],
        },
        "public_brief_md_path": "/tmp/mock_output.md",
        "parsed_chunks": [],
        "explanations": [],
        "relevance_notes": [],
        "terms": [],
        "timeline_events": [],
    }

    # 3. Patch and Run
    with patch("interlines.cli.run_pipeline", return_value=mock_result) as mock_run:
        # Note: We pass "n" to the "Show execution trace log?" prompt to skip it
        # The input="n\n" simulates the user pressing 'n' then Enter.
        result = runner.invoke(app, ["interpret", str(dummy_file)], input="n\n")

        # 4. Assertions
        # CRITICAL: We print result.output if this assertion fails!
        assert result.exit_code == 0, f"CLI Failed with Output:\n{result.output}"

        assert "Complete!" in result.output
        assert "Mock Brief" in result.output  # Verify rendering
        assert "Key Findings" in result.output

        # Verify arguments passed to core logic
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs

        # We passed a path string to CLI, Typer converts to Path, Pipeline gets Path
        assert isinstance(call_kwargs["input_data"], Path)
        assert call_kwargs["input_data"].name == "paper.pdf"
        # Defaults
        assert call_kwargs["enable_history"] is False
        assert call_kwargs["use_llm_planner"] is True


def test_interpret_handles_pipeline_crash(runner: CliRunner, tmp_path: Path) -> None:
    """Ensure exceptions in the pipeline are caught and displayed nicely."""
    dummy_file = tmp_path / "paper.pdf"
    dummy_file.write_text("dummy")

    with patch("interlines.cli.run_pipeline") as mock_run:
        # Simulate a crash deep in the system
        mock_run.side_effect = RuntimeError("LLM Out of credits")

        result = runner.invoke(app, ["interpret", str(dummy_file)])

        # We expect exit code 1 (Application Error), NOT 2 (Usage Error)
        # 2 = Bad arguments, 1 = Runtime exception handled by our try/except block
        assert (
            result.exit_code == 1
        ), f"Expected crash (1), got {result.exit_code}. Output:\n{result.output}"
        assert "Pipeline Error" in result.output
        assert "LLM Out of credits" in result.output
