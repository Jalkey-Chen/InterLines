# src/interlines/cli.py
"""
InterLines Command Line Interface (CLI).

Milestone
---------
M6 | Interface & Deployment
Step 6.3 | Trace Replay

This module implements the user-facing terminal interface using `typer` and `rich`.
It replaces the legacy argparse-based scaffold with a modern, production-ready CLI.

Features
--------
- **Status Spinners**: Visual feedback during long-running AI tasks.
- **Rich Rendering**: Displays the generated brief nicely in the terminal using Markdown.
- **Trace Replay**: Record execution runs to JSON and replay them later for debugging.
- **Flight Recorder**: Automatically saves every run's state to `artifacts/runs/`.

Usage
-----
    # Run the pipeline (Record Mode)
    $ interlines interpret samples/paper.pdf --model gpt-4o

    # Replay a past run (Replay Mode)
    $ interlines replay artifacts/runs/20251209_xxxx_paper.json
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm

from interlines.core.blackboard.memory import Blackboard
from interlines.core.blackboard.trace import TraceSnapshot
from interlines.pipelines.public_translation import PipelineResult, run_pipeline

# Ensure env vars (like OPENAI_API_KEY) are loaded before any logic runs
load_dotenv()

# Initialize Typer app and Rich console
app = typer.Typer(
    help="InterLines: Turn complex papers into accessible public briefs.",
    rich_markup_mode="markdown",
)
console = Console()


# --------------------------------------------------------------------------- #
# Helpers: Rendering & I/O
# --------------------------------------------------------------------------- #


def _render_brief(result: PipelineResult) -> None:
    """
    Helper: Render the structured brief to the console using Rich Markdown.

    This function is used by both `interpret` (live) and `replay` (cached)
    commands to ensure consistent presentation.
    """
    brief_payload = result["public_brief"]
    title = brief_payload.get("title", "Untitled Brief")
    summary = brief_payload.get("summary", "")

    console.rule(f"[bold]{title!s}[/bold]")
    console.print(Markdown(str(summary)))
    console.print("\n")

    for section in brief_payload.get("sections", []):
        # TypedDict handling: explicit casts/checks for Pylance safety
        heading = str(section.get("heading", "Untitled Section"))
        raw_bullets = section.get("bullets", [])
        bullets = raw_bullets if isinstance(raw_bullets, list) else []

        console.print(f"[bold yellow]## {heading}[/bold yellow]")
        for bullet in bullets:
            console.print(f" • {bullet}")
        console.print("")


def _handle_file_export(result: PipelineResult, output_path: Path | None) -> None:
    """Helper: Handle copying the artifact to a user-specified location."""
    generated_path = result.get("public_brief_md_path")
    final_path = generated_path

    # If user requested a specific output location, copy it there
    if output_path and generated_path:
        try:
            shutil.copy(generated_path, output_path)
            final_path = str(output_path)
            console.print(f"[dim]Copied artifact to: {output_path}[/dim]")
        except OSError as e:
            console.print(f"[bold red]⚠️ Failed to save to {output_path}: {e}[/bold red]")

    if final_path:
        console.print(
            Panel(
                f"Saved to: [link=file://{final_path}]{final_path}[/link]",
                title="Artifact",
                border_style="green",
            )
        )


def _inspect_trace(result: PipelineResult) -> None:
    """
    Helper: Interactively show the execution trace and planner report.

    This provides visibility into the "Black Box", showing which agents ran
    and why (Planner decisions).
    """
    if not Confirm.ask("Show execution trace log?", default=False):
        return

    bb = result["blackboard"]
    console.print("\n[bold dim]Execution Trace:[/bold dim]")
    for i, snap in enumerate(bb.traces()):
        if snap.note:
            console.print(f" [dim]{i+1:02d}. {snap.note}[/dim]")

    report = bb.get("planner_report")
    if report:
        console.print("\n[bold dim]Planner Decisions:[/bold dim]")
        # Robust way to handle dict vs Pydantic model for report object
        r_data: dict[str, Any] = report.model_dump() if hasattr(report, "model_dump") else report
        strategy = r_data.get("strategy", "unknown")
        replan = r_data.get("refine_used", False)
        console.print(f" Strategy: [cyan]{strategy}[/cyan]")
        console.print(f" Replan Used: [magenta]{replan}[/magenta]")


# --------------------------------------------------------------------------- #
# Helpers: State Management (Record & Replay)
# --------------------------------------------------------------------------- #


def _save_run_state(result: PipelineResult, source_file: Path) -> Path:
    """
    Helper: Serialize the full pipeline result to a JSON file for replay.

    This acts as a "Flight Recorder", saving the Blackboard state and final
    artifacts so they can be inspected later without re-running costly LLMs.
    """
    runs_dir = Path("artifacts/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean filename (replace spaces with underscores)
    safe_name = source_file.stem.replace(" ", "_")
    save_path = runs_dir / f"{timestamp}_{safe_name}.json"

    # Serialize essentials for replay.
    # We construct a dictionary that mimics the structure of PipelineResult.
    payload = {
        "meta": {
            "source": str(source_file),
            "timestamp": timestamp,
        },
        "public_brief": result["public_brief"],
        "public_brief_md_path": str(result.get("public_brief_md_path", "")),
        # Serialize blackboard traces (convert TraceSnapshot objects to dicts)
        "traces": [
            {"note": t.note, "data": t.data, "timestamp": t.timestamp}
            for t in result["blackboard"].traces()
        ],
        # Save planner report if it exists
        "planner_report": result["blackboard"].get("planner_report"),
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    return save_path


def _load_run_state(json_path: Path) -> PipelineResult:
    """
    Helper: Hydrate a PipelineResult from a JSON run file.

    This reverses `_save_run_state`, reconstructing the Blackboard object
    so that `_render_brief` and `_inspect_trace` can work on cached data.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # 1. Reconstruct Blackboard
    bb = Blackboard()

    # Rehydrate traces (Blackboard._traces is internal list, we append manually)
    if "traces" in data:
        for t_dict in data["traces"]:
            # We construct TraceSnapshot objects so bb.traces() works
            snap = TraceSnapshot(
                timestamp=t_dict.get("timestamp", ""),
                note=t_dict.get("note"),
                data=t_dict.get("data", {}),
            )
            bb._traces.append(snap)

    # Rehydrate planner report for inspection
    if "planner_report" in data:
        bb.put("planner_report", data["planner_report"])

    # 2. Reconstruct Result Dict
    # Note: We don't restore everything (like parsed_chunks) unless needed for future features.
    # For replay, brief + trace is enough.
    return {
        "blackboard": bb,
        "public_brief": data.get("public_brief", {}),
        "public_brief_md_path": data.get("public_brief_md_path"),
        # Fill required TypedDict keys with empty defaults to satisfy type checker
        "parsed_chunks": [],
        "explanations": [],
        "relevance_notes": [],
        "terms": [],
        "timeline_events": [],
    }


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


# Fixed (MyPy): Untyped decorator workaround
@app.command()  # type: ignore[misc]
def interpret(
    file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the input document (PDF, DOCX, TXT).",
        ),
    ],
    history: Annotated[
        bool,
        typer.Option(
            "--history/--no-history",
            "-H",
            help="Enable Timeline/History analysis (slower but deeper).",
        ),
    ] = False,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Override the default Planner model alias (e.g. 'planner').",
        ),
    ] = "planner",
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Custom path to save the Markdown report (e.g. 'report.md').",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show full error tracebacks for debugging.",
        ),
    ] = False,
) -> None:
    """
    Run the InterLines pipeline on a document (Record Mode).

    This command ingests a file, sends it through the Multi-Agent system,
    produces a brief, AND saves the execution state for future replay.
    """
    # 1. Welcome Banner
    console.print(
        Panel.fit(
            f"[bold cyan]InterLines CLI[/bold cyan]\nProcessing: [u]{file.name}[/u]",
            border_style="cyan",
        )
    )

    start_time = time.time()
    result: PipelineResult | None = None

    # 2. Pipeline Execution (with Spinner)
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Initializing agents...", total=None)
            time.sleep(0.5)

            progress.update(task, description=f"[yellow]Planner ({model}) is thinking...")

            # --- THE CORE CALL ---
            result = run_pipeline(
                input_data=file,
                enable_history=history,
                use_llm_planner=True,
            )

            progress.update(task, description="[green]Finalizing artifacts...")
            time.sleep(0.5)

    except Exception as e:
        console.print(f"\n[bold red]❌ Pipeline Error:[/bold red] {e}")
        if verbose:
            traceback.print_exc()
        raise typer.Exit(code=1) from e

    duration = time.time() - start_time
    console.print(f"\n[bold green]✅ Complete![/bold green] (took {duration:.1f}s)\n")

    if result:
        # 3.1 Render
        _render_brief(result)

        # 3.2 Save State (Record)
        try:
            saved_json = _save_run_state(result, file)
            console.print(f"[dim]Run state saved to: {saved_json}[/dim]")
        except Exception as e:
            console.print(f"[dim yellow]Warning: Could not save run state: {e}[/dim]")

        # 3.3 Export
        _handle_file_export(result, output)

        # 3.4 Inspect
        _inspect_trace(result)


# Fixed (MyPy): Untyped decorator workaround
@app.command()  # type: ignore[misc]
def replay(
    run_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the JSON run artifact (in artifacts/runs/).",
        ),
    ],
) -> None:
    """
    Replay a past execution trace without re-running the AI models.

    This command loads a saved JSON state file (generated by `interpret`)
    and re-renders the brief and execution trace instantly.
    """
    console.print(
        Panel.fit(
            f"[bold magenta]InterLines Replay[/bold magenta]\nLoading: [u]{run_file.name}[/u]",
            border_style="magenta",
        )
    )

    try:
        # 1. Hydrate state from disk
        result = _load_run_state(run_file)

        # 2. Render exactly as if it just finished
        _render_brief(result)
        _inspect_trace(result)

    except Exception as e:
        console.print(f"\n[bold red]❌ Replay Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
