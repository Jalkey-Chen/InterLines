# src/interlines/cli.py
"""
InterLines Command Line Interface (CLI).

Milestone
---------
M6 | Interface & Deployment
Step 6.2 | CLI Experience

This module implements the user-facing terminal interface using `typer` and `rich`.

Updates
-------
- Refactored rendering logic into helper functions to reduce Cyclomatic Complexity (C901).
- Added proper exception chaining (B904).
- Added type ignore for Typer decorator (MyPy).
"""

from __future__ import annotations

import shutil
import time
import traceback
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm

from interlines.pipelines.public_translation import PipelineResult, run_pipeline

# Initialize Typer app and Rich console
app = typer.Typer(
    help="InterLines: Turn complex papers into accessible public briefs.",
    rich_markup_mode="markdown",
)
console = Console()


def _render_brief(result: PipelineResult) -> None:
    """Helper: Render the structured brief to the console."""
    brief_payload = result["public_brief"]
    title = brief_payload.get("title", "Untitled Brief")
    summary = brief_payload.get("summary", "")

    console.rule(f"[bold]{title!s}[/bold]")
    console.print(Markdown(str(summary)))
    console.print("\n")

    for section in brief_payload.get("sections", []):
        heading = str(section.get("heading", "Untitled Section"))
        raw_bullets = section.get("bullets", [])
        # Ensure it's a list for iteration safety
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
    """Helper: Interactively show the execution trace and planner report."""
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
        # Robust way to handle dict vs Pydantic model
        r_data: dict[str, Any] = report.model_dump() if hasattr(report, "model_dump") else report
        strategy = r_data.get("strategy", "unknown")
        replan = r_data.get("refine_used", False)
        console.print(f" Strategy: [cyan]{strategy}[/cyan]")
        console.print(f" Replan Used: [magenta]{replan}[/magenta]")


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
    Run the InterLines pipeline on a document.

    This command ingests a file, sends it through the Multi-Agent system,
    and produces a structured public brief.
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

            # We assume input_data handles Path objects via ParserAgent
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
        # Fixed (B904): Use 'from e' to preserve exception chain
        raise typer.Exit(code=1) from e

    duration = time.time() - start_time
    console.print(f"\n[bold green]✅ Complete![/bold green] (took {duration:.1f}s)\n")

    if result:
        # Delegate logic to helper functions (Reducing Complexity C901)
        _render_brief(result)
        _handle_file_export(result, output)
        _inspect_trace(result)


if __name__ == "__main__":
    app()
