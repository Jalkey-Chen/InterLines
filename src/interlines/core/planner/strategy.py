"""Planner strategies that generate a minimal DAG for execution.

We support two modes:

- `public_only`  : linear path without historical timeline enrichment.
- `with_history` : linear path that *includes* a `timeline` step.

Both strategies produce a single linear chain so that `topo_order()` is
unambiguous and easy to assert in tests, while leaving room to expand to
branch/merge shapes later.
"""

from __future__ import annotations

from .dag import DAG


def build_plan(enable_history: bool) -> DAG:
    """Return a `DAG` according to `enable_history`.

    Parameters
    ----------
    enable_history : bool
        If True, include the `timeline` step; otherwise omit it.

    Returns
    -------
    DAG
        The constructed graph with a deterministic linear order.
    """
    dag = DAG()
    dag.strategy = "with_history" if enable_history else "public_only"

    # Common linear backbone
    dag.add_node("parse", "Parse source inputs")
    dag.add_node("translate", "Translate to public language")
    if enable_history:
        dag.add_node("timeline", "Build historical timeline / concept drift")
    dag.add_node("narrate", "Compose narrative")
    dag.add_node("review", "Review & QA")
    dag.add_node("brief", "Assemble public brief")

    # Edges (linear path; `timeline` inserted between translate and narrate)
    dag.add_edge("parse", "translate")
    if enable_history:
        dag.add_edge("translate", "timeline")
        dag.add_edge("timeline", "narrate")
    else:
        dag.add_edge("translate", "narrate")
    dag.add_edge("narrate", "review")
    dag.add_edge("review", "brief")

    return dag


def expected_path(enable_history: bool) -> tuple[str, ...]:
    """Helper for tests: the canonical expected topological order."""
    if enable_history:
        return ("parse", "translate", "timeline", "narrate", "review", "brief")
    return ("parse", "translate", "narrate", "review", "brief")


__all__ = ["build_plan", "expected_path"]
