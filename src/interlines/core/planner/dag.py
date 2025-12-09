"""
DAG structure for orchestrating pipeline agent steps.

Commit 1 (Step 5.1):
- Rename class Dag → DAG (fully uppercase for semantic clarity)
- Introduce DAG.from_plan_spec()
- Preserve legacy behavior so existing tests continue to pass
"""

from __future__ import annotations

from dataclasses import dataclass, field

from interlines.core.contracts.planner import PlannerPlanSpec


@dataclass
class Node:
    """A node in the planner DAG."""

    id: str
    label: str = ""
    meta: dict[str, object] = field(default_factory=dict)


class DAG:
    """
    A minimal, test-friendly directed acyclic graph used by the planner.

    Commit 1 focuses on preserving all existing behavior while preparing for
    DAG-driven pipeline execution in Commit 2.
    """

    # ---------------------------------------------------------
    # NEW: explicit attribute declarations (mypy needs these)
    # ---------------------------------------------------------
    nodes: dict[str, Node]
    edges: dict[str, set[str]]
    rev_edges: dict[str, set[str]]  # <-- added

    def __init__(self, *, strategy: str) -> None:
        self.strategy: str = strategy
        self.nodes = {}
        self.edges = {}
        self.rev_edges = {}  # <-- added

    # ---------------------------------------------------------
    # Helper: rebuild reverse edges
    # ---------------------------------------------------------
    def _recompute_rev_edges(self) -> None:
        """Rebuild reverse adjacency lists.

        This keeps DAG topological sort stable even if edges mutate.
        """
        rev: dict[str, set[str]] = {u: set() for u in self.nodes}
        for u, vs in self.edges.items():
            for v in vs:
                rev.setdefault(v, set()).add(u)
        self.rev_edges = rev

    # ----------------------------------------------------------------------
    # Basic DAG construction
    # ----------------------------------------------------------------------

    def add(self, src: str, dst: str, *, label: str = "") -> None:
        """Add an edge src → dst, creating missing nodes automatically."""
        if src not in self.nodes:
            self.nodes[src] = Node(id=src, label=src)
        if dst not in self.nodes:
            self.nodes[dst] = Node(id=dst, label=dst)

        self.edges.setdefault(src, set()).add(dst)
        self.edges.setdefault(dst, set())  # ensure dst appears as a key

        self._recompute_rev_edges()  # <-- added

    # ----------------------------------------------------------------------
    # Topological sort (Kahn's algorithm)
    # ----------------------------------------------------------------------

    def topological_order(self) -> list[str]:
        """
        Return the nodes in topologically sorted order.

        Notes
        -----
        - A ``list`` is returned (not a ``tuple``) because most planner
          components (including ``plan_spec.steps`` and JSON serialization)
          naturally operate on lists.
        - Returning a list avoids mypy's non-overlap warnings when comparing
          with ``plan_spec.steps`` (which is also a list).
        - Uses a local indegree map; the underlying DAG is never mutated.

        Raises
        ------
        ValueError
            If the DAG contains a cycle.
        """
        # Ensure reverse edges are ready
        if not self.rev_edges:
            self._recompute_rev_edges()

        indeg = {u: len(self.rev_edges.get(u, ())) for u in self.nodes}
        queue = [u for u in self.nodes if indeg[u] == 0]
        out: list[str] = []

        while queue:
            u = queue.pop()
            out.append(u)
            for v in self.edges.get(u, ()):
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)

        if len(out) != len(self.nodes):
            raise ValueError("DAG contains a cycle")

        return out

    # ----------------------------------------------------------------------
    # NEW: Build a DAG directly from PlannerPlanSpec
    # ----------------------------------------------------------------------

    @classmethod
    def from_plan_spec(cls, plan: PlannerPlanSpec) -> DAG:
        """
        Construct a DAG whose topological order exactly matches plan.steps.

        Commit 1: This implementation is intentionally minimal and linear:
        step[i] → step[i+1]
        """
        dag = cls(strategy="from_plan_spec")

        # Create nodes
        for step in plan.steps:
            dag.nodes[step] = Node(id=step, label=step)
            dag.edges.setdefault(step, set())

        # Linear edges
        for a, b in zip(plan.steps, plan.steps[1:], strict=False):
            dag.edges[a].add(b)

        dag._recompute_rev_edges()  # <-- required for topo sort

        return dag

    # ----------------------------------------------------------------------
    # JSON-safe payload used by tests
    # ----------------------------------------------------------------------

    def to_payload(self) -> dict[str, object]:
        """Return JSON-safe structure used by blackboard and tests."""
        return {
            "strategy": self.strategy,
            "nodes": list(self.nodes.keys()),
            "edges": {k: sorted(v) for k, v in self.edges.items()},
            "topo": list(self.topological_order()),
        }


__all__ = ["DAG", "Node"]
