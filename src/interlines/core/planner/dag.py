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

    def __init__(self, *, strategy: str) -> None:
        self.strategy: str = strategy
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, set[str]] = {}

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

    # ----------------------------------------------------------------------
    # Topological sort (Kahn's algorithm)
    # ----------------------------------------------------------------------

    def topological_order(self) -> tuple[str, ...]:
        """Return a valid topological ordering of node IDs."""
        indeg = {n: 0 for n in self.nodes}

        for _src, targets in self.edges.items():
            for t in targets:
                indeg[t] += 1

        queue = [n for n, d in indeg.items() if d == 0]
        out: list[str] = []

        while queue:
            n = queue.pop(0)
            out.append(n)
            for t in self.edges.get(n, ()):
                indeg[t] -= 1
                if indeg[t] == 0:
                    queue.append(t)

        if len(out) != len(self.nodes):
            raise ValueError("Graph has a cycle or disconnected components.")

        return tuple(out)

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
