"""Minimal DAG model and serializer for the InterLines planner.

This module provides a tiny, serializable, *directed acyclic graph* (DAG)
abstraction used by the planning strategy layer.

Key features
------------
- `DAG.add_node(id, label)`, `DAG.add_edge(u, v)` with acyclicity check.
- `DAG.topo_order()` returns a deterministic topological ordering.
- `DAG.to_payload()` returns a JSON-safe dictionary (no `Any` leaks).
- `DAG.to_snapshot(note)` converts the current DAG into a `Snapshot` that can
  be written to `artifacts/trace/` using `TraceWriter`.

Design notes
------------
- The DAG is intentionally small and dependency-free to keep planning logic
  transparent and testable.
- We keep node ids stable (snake_case) and store a human-readable `label`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from interlines.core.blackboard.memory import Snapshot


@dataclass(frozen=True, slots=True)
class Node:
    """A graph node with a stable `id` and a human-readable `label`."""

    id: str
    label: str


@dataclass(slots=True)
class DAG:
    """A tiny directed acyclic graph with serialization helpers."""

    # Public fields
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)
    strategy: str = "public_only"  # e.g., "public_only" | "with_history"

    # ------------------------------- Build API --------------------------------
    def add_node(self, node_id: str, label: str) -> None:
        """Add a node (idempotent on id)."""
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id, label)

    def add_edge(self, u: str, v: str) -> None:
        """Add a directed edge `u -> v` and assert no cycles are introduced."""
        if (u, v) in self.edges:
            return
        if u not in self.nodes or v not in self.nodes:
            raise KeyError("Both endpoints must be added before adding an edge.")
        # Tentatively add and verify acyclicity via Kahn topo
        self.edges.append((u, v))
        try:
            _ = self.topo_order()
        except ValueError:
            # revert and re-raise
            self.edges.pop()
            raise

    # ------------------------------- Query API --------------------------------
    def topo_order(self) -> tuple[str, ...]:
        """Return a deterministic topological ordering of node ids.

        Raises
        ------
        ValueError
            If a cycle is detected (i.e., not all nodes can be ordered).
        """
        indeg: dict[str, int] = {nid: 0 for nid in self.nodes}
        for _, v in self.edges:
            indeg[v] += 1

        # Stable queue: process by lexicographic id to keep order deterministic
        ready: list[str] = sorted([n for n, d in indeg.items() if d == 0])
        order: list[str] = []
        # Build adjacency once for speed
        adj: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for u, v in self.edges:
            adj[u].append(v)
        for lst in adj.values():
            lst.sort()

        while ready:
            u = ready.pop(0)
            order.append(u)
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    # maintain lexicographic order
                    ready.append(v)
                    ready.sort()

        if len(order) != len(self.nodes):
            raise ValueError("Cycle detected in DAG.")
        return tuple(order)

    # ------------------------------ Serialization -----------------------------
    def to_payload(self) -> dict[str, object]:
        """Return a JSON-safe dictionary describing this DAG.

        The payload is designed to be embedded into a blackboard `Snapshot.data`.
        """
        node_list: list[dict[str, str]] = [
            {"id": n.id, "label": n.label} for n in self.nodes.values()
        ]
        edge_list: list[dict[str, str]] = [{"src": u, "dst": v} for (u, v) in self.edges]
        return {
            "strategy": self.strategy,
            "nodes": node_list,
            "edges": edge_list,
            "topo": list(self.topo_order()),
        }

    def to_snapshot(self, note: str = "planner") -> Snapshot:
        """Convert the DAG into a `Snapshot` ready for trace writing.

        Notes
        -----
        - `revision` is set to `0` here (not tied to the blackboard mut counter).
        - `keys` is the ordered tuple of node ids for quick visual inspection.
        """
        payload = self.to_payload()
        return Snapshot(
            created_at=_now_utc(),
            revision=0,
            note=note,
            keys=self.topo_order(),
            data={"dag": payload},
        )


# ------------------------------ small helpers --------------------------------
def _now_utc() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(UTC)


__all__ = ["Node", "DAG"]
