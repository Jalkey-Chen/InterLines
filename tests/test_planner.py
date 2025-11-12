"""Planner DAG tests: strategy-dependent path and trace serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from interlines.core.blackboard.storage import TraceWriter
from interlines.core.planner.strategy import build_plan, expected_path


def test_public_only_path() -> None:
    """When history is disabled, the timeline step is omitted."""
    dag = build_plan(enable_history=False)
    assert dag.strategy == "public_only"
    assert dag.topo_order() == expected_path(False)


def test_with_history_path() -> None:
    """When history is enabled, the timeline step is included."""
    dag = build_plan(enable_history=True)
    assert dag.strategy == "with_history"
    assert dag.topo_order() == expected_path(True)


def test_dag_serializes_to_trace(tmp_path: Path, monkeypatch: Any) -> None:
    """DAG snapshot can be written to artifacts/trace and read back."""
    outdir = tmp_path / "trace"
    monkeypatch.setenv("INTERLINES_TRACE_DIR", str(outdir))

    dag = build_plan(enable_history=True)
    snap = dag.to_snapshot(note="planner.dag")

    writer = TraceWriter()  # picks up env var
    path = writer.write(snap)
    assert path.exists()

    payload: dict[str, Any]
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["note"] == "planner.dag"
    assert "created_at" in payload and payload["created_at"].endswith("Z")
    assert "dag" in payload["data"]
    dag_json = payload["data"]["dag"]
    assert dag_json["strategy"] == "with_history"
    assert dag_json["topo"] == list(expected_path(True))
    assert {n["id"] for n in dag_json["nodes"]} >= set(expected_path(True))
