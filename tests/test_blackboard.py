"""Unit tests for the in-memory blackboard and disk trace writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from interlines.core.blackboard.memory import Blackboard
from interlines.core.blackboard.storage import TraceWriter


def test_put_get_basic() -> None:
    """Basic put/get behavior and key listing are correct."""
    bb = Blackboard()
    bb.put("a", 1)
    bb.put("b", {"x": 2})
    assert bb.get("a") == 1
    assert bb.get("missing", default=None) is None
    assert set(bb.keys()) == {"a", "b"}
    assert len(bb) == 2


def test_trace_in_memory_snapshot() -> None:
    """`trace()` returns a consistent, JSON-safe snapshot."""
    bb = Blackboard()
    bb.put("k", {"v": 1})
    snap = bb.trace("first")
    assert snap.revision == 1
    assert "k" in snap.data and snap.data["k"] == {"v": 1}
    # snapshots are accumulated
    assert len(bb.traces()) == 1


def test_trace_writes_files(tmp_path: Path, monkeypatch: Any) -> None:
    """Snapshots can be written to disk under the configured directory."""
    outdir = tmp_path / "trace"
    monkeypatch.setenv("INTERLINES_TRACE_DIR", str(outdir))

    bb = Blackboard()
    bb.put("title", "PKI")
    s1 = bb.trace("init")

    writer = TraceWriter()  # picks up env var for base_dir
    p1 = writer.write(s1)
    assert p1.exists()

    # Inspect JSON payload
    payload: dict[str, Any]
    with p1.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["revision"] == 1
    assert payload["note"] == "init"
    assert payload["data"]["title"] == "PKI"
    assert "created_at" in payload and payload["created_at"].endswith("Z")

    # A second snapshot produces a second file
    bb.put("x", 2)
    s2 = bb.trace("second")
    p2 = writer.write(s2)
    assert p2.exists() and p2 != p1

    # Clean up env var to avoid leaking across tests
    monkeypatch.delenv("INTERLINES_TRACE_DIR", raising=False)
