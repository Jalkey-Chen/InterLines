"""Disk-backed trace writer for blackboard snapshots.

This module adds a small utility to persist `Snapshot` objects as JSON files.

- Default directory: `INTERLINES_TRACE_DIR` env var or `artifacts/trace/`
- Filename pattern:  `YYYYmmddTHHMMSSmmmZ_rev{rev:06d}.json`
- Content:           a JSON object mirroring the `Snapshot` dataclass

Usage
-----
>>> writer = TraceWriter()  # uses default dir
>>> path = writer.write(snap)  # returns the file path
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import UTC
from pathlib import Path

from .memory import Snapshot


def _default_dir() -> Path:
    root = os.getenv("INTERLINES_TRACE_DIR")
    return Path(root) if root else Path("artifacts") / "trace"


class TraceWriter:
    """Persist blackboard snapshots to disk as JSON files."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir: Path = base_dir if base_dir is not None else _default_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write(self, snap: Snapshot) -> Path:
        """Write `snap` to disk and return the created file path."""
        ts = snap.created_at.astimezone(UTC)
        stamp = ts.strftime("%Y%m%dT%H%M%S%f")[:-3] + "Z"  # millisecond precision
        filename = f"{stamp}_rev{snap.revision:06d}.json"
        path = self.base_dir / filename

        payload = asdict(snap)
        # Ensure ISO-8601 formatting for datetime fields
        payload["created_at"] = ts.isoformat()

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return path
