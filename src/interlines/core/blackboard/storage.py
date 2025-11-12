"""Disk-backed trace writer for blackboard snapshots.

This module adds a small utility to persist `Snapshot` objects as JSON files.

- Default directory: `INTERLINES_TRACE_DIR` env var or `artifacts/trace/`
- Filename pattern:  `YYYYmmddTHHMMSSmmmZ_rev{rev:06d}.json`
- Content:           a JSON object mirroring the `Snapshot` dataclass

Timestamp format
----------------
We normalize timestamps to UTC and serialize as ISO-8601 strings with a
trailing `"Z"` and millisecond precision, e.g., `"2025-11-12T02:02:37.104Z"`.

Usage
-----
>>> writer = TraceWriter()  # uses default dir
>>> path = writer.write(snap)  # returns the file path
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .memory import Snapshot


def _default_dir() -> Path:
    """Return the default base directory for trace artifacts."""
    root = os.getenv("INTERLINES_TRACE_DIR")
    return Path(root) if root else Path("artifacts") / "trace"


def _utc_millis_z(dt: datetime) -> str:
    """Format a datetime as ISO-8601 UTC with millisecond precision and 'Z'.

    Examples
    --------
    >>> _utc_millis_z(datetime(2025, 11, 12, 2, 2, 37, 104380, tzinfo=timezone.utc))
    '2025-11-12T02:02:37.104Z'
    """
    ts = dt.astimezone(UTC)
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class TraceWriter:
    """Persist blackboard snapshots to disk as JSON files."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir: Path = base_dir if base_dir is not None else _default_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write(self, snap: Snapshot) -> Path:
        """Write `snap` to disk and return the created file path.

        Notes
        -----
        - The JSON payload mirrors `Snapshot` (with `created_at` normalized).
        - Filenames include a UTC timestamp and the current revision to keep
          ordering stable and readable during debugging.
        """
        ts = snap.created_at.astimezone(UTC)
        stamp = ts.strftime("%Y%m%dT%H%M%S%f")[:-3] + "Z"  # millisecond precision
        filename = f"{stamp}_rev{snap.revision:06d}.json"
        path = self.base_dir / filename

        payload = asdict(snap)
        # Normalize datetime to ISO-8601 UTC with 'Z' suffix (millisecond precision)
        payload["created_at"] = _utc_millis_z(ts)

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return path
