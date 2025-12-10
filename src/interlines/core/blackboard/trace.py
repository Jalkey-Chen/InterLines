"""
Trace Snapshot definition.

Milestone
---------
M1 | Blackboard & State
Step 1.2 | Trace persistence

This module defines the immutable record of a blackboard state at a specific
point in time. It is separated from ``memory.py`` to avoid circular imports
and to keep the data structure definition clean and reusable by the CLI replay system.

Design Notes
------------
- **Immutability**: Once created, a snapshot should not change. We use ``frozen=True``.
- **Serialization**: We use ``str`` for timestamps here (instead of datetime objects)
  to simplify the JSON serialization logic in the CLI and storage layers. The
  conversion happens at the moment of capture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TraceSnapshot:
    """
    Immutable record of a blackboard snapshot.

    Attributes
    ----------
    timestamp : str
        ISO-8601 formatted timestamp string (e.g., "2023-10-27T10:00:00.123Z").
        Represents the exact UTC time when this snapshot was captured.
    revision : int
        The sequential revision number of the blackboard at capture time.
    note : str | None
        Optional human-readable label (e.g., 'after planner step 1').
        Used for filtering and display in the CLI trace inspector.
    data : dict[str, Any]
        JSON-safe, shallow copy of the blackboard content at that moment.
        This dictionary holds the actual artifacts (e.g., plan specs, drafts).
    """

    timestamp: str
    revision: int
    note: str | None
    data: dict[str, Any] = field(default_factory=dict)
