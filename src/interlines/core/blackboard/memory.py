"""In-memory Blackboard with typed get/put and trace snapshots (in-memory).

This module implements a minimal, dependency-free blackboard that stores
arbitrary Python values under string keys. It provides:

- `put(key, value)`: insert or update an entry and bump the revision counter.
- `get(key, default=None)`: retrieve a value with an optional default.
- `trace(note=None)`: capture a *snapshot dict* (metadata + shallow, JSON-safe
  view of current state) and keep it in memory for later inspection.

In Step 1.2 Commit 2/3, we will add a disk-backed trace writer so that
`snapshot` objects can be persisted under `artifacts/trace/`.

Design goals
------------
- Keep the public API tiny and easy to mock.
- Make types mypy-friendly; avoid returning `Any` unannotated.
- Avoid side effects in `trace` (no file IO in this commit).

Example
-------
>>> bb = Blackboard()
>>> bb.put("k", 42)
>>> bb.get("k")
42
>>> snap = bb.trace("first")
>>> sorted(snap["keys"])
['k']
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeVar, cast

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Immutable record of a blackboard snapshot.

    Attributes
    ----------
    created_at : datetime
        UTC timestamp when the snapshot was taken.
    revision : int
        Current blackboard revision at snapshot time (increments on mutation).
    note : Optional[str]
        Optional human-readable note (e.g., 'after planner step 1').
    keys : Tuple[str, ...]
        Sorted tuple of keys present in the blackboard.
    data : Dict[str, Any]
        JSON-safe shallow copy of the blackboard content.
    """

    created_at: datetime
    revision: int
    note: str | None
    keys: tuple[str, ...]
    data: dict[str, Any]


def _jsonify(value: Any) -> Any:
    """Return a JSON-safe representation of `value` (shallow).

    - Primitives (None, bool, int, float, str) are returned as-is.
    - dict -> dict (keys coerced to str)
    - list/tuple -> list
    - objects -> `repr(obj)`.

    This avoids importing Pydantic here; callers can pre-convert models.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            out[str(k)] = _jsonify(v)
        return out
    if isinstance(value, list | tuple):
        return [_jsonify(v) for v in value]
    # Fallback: stable textual representation
    return repr(value)


class Blackboard:
    """Simple in-memory key-value store with revisioned trace snapshots.

    Notes
    -----
    - Revisions bump on every `put`.
    - `trace()` captures a shallow, JSON-safe view and stores it internally.
    - File persistence is added in Commit 2/3.
    """

    __slots__ = ("_store", "_rev", "_snaps")

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._rev: int = 0
        self._snaps: list[Snapshot] = []

    # ------------------------------- KV API ---------------------------------
    def put(self, key: str, value: Any) -> None:
        """Insert or update `key` with `value` and bump the revision counter."""
        self._store[key] = value
        self._rev += 1

    def get(self, key: str, default: T | None = None) -> T | None:
        """Return the stored value for `key`, or `default` if not found."""
        if key in self._store:
            return cast(T | None, self._store[key])
        return default

    def keys(self) -> tuple[str, ...]:
        """Return the current keys as a sorted tuple (stable for tests)."""
        return tuple(sorted(self._store.keys()))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._store)

    # ------------------------------- Trace API ------------------------------
    def trace(self, note: str | None = None) -> Snapshot:
        """Capture an immutable snapshot of the current blackboard state.

        Parameters
        ----------
        note : Optional[str]
            Optional human-readable label for the snapshot.

        Returns
        -------
        Snapshot
            A dataclass containing metadata and a shallow, JSON-safe copy.
        """
        data_copy: dict[str, Any] = {k: _jsonify(v) for k, v in self._store.items()}
        snap = Snapshot(
            created_at=datetime.now(UTC),
            revision=self._rev,
            note=note,
            keys=tuple(sorted(self._store.keys())),
            data=data_copy,
        )
        self._snaps.append(snap)
        return snap

    def traces(self) -> tuple[Snapshot, ...]:
        """Return all recorded snapshots (immutable tuple)."""
        return tuple(self._snaps)


__all__ = ["Blackboard", "Snapshot"]
