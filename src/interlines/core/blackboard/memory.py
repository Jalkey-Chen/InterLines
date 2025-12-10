"""
In-memory Blackboard with typed get/put and trace snapshots.

Milestone
---------
M1 | Blackboard & State
Step 1.1 | In-memory implementation

This module implements a minimal, dependency-free blackboard that stores
arbitrary Python values under string keys. It provides:

- ``put(key, value)``: insert or update an entry and bump the revision counter.
- ``get(key, default=None)``: retrieve a value with an optional default.
- ``trace(note=None)``: capture a *snapshot* of the current state.

Updates (Step 6.3)
------------------
- Refactored to use :class:`TraceSnapshot` from ``.trace`` module.
- Renamed internal snapshot storage to ``_traces`` to align with CLI expectations.
- Updated ``trace()`` to serialize timestamps to ISO strings immediately upon capture.

Design Goals
------------
- **Minimal API**: Keep the surface area small (`get`/`put`/`trace`).
- **Type Safety**: Use Generics (TypeVar) for `get` return types.
- **Observability**: Every state change increments a revision; every trace
  captures a full view of the data at that revision.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, TypeVar, cast

from .trace import TraceSnapshot

T = TypeVar("T")


def _jsonify(value: Any) -> Any:
    """
    Return a JSON-safe representation of ``value`` (shallow conversion).

    Strategies:
    - Primitives (None, bool, int, float, str) -> returned as-is.
    - dict -> new dict with keys coerced to str.
    - list/tuple -> new list with recursive conversion.
    - objects -> ``repr(obj)`` fallback.

    This avoids importing Pydantic here to keep the core lightweight; callers
    are expected to store Pydantic models as dicts using ``model_dump()`` if
    they want clean JSON serialization.
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
    # Fallback: stable textual representation for complex objects
    return repr(value)


class Blackboard:
    """
    Simple in-memory key-value store with revisioned trace snapshots.

    Attributes
    ----------
    _store : dict[str, Any]
        The actual key-value storage.
    _rev : int
        Monotonically increasing revision counter (bumps on every mutation).
    _traces : list[TraceSnapshot]
        History of captured snapshots.
    """

    # Updated slots to match new internal naming (was _snaps)
    __slots__ = ("_store", "_rev", "_traces")

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._rev: int = 0
        # Renamed from _snaps to _traces to align with CLI expectations
        self._traces: list[TraceSnapshot] = []

    # ------------------------------- KV API ---------------------------------

    def put(self, key: str, value: Any) -> None:
        """
        Insert or update ``key`` with ``value`` and bump the revision counter.

        Parameters
        ----------
        key : str
            The identifier for the artifact (e.g., "planner_plan_spec").
        value : Any
            The artifact data. Should ideally be JSON-serializable or a Pydantic model
            dump to ensure the trace log is readable.
        """
        self._store[key] = value
        self._rev += 1

    def get(self, key: str, default: T | None = None) -> T | None:
        """
        Return the stored value for ``key``, or ``default`` if not found.

        Parameters
        ----------
        key : str
            The key to look up.
        default : T | None
            Value to return if key is missing.

        Returns
        -------
        T | None
            The value cast to the expected type T.
        """
        if key in self._store:
            return cast(T | None, self._store[key])
        return default

    def keys(self) -> tuple[str, ...]:
        """Return the current keys as a sorted tuple (stable for tests)."""
        return tuple(sorted(self._store.keys()))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._store)

    # ------------------------------- Trace API ------------------------------

    def trace(self, note: str | None = None) -> TraceSnapshot:
        """
        Capture an immutable snapshot of the current blackboard state.

        This method creates a shallow copy of the store, sanitizes it for JSON,
        timestamps it, and appends it to the internal trace log.

        Parameters
        ----------
        note : str | None
            Optional human-readable label explaining *why* this trace was taken
            (e.g., 'after planner step 1').

        Returns
        -------
        TraceSnapshot
            A dataclass containing metadata and a shallow, JSON-safe copy of the data.
        """
        # Shallow copy and sanitize data
        data_copy: dict[str, Any] = {k: _jsonify(v) for k, v in self._store.items()}

        # Generate ISO timestamp string immediately to freeze time
        # We use millisecond precision and 'Z' to denote UTC
        ts_str = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        snap = TraceSnapshot(
            timestamp=ts_str,
            note=note,
            data=data_copy,
        )
        self._traces.append(snap)
        return snap

    def traces(self) -> tuple[TraceSnapshot, ...]:
        """
        Return all recorded snapshots (immutable tuple).

        Returns
        -------
        tuple[TraceSnapshot, ...]
            The full history of traces for this session.
        """
        return tuple(self._traces)


__all__ = ["Blackboard"]
