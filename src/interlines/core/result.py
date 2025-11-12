"""Lightweight, typed Result container for explicit success/failure returns.

Motivation
----------
For the upcoming planner/agent pipeline we prefer explicit error handling over
exceptions in inner loops. This module provides a minimal `Result[T, E]` with:
- `Ok(value)` / `Err(error)` variants,
- combinators: `map`, `map_err`, `flat_map`, `or_else`,
- ergonomic helpers: `unwrap`, `expect`, `unwrap_err`, `get_or`.

Design goals
------------
- Tiny and dependency-free; friendly to `mypy --strict`.
- Side-effect free methods with straightforward semantics.

Example
-------
>>> from interlines.core.result import ok, err, Result
>>> def parse_int(x: str) -> Result[int, str]:
...     return ok(int(x)) if x.isdigit() else err("not a digit")
>>> ok("42").flat_map(parse_int).unwrap()
42
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, NoReturn, TypeVar, cast, overload

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")
F = TypeVar("F")


class Result(Generic[T, E]):
    """Sum type representing either success (`Ok[T]`) or failure (`Err[E]`)."""

    # ----- Introspection -----------------------------------------------------
    def is_ok(self) -> bool:
        """Return ``True`` if this is an :class:`Ok` value."""
        return isinstance(self, Ok)

    def is_err(self) -> bool:
        """Return ``True`` if this is an :class:`Err` value."""
        return isinstance(self, Err)

    # ----- Unwraps -----------------------------------------------------------
    @overload
    def unwrap(self) -> T: ...
    @overload
    def unwrap(self, default: T) -> T: ...

    def unwrap(self, default: T | None = None) -> T:
        """Return the inner value if ``Ok``, else raise or return ``default``.

        Parameters
        ----------
        default:
            Optional fallback value to return when this is ``Err``. If omitted,
            a :class:`RuntimeError` is raised on ``Err``.
        """
        if isinstance(self, Ok):
            # Cast to narrow `self` so mypy knows `.value` is `T`
            return cast(Ok[T, E], self).value
        if default is not None:
            return default
        raise RuntimeError(f"Attempted to unwrap Err: {self!r}")

    def expect(self, msg: str) -> T:
        """Return the inner value if ``Ok``, else raise ``RuntimeError(msg)``."""
        if isinstance(self, Ok):
            return cast(Ok[T, E], self).value
        raise RuntimeError(msg)

    def unwrap_err(self) -> E:
        """Return the error value if ``Err``, else raise."""
        if isinstance(self, Err):
            return cast(Err[T, E], self).error
        raise RuntimeError(f"Attempted to unwrap_err on Ok: {self!r}")

    # ----- Combinators -------------------------------------------------------
    def map(self, fn: Callable[[T], U]) -> Result[U, E]:
        """Apply ``fn`` to the success value; propagate error unchanged."""
        if isinstance(self, Ok):
            return Ok(fn(cast(Ok[T, E], self).value))
        # mypy: safe cast from Result[T,E] to Result[U,E] when self is Err
        return cast(Result[U, E], self)

    def map_err(self, fn: Callable[[E], F]) -> Result[T, F]:
        """Apply ``fn`` to the error value; propagate success unchanged."""
        if isinstance(self, Err):
            return Err(fn(cast(Err[T, E], self).error))
        return cast(Result[T, F], self)

    def flat_map(self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Chain computations that already return a :class:`Result`."""
        if isinstance(self, Ok):
            return fn(cast(Ok[T, E], self).value)
        return cast(Result[U, E], self)

    # ----- Utilities ---------------------------------------------------------
    def or_else(self, fallback: Callable[[E], Result[T, E]]) -> Result[T, E]:
        """If ``Err``, call ``fallback(err)``; otherwise return ``self``."""
        if isinstance(self, Err):
            return fallback(cast(Err[T, E], self).error)
        return self

    def get_or(self, default: T) -> T:
        """Return the success value or a default if ``Err``."""
        return self.unwrap(default)

    # ----- Dunder helpers ----------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - trivial representation
        if isinstance(self, Ok):
            return f"Ok({cast(Ok[T, E], self).value!r})"
        if isinstance(self, Err):
            return f"Err({cast(Err[T, E], self).error!r})"
        return "Result(?)"


@dataclass(frozen=True)
class Ok(Result[T, E]):
    """Successful result wrapping a value of type ``T``."""

    value: T


@dataclass(frozen=True)
class Err(Result[T, E]):
    """Failed result wrapping an error payload of type ``E``."""

    error: E


# ----- Convenience constructors ----------------------------------------------
def ok(value: T) -> Result[T, E]:
    """Construct :class:`Ok` with better type inference at call sites."""
    return Ok(value)


def err(error: E) -> Result[T, E]:
    """Construct :class:`Err` with better type inference at call sites."""
    return Err(error)


def never(msg: str) -> NoReturn:
    """Raise a ``RuntimeError(msg)`` to mark a non-returning code path."""
    raise RuntimeError(msg)
