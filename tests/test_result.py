"""Unit tests for the lightweight Result utilities."""

from __future__ import annotations

from interlines.core.result import Err, Result, err, ok


def test_ok_map_and_flat_map() -> None:
    """`Ok` should map/flat_map and keep values typed."""
    r: Result[int, str] = ok(10)
    r2 = r.map(lambda x: x + 5).flat_map(lambda x: ok(x * 2))
    assert r2.is_ok() and r2.unwrap() == 30


def test_err_propagation_and_map_err() -> None:
    """`Err` should propagate through map/flat_map and allow mapping the error."""
    r: Result[int, str] = err("boom")
    assert r.is_err()
    assert r.map(lambda x: x + 1).is_err()
    r2 = r.map_err(lambda e: f"{e}!")
    assert isinstance(r2, Err) and r2.unwrap_err() == "boom!"


def test_unwrap_variants_and_defaults() -> None:
    """Unwrap behavior: default value and explicit error raising."""
    assert ok("x").unwrap() == "x"
    assert err("e").unwrap(default="fallback") == "fallback"


def test_or_else_invocation() -> None:
    """`or_else` should call the fallback only on Err."""
    called: dict[str, bool] = {"hit": False}

    def fb(_: str) -> Result[int, str]:
        called["hit"] = True
        return ok(7)

    out = err("nope").or_else(fb)
    assert out.is_ok() and out.unwrap() == 7 and called["hit"] is True
