"""Minimal smoke tests to keep CI meaningful at Step 0.1.

These tests ensure the package imports correctly and that the CLI entry point
returns a zero exit code. They serve as a sanity check for the CI pipeline.
"""

from __future__ import annotations

import importlib


def test_package_imports() -> None:
    """Ensure the package can be imported and exposes __version__."""
    mod = importlib.import_module("interlines")
    assert hasattr(mod, "__version__")


def test_cli_main_returns_zero() -> None:
    """Ensure the CLI entry point returns 0 (success) in the stub phase."""
    cli = importlib.import_module("interlines.cli")
    assert cli.main([]) == 0
