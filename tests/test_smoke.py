"""
Smoke tests for package structure and availability.

Milestone
---------
M0 | Setup & Architecture

Scope
-----
These tests strictly verify that the package is installed correctly in the
environment and that top-level modules are importable.
"""

from __future__ import annotations

import importlib

from interlines import __version__


def test_package_importable() -> None:
    """Ensure the top-level package can be imported."""
    mod = importlib.import_module("interlines")
    assert mod is not None


def test_version_is_set() -> None:
    """Ensure the package exposes a valid version string."""
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_cli_module_exposes_app() -> None:
    """
    Ensure the CLI module exposes the Typer 'app' object.

    This replaces the old test that checked for a 'main' function.
    The presence of 'app' is required for the entry point defined in
    pyproject.toml (`interlines.cli:app`).
    """
    cli = importlib.import_module("interlines.cli")
    assert hasattr(cli, "app"), "interlines.cli must expose an 'app' Typer object."
