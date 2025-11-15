"""Minimal smoke tests kept from Step 0.1, updated for the Step 0.4 CLI.

What changed
------------
The CLI now requires a subcommand. We therefore call `main(["version"])`
instead of `main([])` and keep the import sanity check for the package.
"""

from __future__ import annotations

import importlib


def test_package_imports() -> None:
    """Ensure the package can be imported and exposes __version__."""
    mod = importlib.import_module("interlines")
    assert hasattr(mod, "__version__")


def test_cli_main_returns_zero() -> None:
    """Ensure the CLI entry point returns 0 when given a valid subcommand."""
    cli = importlib.import_module("interlines.cli")
    assert cli.main(["version"]) == 0
