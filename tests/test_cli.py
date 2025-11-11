"""Tests for the InterLines command-line interface (CLI).

These tests call `interlines.cli.main(argv)` directly to avoid spawning a
subprocess, which keeps them fast and platform-consistent in CI.

What we verify
--------------
1) `version` prints the installed package version and returns exit code 0.
2) `env --json` returns a stable JSON object with required keys and values
   reflecting environment overrides after cache clear.
3) `doctor` returns exit code 0 when the development environment is set up
   (Python >= 3.11, core libs importable, essential env vars present).

Notes
-----
- We annotate pytest fixtures as `Any` to satisfy `mypy --strict` without adding
  extra typing stubs for pytest. Test functions themselves are annotated with
  `-> None` per project typing rules.
"""

from __future__ import annotations

import json
from typing import Any

from interlines import __version__ as PKG_VERSION
from interlines.cli import main


def test_cli_version_prints_version(capsys: Any) -> None:
    """`interlines version` should print the package version and exit 0."""
    rc = main(["version"])
    captured = capsys.readouterr().out.strip()
    assert rc == 0
    assert captured == f"InterLines {PKG_VERSION}"


def test_cli_env_json_reflects_env_overrides(monkeypatch: Any, capsys: Any) -> None:
    """`interlines env --json` should emit overrides from environment variables.

    We set `INTERLINES_ENV=test` and `LOG_LEVEL=DEBUG`, then assert the JSON payload
    includes those values and the canonical keys.
    """
    monkeypatch.setenv("INTERLINES_ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    # OPENAI_API_KEY is optional; set to ensure deterministic output shape
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-xxx")

    rc = main(["env", "--json"])
    out = capsys.readouterr().out.strip()

    assert rc == 0
    payload = json.loads(out)
    assert isinstance(payload, dict)
    # Required keys and values
    assert payload["environment"] == "test"
    assert payload["log_level"] == "DEBUG"
    # We expose the presence of key via JSON; value should be non-null here
    assert payload["openai_key"] is not None


def test_cli_doctor_exits_zero(monkeypatch: Any) -> None:
    """`interlines doctor` should succeed in a dev/test environment.

    We ensure minimal env presence and expect a zero exit status. Detailed text
    output is not asserted because it may vary slightly across platforms.
    """
    monkeypatch.setenv("INTERLINES_ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    rc = main(["doctor"])
    assert rc == 0
