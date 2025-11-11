"""Typed smoke tests for the Step 0.2 settings loader.

What's changed in this commit
-----------------------------
- Remove the direct `import pytest` to avoid mypy complaining about missing
  stubs under `--strict`. Instead, we annotate the `monkeypatch` fixture as
  `Any`, which keeps the test fully typed without introducing a new stub dep.
- Fix the import path to use the package layout: `interlines.core.settings`.

These tests verify three guarantees:
1) Importing the module-level `settings` yields a `Settings` instance.
2) Environment variables override defaults after clearing the loader cache.
3) `get_logger()` respects the configured LOG_LEVEL when constructing loggers.
"""

from __future__ import annotations

import logging
from typing import Any

from interlines.core.settings import (
    Settings,
    get_logger,
    load_settings,
    settings,
)


def test_settings_instance_type() -> None:
    """`settings` should be an instance of the typed `Settings` model."""
    assert isinstance(settings, Settings)


def test_env_overrides_with_cache_clear(monkeypatch: Any) -> None:
    """Changing env vars should take effect after `load_settings.cache_clear()`.

    Steps
    -----
    1) Set INTERLINES_ENV to "test" and LOG_LEVEL to "DEBUG" via `monkeypatch`.
    2) Clear the loader cache so a new Settings instance is constructed.
    3) Assert that the new instance reflects the env overrides.
    """
    monkeypatch.setenv("INTERLINES_ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    load_settings.cache_clear()
    s = load_settings()

    assert s.environment == "test"
    assert s.log_level == "DEBUG"


def test_get_logger_respects_level(monkeypatch: Any) -> None:
    """`get_logger()` should apply the numeric level derived from `LOG_LEVEL`.

    We set LOG_LEVEL=ERROR and expect the created logger's level to be logging.ERROR.
    A unique logger name avoids side effects between tests.
    """
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    load_settings.cache_clear()
    _ = load_settings()

    logger_name = "interlines.tests.settings"
    logger = get_logger(logger_name)

    assert logger.level == logging.ERROR
    # Sanity: the handler exists and uses our simple formatter.
    assert logger.handlers, "Expected at least one StreamHandler to be attached."
