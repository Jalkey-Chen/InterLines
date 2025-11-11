"""Centralized application configuration using Pydantic Settings (v2).

This module exposes a single, cached `settings` instance that reads from:
- Real environment variables (highest precedence)
- `.env` files at the repository root: .env, .env.local, .env.dev/.env.test/.env.prod
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EnvName = Literal["dev", "test", "prod"]
LogLevelName = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Typed application configuration loaded from env and `.env` files.

    Attributes
    ----------
    environment : EnvName
        Runtime environment flag; maps from `INTERLINES_ENV`.
    log_level : LogLevelName
        Global log level string; maps from `LOG_LEVEL`.
    openai_api_key : Optional[str]
        Optional API key used in later milestones (LLM client). Maps from `OPENAI_API_KEY`.
    """

    environment: EnvName = Field(default="dev", alias="INTERLINES_ENV")
    log_level: LogLevelName = Field(default="INFO", alias="LOG_LEVEL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local", ".env.dev", ".env.test", ".env.prod"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def is_dev(self) -> bool:
        """Return True if running in the development environment."""
        return self.environment == "dev"

    @property
    def is_test(self) -> bool:
        """Return True if running in the test environment."""
        return self.environment == "test"

    @property
    def is_prod(self) -> bool:
        """Return True if running in the production environment."""
        return self.environment == "prod"

    def log_level_numeric(self) -> int:
        """Return the numeric logging level corresponding to `self.log_level`."""
        return getattr(logging, self.log_level, logging.INFO)


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Create and cache a `Settings` instance.

    We keep this behind an LRU cache so tests can force a rebuild via
    `load_settings.cache_clear()` after mutating `os.environ`.
    """
    os.environ.setdefault("INTERLINES_ENV", "dev")
    return Settings()


# Export a ready-to-use singleton (import-time read of env / .env files).
settings: Settings = load_settings()


def get_logger(name: str = "interlines") -> logging.Logger:
    """Return a process-global logger configured to `settings.log_level`."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(settings.log_level_numeric())
    logger.propagate = False
    return logger
