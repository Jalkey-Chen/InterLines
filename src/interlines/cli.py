# src/interlines/cli.py
# -----------------------------------------------------------------------------
# Step 0.4 • Commit 1/3
# Commit: feat(cli): add argparse-based CLI with `version`, `env`, and `doctor`
# -----------------------------------------------------------------------------
"""InterLines command-line interface.

This CLI intentionally uses only the Python standard library (argparse, json,
importlib) so it remains dependency-light. It exposes three subcommands:

Subcommands
-----------
- `version` : Print the installed InterLines version.
- `env`     : Show effective configuration (INTERLINES_ENV, LOG_LEVEL, etc.).
              Use `--json` to emit machine-readable JSON.
- `doctor`  : Run a series of environment checks (Python version, key imports,
              and essential env variables). Exits with code 0 on success, 1 on
              any failure. Use `--verbose` for more details.

Examples
--------
$ interlines version
InterLines 0.0.1

$ interlines env
environment = dev
log_level   = INFO
openai_key  = <unset>

$ interlines env --json
{"environment":"dev","log_level":"INFO","openai_key":null}

$ interlines doctor
✓ Python >= 3.11
✓ pydantic present
✓ pydantic_settings present
✓ fastapi present
✓ INTERLINES_ENV set (dev)
✓ LOG_LEVEL set (INFO)

Design notes
------------
- `main(argv)` returns an exit status (int) to make tests straightforward.
- We log only minimal information; richer logging should use the application
  loggers created via `get_logger()` in runtime code paths.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from typing import Any

from interlines import __version__
from interlines.core.settings import Settings, get_logger, load_settings


# ----- Helpers ----------------------------------------------------------------
def _json_dumps(payload: dict[str, Any]) -> str:
    """Serialize a dict to compact JSON with stable key order.

    Parameters
    ----------
    payload : dict
        Mapping to serialize.

    Returns
    -------
    str
        Compact JSON string with sorted keys.
    """
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _check_import(modname: str) -> bool:
    """Return True if a module can be imported (discovered by importlib)."""
    return importlib.util.find_spec(modname) is not None


# ----- Subcommand implementations ---------------------------------------------
def cmd_version(_: argparse.Namespace) -> int:
    """Print the installed InterLines version."""
    print(f"InterLines {__version__}")
    return 0


def cmd_env(ns: argparse.Namespace) -> int:
    """Print effective configuration, optionally as JSON."""
    # Refresh settings to reflect any env changes the caller might have set.
    load_settings.cache_clear()
    s: Settings = load_settings()

    data: dict[str, Any] = {
        "environment": s.environment,
        "log_level": s.log_level,
        "openai_key": os.environ.get("OPENAI_API_KEY") or None,
    }

    if ns.json:
        print(_json_dumps(data))
    else:
        # Tab-aligned human-friendly printout.
        env = data["environment"]
        lvl = data["log_level"]
        key = "<unset>" if data["openai_key"] is None else "<set>"
        print(f"environment = {env}")
        print(f"log_level   = {lvl}")
        print(f"openai_key  = {key}")
    return 0


def cmd_doctor(ns: argparse.Namespace) -> int:
    """Run environment checks; return 0 if all checks pass, else 1.

    Checks
    ------
    - Python version >= 3.11
    - Core libs importable: pydantic, pydantic_settings
    - API libs importable (recommended): fastapi
    - Essential env vars present (or resolvable via defaults):
        INTERLINES_ENV, LOG_LEVEL
    """
    log = get_logger("interlines.cli.doctor")
    ok: bool = True

    def good(msg: str) -> None:
        print(f"✓ {msg}")

    def bad(msg: str) -> None:
        nonlocal ok
        ok = False
        print(f"✗ {msg}")

    # Python version
    py_ok = sys.version_info >= (3, 11)
    (good if py_ok else bad)(f"Python >= 3.11 (detected {platform.python_version()})")

    # Imports
    for mod in ("pydantic", "pydantic_settings", "fastapi"):
        present = _check_import(mod)
        (good if present else bad)(f"{mod} present")

    # Settings / env vars
    load_settings.cache_clear()
    s = load_settings()
    if s.environment:
        good(f"INTERLINES_ENV set ({s.environment})")
    else:
        bad("INTERLINES_ENV missing")

    if s.log_level:
        good(f"LOG_LEVEL set ({s.log_level})")
    else:
        bad("LOG_LEVEL missing")

    if ns.verbose:
        log.info("doctor: settings=%s", {"env": s.environment, "level": s.log_level})

    return 0 if ok else 1


# ----- Parser construction -----------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser and subparsers.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with subcommands: version, env, doctor.
    """
    parser = argparse.ArgumentParser(
        prog="interlines",
        description="InterLines CLI — utilities for local development and diagnostics.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # version
    p_ver = sub.add_parser("version", help="Print the installed InterLines version.")
    p_ver.set_defaults(func=cmd_version)

    # env
    p_env = sub.add_parser("env", help="Show effective InterLines configuration.")
    p_env.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-friendly text.",
    )
    p_env.set_defaults(func=cmd_env)

    # doctor
    p_doc = sub.add_parser("doctor", help="Run environment checks.")
    p_doc.add_argument("--verbose", action="store_true", help="Print additional details.")
    p_doc.set_defaults(func=cmd_doctor)

    return parser


# ----- Entry point -------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `interlines`.

    Parameters
    ----------
    argv : Optional[list[str]]
        Argument vector; if None, defaults to `sys.argv[1:]`.

    Returns
    -------
    int
        Exit status code (0 success, non-zero on errors).
    """
    parser = _build_parser()
    ns = parser.parse_args(sys.argv[1:] if argv is None else argv)
    func = getattr(ns, "func", None)
    if func is None:
        parser.print_help()
        return 2
    return int(func(ns))


if __name__ == "__main__":
    raise SystemExit(main())
