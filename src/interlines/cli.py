"""CLI placeholders for Step 0.1.

The CLI exposes a `main()` that returns success (0) so CI can import, type-check,
and run a smoke test without failing. Real subcommands will be added in later steps.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Entry point for `interlines` console script.

    Parameters
    ----------
    argv : list[str] | None
        Optional argument vector. If None, uses sys.argv[1:].

    Returns
    -------
    int
        Process exit code. Always 0 in Step 0.1 to keep CI green.
    """
    _ = argv if argv is not None else sys.argv[1:]
    # Intentionally no behavior yet.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
