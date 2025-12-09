"""
ASGI Entry Point for InterLines API.

This module exposes the `app` object required by ASGI servers (Uvicorn/Gunicorn).
It enables running the API via:
    $ uvicorn interlines.api.server:app --reload
Or directly via python:
    $ python -m interlines.api.server
"""

import uvicorn

from interlines.api.app import create_app

# Factory invocation
app = create_app()


def main() -> None:
    """Run the API server locally for development."""
    uvicorn.run(
        "interlines.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
