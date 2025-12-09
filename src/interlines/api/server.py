"""
ASGI Entry Point for InterLines API.

This module exposes the `app` object required by ASGI servers (Uvicorn/Gunicorn).
It proactively loads environment variables from `.env` to ensure that configuration
is available before the application factory or any module-level logic runs.

Usage
-----
Run via the module entry point:
    $ uv run python -m interlines.api.server

Or via uvicorn directly:
    $ uv run uvicorn interlines.api.server:app --reload
"""

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from interlines.api.app import create_app

# --------------------------------------------------------------------------- #
# Environment Setup
# --------------------------------------------------------------------------- #

# Load environment variables from .env BEFORE importing the application factory.
# This ensures that global singletons (like JobStore) or settings modules
# that read os.environ at import time can initialize correctly.
env_path = Path(".env")
load_dotenv(dotenv_path=env_path)

# Factory invocation
app = create_app()


def main() -> None:
    """Run the API server locally for development."""
    # Debug: Verify that all critical API keys are loaded.
    # This helps confirm that python-dotenv successfully found the file and
    # that the developer has populated the necessary secrets.
    print("[Server] Loading environment variables from .env...")

    expected_keys = [
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "MOONSHOT_API_KEY",
        "DEEPSEEK_API_KEY",
        "ZHIPU_API_KEY",
        "XAI_API_KEY",
    ]

    print(f"{'[ Key Check ]':=^60}")
    for var_name in expected_keys:
        value = os.getenv(var_name, "")
        if value:
            # Mask the key for security, showing only the first 8 chars
            masked = f"{value[:8]}..."
            print(f"{var_name:<20} : ✅ Loaded ({masked})")
        else:
            print(f"{var_name:<20} : ❌ Missing")
    print(f"{'='*60}\n")

    uvicorn.run(
        "interlines.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
