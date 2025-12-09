# scripts/smoke.py
"""
Smoke Test Script for InterLines Pipeline.

Usage
-----
Execute from the project root using `uv`:
    $ uv run python scripts/smoke.py

Purpose
-------
This script performs a direct, synchronous execution of the `run_pipeline` function.
It serves as a "Smoke Test" to verify:
1.  **Environment Configuration**: Are API keys loaded and valid?
2.  **LLM Connectivity**: Can we successfully call OpenAI/DeepSeek providers?
3.  **Pipeline Logic**: Does the DAG execute from Parse -> Brief?
4.  **Observability**: Are traces and reports correctly written to the blackboard?

Unlike unit tests, this script touches real external APIs (costing tokens).
"""

import logging
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

# Fixed (E402): Imports moved to top. Logic for loading env vars runs after imports,
# which is safe because our modules read env vars at runtime (inside functions),
# not at import time.
from interlines.pipelines.public_translation import run_pipeline

# --------------------------------------------------------------------------- #
# Environment Setup
# --------------------------------------------------------------------------- #
# Explicitly load environment variables.
env_path = Path(".env")
if env_path.exists():
    load_dotenv(env_path)
    print("✅ Loaded .env file")
else:
    print("⚠️  Warning: No .env file found! Agents may fail due to missing keys.")

# Configure Logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# --------------------------------------------------------------------------- #
# Test Data
# --------------------------------------------------------------------------- #
# A complex text sample ("Distributed Confidence") that challenges the planner
# and explainer, increasing the likelihood of interesting traces.
# Fixed (W291): Removed trailing whitespace from the multi-line string.
TEST_TEXT = """
The concept of 'Distributed Confidence' in AI safety suggests that reliance
should not be placed on a single component but spread across multiple
heterogeneous systems. Instead of treating confidence as a fixed trait of
one model, it sees confidence as a social signal.

In our everyday decisions — in teams, in families, in classrooms — we rarely
trust people because they all say the same thing. We trust them because we
can see their differences, and how those differences are coordinated.
"""


def main() -> None:
    """Execute the smoke test workflow."""
    print("\n🔥 Starting Smoke Test (Direct Pipeline Mode)...\n")
    print(f"Input Length: {len(TEST_TEXT)} chars")

    # ----------------------------------------------------------------------- #
    # Execution Phase
    # ----------------------------------------------------------------------- #
    try:
        print("... Invoking run_pipeline() ...")
        # We enable `use_llm_planner` to test the real routing logic.
        result = run_pipeline(
            input_text=TEST_TEXT,
            enable_history=False,
            use_llm_planner=True,
        )
    except Exception as exc:
        print(f"\n❌ Pipeline Crashed: {exc}")
        traceback.print_exc()
        return

    # ----------------------------------------------------------------------- #
    # Inspection Phase
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 60)
    print("✅ Pipeline Finished Successfully!")
    print("=" * 60)

    # 1. Inspect the Public Brief (The final product)
    brief = result["public_brief"]
    title = brief.get("title", "No Title")
    summary = brief.get("summary", "")

    print(f"\n📝 Title: {title}")
    print(f"📄 Summary: {str(summary)[:200]}...")

    print("\n📌 Sections:")
    sections = brief.get("sections", [])

    for sec in sections:
        # Access fields via .get() since it's a dict
        heading = str(sec.get("heading", "Untitled"))

        # Cast bullets to list for len()
        bullets_raw = sec.get("bullets", [])
        bullets = bullets_raw if isinstance(bullets_raw, list) else []

        print(f"  - [{heading}]: {len(bullets)} bullets")

    # 2. Inspect the Trace Log (Observability)
    bb = result["blackboard"]
    print("\n🕵️  Trace Log (Key Events):")
    for i, snap in enumerate(bb.traces()):
        if snap.note:
            print(f"  {i+1}. {snap.note}")

    # 3. Inspect Planner Decisions
    report = bb.get("planner_report")
    if report:
        # Fixed (MyPy): Removed unnecessary type ignore.
        r_dict = report if isinstance(report, dict) else report.model_dump()

        print("\n🧠 Planner Report:")
        print(f"  - Strategy: {r_dict.get('strategy')}")
        print(f"  - Replan Used: {r_dict.get('refine_used')}")

        if r_dict.get("refine_used"):
            print(f"  - Reason: {r_dict.get('replan_reason')}")
            print(f"  - Steps: {r_dict.get('replan_steps')}")

    # 4. Output Artifact Location
    path = result.get("public_brief_md_path")
    print(f"\n💾 Artifact saved to: {path}")


if __name__ == "__main__":
    main()
