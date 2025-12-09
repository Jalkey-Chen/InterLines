# scripts/smoke.py
"""
Smoke Test Script for InterLines Pipeline.

Usage
-----
1. Test with default hardcoded text:
    $ uv run python scripts/smoke.py

2. Test with a local file (PDF/DOCX) - The Pipeline handles extraction internally!
    $ uv run python scripts/smoke.py --file samples/paper.pdf

Dependencies
------------
Ensure optional extractor dependencies are installed:
    $ uv add pdfplumber python-docx
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

# Note: We import run_pipeline which now accepts `str | Path`
from interlines.pipelines.public_translation import run_pipeline

# --------------------------------------------------------------------------- #
# Environment Setup
# --------------------------------------------------------------------------- #
env_path = Path(".env")
if env_path.exists():
    load_dotenv(env_path)
    print("✅ Loaded .env file")
else:
    print("⚠️  Warning: No .env file found! Agents may fail due to missing keys.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# --------------------------------------------------------------------------- #
# Test Data
# --------------------------------------------------------------------------- #
DEFAULT_TEXT = """
The concept of 'Distributed Confidence' in AI safety suggests that reliance
should not be placed on a single component but spread across multiple
heterogeneous systems. Instead of treating confidence as a fixed trait of
one model, it sees confidence as a social signal.
"""


def main() -> None:
    """Execute the smoke test workflow."""
    parser = argparse.ArgumentParser(description="Run InterLines Smoke Test")
    parser.add_argument("--file", "-f", type=str, help="Path to input file (.pdf, .docx, .txt)")
    args = parser.parse_args()

    # 1. Prepare Input Data
    # We construct a Path object if a file is provided, otherwise use string.
    input_data: str | Path
    if args.file:
        input_path = Path(args.file)
        if not input_path.exists():
            print(f"❌ File not found: {input_path}")
            return
        print(f"\n📂 Using input file: {input_path}")
        input_data = input_path
    else:
        print("\n📝 Using default test text (No --file provided)")
        input_data = DEFAULT_TEXT

    # 2. Execution Phase
    try:
        print("... Invoking run_pipeline() ...")
        # The pipeline now accepts Path objects directly!
        result = run_pipeline(
            input_data=input_data,
            enable_history=False,
            use_llm_planner=True,
        )
    except Exception as exc:
        print(f"\n❌ Pipeline Crashed: {exc}")
        traceback.print_exc()
        return

    # 3. Inspection Phase
    print("\n" + "=" * 60)
    print("✅ Pipeline Finished Successfully!")
    print("=" * 60)

    # Inspect Brief
    brief = result["public_brief"]
    title = brief.get("title", "No Title")
    summary = brief.get("summary", "")

    print(f"\n📝 Title: {title}")
    # Cast for safety in case summary is None (though typed as str)
    print(f"📄 Summary: {str(summary)[:300]}...")

    print("\n📌 Sections:")
    sections = brief.get("sections", [])
    for sec in sections:
        # Strict dict access
        heading = str(sec.get("heading", "Untitled"))
        bullets_raw = sec.get("bullets", [])
        bullets = bullets_raw if isinstance(bullets_raw, list) else []
        print(f"  - [{heading}]: {len(bullets)} bullets")

    # Inspect Trace
    bb = result["blackboard"]
    print("\n🕵️  Trace Log (Key Events):")
    for i, snap in enumerate(bb.traces()):
        if snap.note:
            print(f"  {i+1}. {snap.note}")

    # Inspect Planner Report
    report = bb.get("planner_report")
    if report:
        r_dict = report if isinstance(report, dict) else report.model_dump()
        print("\n🧠 Planner Report:")
        print(f"  - Strategy: {r_dict.get('strategy')}")
        print(f"  - Replan Used: {r_dict.get('refine_used')}")
        if r_dict.get("refine_used"):
            print(f"  - Reason: {r_dict.get('replan_reason')}")

    # Output Location
    path = result.get("public_brief_md_path")
    print(f"\n💾 Artifact saved to: {path}")


if __name__ == "__main__":
    main()
