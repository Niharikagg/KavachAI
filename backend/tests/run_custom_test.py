"""
run_custom_test.py
------------------
Terminal runner for the KavachAI privacy pipeline.

Usage (from the project root):
    python backend/tests/run_custom_test.py
    python -m backend.tests.run_custom_test

Edit ``backend/tests/test_input.json`` to test a different sentence —
only the ``"content"`` field inside ``messages`` needs to change.

The runner calls the REAL production pipeline (process_conversation).
No M1/M2/M3 logic is duplicated here.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that package imports resolve when
# this script is run directly (python backend/tests/run_custom_test.py).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.modules.pipeline import process_conversation  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIVIDER = "=" * 60
_THIN    = "-" * 60


def _section(title: str) -> None:
    print(f"\n{_DIVIDER}")
    print(title)
    print(_THIN)


def _print_m1(m1: dict) -> None:
    _section("M1 — EXTRACTED ATTRIBUTES")
    attrs = m1.get("attributes", [])
    if not attrs:
        print("  (no attributes detected)")
    else:
        for i, attr in enumerate(attrs, 1):
            print(f"\n  [{i}] TYPE        : {attr.get('type', '—')}")
            print(f"       VALUE       : {attr.get('value', '—')}")
            print(f"       CONFIDENCE  : {attr.get('confidence', '—')}")
            print(f"       SPECIFICITY : {attr.get('specificity', '—')}")

    indicators = m1.get("contextual_indicators", [])
    print(f"\n  CONTEXTUAL INDICATORS ({len(indicators)}):")
    if not indicators:
        print("    (none)")
    else:
        for ind in indicators:
            print(f"    • {ind.get('indicator', '—')}  [{ind.get('attribute_type', '—')}]")


def _print_m2(m2: dict) -> None:
    _section("M2 — RISK ANALYSIS")
    print(f"\n  RISK SCORE  : {m2.get('risk_score', '—')}")
    print(f"  RISK LEVEL  : {m2.get('risk_level', '—')}")
    features = m2.get("features", {})
    if features:
        print("\n  FEATURES:")
        for name, val in features.items():
            print(f"    {name:<42} : {val}")


def _print_m3(m3: dict) -> None:
    _section("M3 — SANITIZED OUTPUT")
    print(f"\n  ORIGINAL TEXT   : {m3.get('original_text', '—')}")
    print(f"  SANITIZED TEXT  : {m3.get('sanitized_text', '—')}")

    transformations = m3.get("transformations", [])
    print(f"\n  TRANSFORMATIONS ({len(transformations)}):")
    if not transformations:
        print("    (none)")
    else:
        for t in transformations:
            print(
                f"    • [{t.get('attribute', '?')}]"
                f"  \"{t.get('original', '?')}\""
                f"  →  \"{t.get('new_value', '?')}\""
                f"  (cost {t.get('info_loss_cost', '?')})"
            )

    utility = m3.get("utility", {})
    print(f"\n  INITIAL RISK SCORE  : {m3.get('initial_risk_score', '—')}")
    print(f"  FINAL RISK SCORE    : {m3.get('final_risk_score', '—')}")
    print(f"  INITIAL RISK LEVEL  : {m3.get('initial_risk_level', '—')}")
    print(f"  FINAL RISK LEVEL    : {m3.get('final_risk_level', '—')}")
    print(f"  INFORMATION LOSS    : {utility.get('information_loss', '—')}")
    print(f"  UTILITY RETAINED    : {utility.get('information_retained', '—')}")
    attrs_changed = utility.get("attributes_changed", "—")
    attrs_total   = utility.get("attributes_total", "—")
    print(f"  ATTRIBUTES CHANGED  : {attrs_changed} / {attrs_total}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    # Locate test_input.json relative to this script
    input_file = Path(__file__).parent / "test_input.json"

    if not input_file.exists():
        print(f"ERROR: test input file not found at {input_file}", file=sys.stderr)
        return 1

    try:
        payload = json.loads(input_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: failed to parse {input_file}: {exc}", file=sys.stderr)
        return 1

    conversation_id = payload.get("conversation_id", "custom_test_001")
    messages        = payload.get("messages", [])

    if not messages:
        print("ERROR: 'messages' list is empty in test_input.json", file=sys.stderr)
        return 1

    # Show the raw input
    _section("CUSTOM INPUT")
    raw_content = messages[0].get("content", "")
    print(f"\n  {raw_content}")

    # Run the REAL production pipeline
    try:
        result = process_conversation(conversation_id, messages)
    except Exception:  # noqa: BLE001
        print("\nPIPELINE ERROR — traceback follows:\n", file=sys.stderr)
        traceback.print_exc()
        return 1

    _print_m1(result["m1"])
    _print_m2(result["m2"])
    _print_m3(result["m3"])

    _section("PIPELINE COMPLETE")
    print(f"\n  conversation_id : {result['conversation_id']}")
    print(f"  risk reduced by : {result['m3'].get('risk_reduced', '—')}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
