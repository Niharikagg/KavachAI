"""
run_custom_test.py

Terminal runner for the KavachAI privacy pipeline.

Usage:

    Usage:

     .venv\Scripts\python.exe backend\tests\run_custom_test.py

Edit the root input.json to test a different sentence.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from backend.modules.pipeline import process_conversation  # noqa: E402


# ============================================================
# DISPLAY HELPERS
# ============================================================

_DIVIDER = "=" * 60
_THIN = "-" * 60


def _section(title: str) -> None:
    print(f"\n{_DIVIDER}")
    print(title)
    print(_THIN)


# ============================================================
# M1
# ============================================================

def _print_m1(m1: dict) -> None:

    _section("M1 OUTPUT")

    attrs = m1.get("attributes", [])

    if not attrs:
        print("  (no attributes detected)")
    else:

        for i, attr in enumerate(attrs, 1):

            print(
                f"\n  [{i}] TYPE        : "
                f"{attr.get('type', '—')}"
            )

            print(
                f"       VALUE       : "
                f"{attr.get('value', '—')}"
            )

            print(
                f"       CONFIDENCE  : "
                f"{attr.get('confidence', '—')}"
            )

            print(
                f"       SPECIFICITY : "
                f"{attr.get('specificity', '—')}"
            )

    indicators = m1.get(
        "contextual_indicators",
        [],
    )

    print(
        f"\n  CONTEXTUAL INDICATORS "
        f"({len(indicators)}):"
    )

    if not indicators:

        print("    (none)")

    else:

        for ind in indicators:

            print(
                f"    • "
                f"{ind.get('indicator', '—')}"
                f"  [{ind.get('attribute_type', '—')}]"
            )


# ============================================================
# M2
# ============================================================

def _print_m2(m2: dict) -> None:

    _section("M2 OUTPUT")

    print(
        f"\n  RISK SCORE  : "
        f"{m2.get('risk_score', '—')}"
    )

    print(
        f"  RISK LEVEL  : "
        f"{m2.get('risk_level', '—')}"
    )

    features = m2.get(
        "features",
        {},
    )

    if features:

        print("\n  FEATURES:")

        for name, value in features.items():

            print(
                f"    {name:<42} : "
                f"{value}"
            )


# ============================================================
# M3
# ============================================================

def _print_m3(m3: dict) -> None:

    _section("M3 OUTPUT")

    print(
        f"\n  ORIGINAL TEXT   : "
        f"{m3.get('original_text', '—')}"
    )

    print(
        f"  SANITIZED TEXT  : "
        f"{m3.get('sanitized_text', '—')}"
    )

    transformations = m3.get(
        "transformations",
        [],
    )

    print(
        f"\n  TRANSFORMATIONS "
        f"APPLIED ({len(transformations)}):"
    )

    if not transformations:

        print("    (none)")

    else:

        for t in transformations:

            attribute = t.get(
                "attribute",
                "—",
            )

            original = t.get(
                "original_value",
                "—",
            )

            generalized = t.get(
                "generalized_value",
                "—",
            )

            cost = t.get(
                "cost",
                "—",
            )

            print(
                f"    • [{attribute}]"
                f"  \"{original}\""
                f"  →  \"{generalized}\""
                f"  (cost {cost})"
            )

    print(
        f"\n  INITIAL RISK SCORE  : "
        f"{m3.get('initial_risk_score', '—')}"
    )

    print(
        f"  FINAL RISK SCORE    : "
        f"{m3.get('final_risk_score', '—')}"
    )

    print(
        f"  INITIAL RISK LEVEL  : "
        f"{m3.get('initial_risk_level', '—')}"
    )

    print(
        f"  FINAL RISK LEVEL    : "
        f"{m3.get('final_risk_level', '—')}"
    )

    print(
        f"  INFORMATION LOSS    : "
        f"{m3.get('information_loss', '—')}"
    )

    print(
        f"  UTILITY RETAINED    : "
        f"{m3.get('information_retained', '—')}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # "attributes_changed" in the optimizer currently means
    # number of transformations, NOT unique attributes.
    # --------------------------------------------------------

    print(
        f"  TRANSFORMATIONS     : "
        f"{len(transformations)}"
    )

    print(
        f"  ORIGINAL ATTRIBUTES : "
        f"{m3.get('total_attributes', '—')}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    input_file = (
        _PROJECT_ROOT
        / "input.json"
    )

    if not input_file.exists():

        print(
            f"ERROR: test input file not found "
            f"at {input_file}",
            file=sys.stderr,
        )

        return 1

    try:

        payload = json.loads(
            input_file.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as exc:

        print(
            f"ERROR: failed to parse "
            f"{input_file}: {exc}",
            file=sys.stderr,
        )

        return 1

    conversation_id = payload.get(
        "conversation_id",
        "custom_test_001",
    )

    message = payload.get(
        "message",
        "",
    )

    if not message:

        print(
            "ERROR: 'message' is empty "
            "in input.json",
            file=sys.stderr,
        )

        return 1

    messages = [
        {
            "role": "user",
            "content": message,
        }
    ]

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    _section("CUSTOM INPUT")

    print(f"\n  {message}")

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    try:

        result = process_conversation(
            conversation_id,
            messages,
        )

    except Exception:

        print(
            "\nPIPELINE ERROR — traceback follows:\n",
            file=sys.stderr,
        )

        traceback.print_exc()

        return 1

    # --------------------------------------------------------
    # OUTPUTS
    # --------------------------------------------------------

    _print_m1(
        result["m1"]
    )

    _print_m2(
        result["m2"]
    )

    _print_m3(
        result["m3"]
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    _section("PIPELINE COMPLETE")

    print(
        f"\n  conversation_id : "
        f"{result['conversation_id']}"
    )

    m3 = result.get(
        "m3",
        {},
    )

    risk_reduction = m3.get(
        "risk_reduction",
        "—",
    )

    print(
        f"  risk reduced by : "
        f"{risk_reduction}"
    )

    print()

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )