"""
m1_m2_adapter.py — Thin adapter connecting Module 1 (NLP) to Module 2 (Risk).

Responsibility:
  1. Validate and convert M1 output JSON into M2 conversation format.
  2. Map M1 indicator names to M2's expected indicator taxonomy.
  3. Invoke M2 build_features() and score_risk() without modifying M2 logic.
"""

from __future__ import annotations

from typing import Any

from backend.modules.risk.features import build_features
from backend.modules.risk.scorer import score_risk

# Mapping between M1 contextual indicator names and M2 indicator weight keys.
# - M1 derives 'small_location' when a small village/town/city near a location is detected.
#   M2 weights this under 'village_level_location' (weight = 0.6).
# - M1 derives 'rare_condition' when a rare health condition is detected.
#   M2 weights this under 'sensitive_health_attribute' (weight = 0.2).
# - 'exact_age' (0.3), 'exact_date' (0.15), and 'uniqueness_word' (1.0) match directly.
M1_TO_M2_INDICATOR_MAP = {
    "small_location": "village_level_location",
    "rare_condition": "sensitive_health_attribute",
}


def adapt_m1_to_m2(m1_output: dict[str, Any]) -> dict[str, Any]:
    """
    Convert an M1 output dictionary into the input structure expected by M2.

    M1 Public Contract:
    {
        "conversation_id": str,
        "original_text": str,
        "attributes": [
            {
                "type": str,
                "value": str,
                "confidence": float,
                "specificity": float,
                "source_message_id": int
            }
        ],
        "contextual_indicators": [
            {
                "indicator": str,
                "attribute_type": str
            }
        ]
    }

    M2 Expects:
    {
        "conversation_id": str,
        "attributes": list[dict],      # consumed: attribute['type'], attribute['specificity']
        "contextual_indicators": list[dict] # consumed: indicator['indicator']
    }

    Note:
      - 'confidence' from M1 is preserved on attributes, but not used in M2 feature calculation.
      - 'specificity' from M1 is directly used in M2 feature calculations.
      - 'type' from M1 is directly evaluated against M2 SENSITIVE_CATEGORIES.
    """
    if not isinstance(m1_output, dict):
        raise TypeError(f"Expected dict for m1_output, got {type(m1_output).__name__}")

    raw_attributes = m1_output.get("attributes", [])
    if not isinstance(raw_attributes, list):
        raise TypeError(f"Expected list for attributes, got {type(raw_attributes).__name__}")

    # Preserve attributes while ensuring required keys for M2
    adapted_attributes = []
    for attr in raw_attributes:
        if not isinstance(attr, dict):
            continue
        adapted_attr = dict(attr)
        # Ensure specificity is float
        adapted_attr["specificity"] = float(adapted_attr.get("specificity", 0.0))
        adapted_attr["type"] = str(adapted_attr.get("type", "")).upper().strip()
        adapted_attributes.append(adapted_attr)

    raw_indicators = m1_output.get("contextual_indicators", [])
    adapted_indicators = []
    if isinstance(raw_indicators, list):
        for ind in raw_indicators:
            if isinstance(ind, dict):
                ind_name = str(ind.get("indicator", "")).strip()
                mapped_name = M1_TO_M2_INDICATOR_MAP.get(ind_name, ind_name)
                adapted_ind = dict(ind)
                adapted_ind["indicator"] = mapped_name
                adapted_indicators.append(adapted_ind)
            elif isinstance(ind, str):
                ind_name = ind.strip()
                mapped_name = M1_TO_M2_INDICATOR_MAP.get(ind_name, ind_name)
                adapted_indicators.append({
                    "indicator": mapped_name,
                    "attribute_type": "CONTEXTUAL",
                })

    adapted_conversation: dict[str, Any] = {
        "conversation_id": m1_output.get("conversation_id", ""),
        "original_text": m1_output.get("original_text", ""),
        "attributes": adapted_attributes,
        "contextual_indicators": adapted_indicators,
    }
    return adapted_conversation


def analyze_with_m2(m1_output: dict[str, Any]) -> dict[str, Any]:
    """
    End-to-end integration:
      1. Adapts M1 output via adapt_m1_to_m2()
      2. Computes features via M2 build_features()
      3. Computes risk via M2 score_risk()
    """
    adapted = adapt_m1_to_m2(m1_output)
    features = build_features(adapted)
    risk = score_risk(features)
    return {
        "features": features,
        "risk_score": risk["risk_score"],
        "risk_level": risk["risk_level"],
    }


# Convenience alias
score_m1_risk = analyze_with_m2
