"""
m2_m3_adapter.py — Thin adapter connecting Module 2 (Risk) to Module 3 (Privacy Optimizer).

Responsibility:
  1. Accepts M1 conversation context (conversation_id, original_text, attributes, contextual_indicators)
     and authoritative M2 result (features, risk_score, risk_level).
  2. Merges them into the exact input structure expected by M3 optimize().
  3. Does NOT duplicate M2 feature calculation or risk calculation.
"""

from __future__ import annotations

from typing import Any


def adapt_m2_to_m3(
    conversation_context: dict[str, Any],
    m2_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge M1 conversation context with authoritative M2 risk results into
    the dictionary contract required by M3 optimizer.optimize().

    Parameters
    ----------
    conversation_context : dict
        Contains M1 data:
          - conversation_id: str
          - original_text: str
          - attributes: list[dict]
          - contextual_indicators: list[dict] | list[str]
    m2_result : dict
        Contains M2 risk output:
          - features: dict[str, float]
          - risk_score: float (M2 scale: 0.0 - 10.0)
          - risk_level: str ("LOW" | "MEDIUM" | "HIGH")

    Returns
    -------
    dict
        M3-ready input dictionary with conversation metadata and authoritative M2 risk scores.
    """
    if not isinstance(conversation_context, dict):
        raise TypeError(f"Expected dict for conversation_context, got {type(conversation_context).__name__}")
    if not isinstance(m2_result, dict):
        raise TypeError(f"Expected dict for m2_result, got {type(m2_result).__name__}")

    return {
        "conversation_id": conversation_context.get("conversation_id", ""),
        "original_text": conversation_context.get("original_text", ""),
        "attributes": list(conversation_context.get("attributes", [])),
        "contextual_indicators": list(conversation_context.get("contextual_indicators", [])),
        "features": m2_result.get("features", {}),
        "risk_score": m2_result["risk_score"],
        "risk_level": m2_result.get("risk_level", ""),
        "initial_risk_score": m2_result["risk_score"],
        "initial_risk_level": m2_result.get("risk_level", ""),
    }
