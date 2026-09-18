"""
optimizer.py
------------
Finds the minimum set of transformations that brings re-identification risk
below the configured threshold, while minimising information loss.

Algorithm (greedy cost-ordered search):
  1. Collect all candidate transformations for each attribute.
  2. Sort candidates by info_loss_cost ascending (cheapest first).
  3. While risk > threshold AND candidates remain:
       a. Apply the next cheapest transformation.
       b. Re-score risk.
       c. If risk ≤ threshold → stop.
  4. Return sanitized text, transformations applied, final risk, utility stats.

The risk re-scorer is a local placeholder that Module 2 can replace by
passing in a custom `scorer_fn` callable with signature:
    scorer_fn(attributes: list[dict]) -> float
"""

from __future__ import annotations
import copy
import re

from backend.modules.privacy.generalizer import get_transformation
from backend.modules.privacy.utility import utility_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_RISK_THRESHOLD = 0.40   # risk must fall to or below this value

# ---------------------------------------------------------------------------
# Placeholder risk re-scorer
# (Module 2 will replace this with their ML model)
# ---------------------------------------------------------------------------

# Weights used by the placeholder scorer — how much each attribute
# type contributes to re-identification risk.
_RISK_WEIGHTS: dict[str, float] = {
    "UNIQUENESS_WORD": 0.30,
    "LOCATION":        0.25,
    "AGE":             0.15,
    "FACILITY":        0.12,
    "DATE":            0.10,
    "HEALTH":          0.08,
    "GENDER":          0.05,
    "OCCUPATION":      0.05,
    "NAME":            0.40,
    "PHONE":           0.40,
    "AADHAAR":         0.40,
    "PAN":             0.40,
    "EMAIL":           0.35,
    "ADDRESS":         0.20,
}
_DEFAULT_RISK_WEIGHT = 0.05


def _placeholder_scorer(attributes: list[dict]) -> float:
    """
    Heuristic risk score based on attribute specificity and type weights.

    Score = weighted sum of (specificity × type_weight) for all attributes,
    clamped to [0.0, 1.0].

    This is intentionally simple — it is only used until Module 2 plugs in
    their real model via the `scorer_fn` parameter of `optimize`.
    """
    if not attributes:
        return 0.0

    total = 0.0
    for attr in attributes:
        attr_type   = attr.get("type", "").upper()
        specificity = float(attr.get("specificity", 0.5))
        weight      = _RISK_WEIGHTS.get(attr_type, _DEFAULT_RISK_WEIGHT)
        total      += specificity * weight

    # Clamp and scale: sum can exceed 1.0, so we use a soft cap via tanh-like
    # mapping — but keep it simple with a plain clamp for now.
    return round(min(total, 1.0), 4)


# ---------------------------------------------------------------------------
# Text patching helper
# ---------------------------------------------------------------------------

def _apply_text_patch(text: str, original: str, new_value: str) -> str:
    """
    Replace `original` with `new_value` in `text` (case-insensitive).
    If new_value is empty (uniqueness word removal), also strip a
    leading/trailing comma or space around the removed word.
    """
    if not original:
        return text

    pattern = re.compile(re.escape(original), re.IGNORECASE)

    if not new_value:
        # Remove the word and any immediately surrounding punctuation/spaces
        text = pattern.sub("", text)
        # Clean up double spaces or leading/trailing spaces
        text = re.sub(r"  +", " ", text).strip()
        # Clean up sentences that now start with lowercase due to removal
        # (best-effort: capitalise first letter)
        if text:
            text = text[0].upper() + text[1:]
    else:
        text = pattern.sub(new_value, text)

    return text


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

def optimize(
    conversation_input: dict,
    threshold: float = DEFAULT_RISK_THRESHOLD,
    scorer_fn=None,
) -> dict:
    """
    Run the privacy optimization pipeline on a conversation.

    Parameters
    ----------
    conversation_input : dict
        The full output from Module 2.  Must contain:
          - conversation_id : str
          - original_text   : str
          - attributes      : list[dict]
          - risk_score      : float
          - risk_level      : str
    threshold : float
        Target maximum re-identification risk score (default 0.40).
    scorer_fn : callable | None
        Optional external risk scorer (Module 2 integration point).
        Signature: scorer_fn(attributes: list[dict]) -> float
        If None, uses the built-in placeholder scorer.

    Returns
    -------
    dict with keys:
      conversation_id, original_text, sanitized_text,
      initial_risk_score, final_risk_score, risk_reduced,
      transformations, utility, threshold_used
    """
    scorer = scorer_fn if scorer_fn is not None else _placeholder_scorer

    conversation_id = conversation_input["conversation_id"]
    original_text   = conversation_input["original_text"]
    initial_risk    = float(conversation_input["risk_score"])
    original_attrs  = conversation_input["attributes"]

    # Deep-copy attributes so we can mutate specificity without affecting input
    working_attrs = copy.deepcopy(original_attrs)
    sanitized_text = original_text

    # -----------------------------------------------------------------------
    # Step 1: If already below threshold, return immediately (no change)
    # -----------------------------------------------------------------------
    current_risk = scorer(working_attrs)

    if current_risk <= threshold:
        return _build_result(
            conversation_id=conversation_id,
            original_text=original_text,
            sanitized_text=sanitized_text,
            initial_risk=initial_risk,
            final_risk=current_risk,
            transformations=[],
            original_attrs=original_attrs,
            threshold=threshold,
        )

    # -----------------------------------------------------------------------
    # Step 2: Build candidate transformation list (sorted by cost, cheapest first)
    # -----------------------------------------------------------------------
    candidates = []
    for attr in working_attrs:
        t = get_transformation(attr)
        if t is not None:
            candidates.append(t)

    # Sort cheapest transformation first
    candidates.sort(key=lambda x: x["info_loss_cost"])

    applied_transformations = []

    # -----------------------------------------------------------------------
    # Step 3: Greedy loop — apply cheapest transformation, re-score, repeat
    # -----------------------------------------------------------------------
    for candidate in candidates:
        if current_risk <= threshold:
            break

        attr_type = candidate["attribute"]
        original_value = candidate["original"]
        new_value = candidate["new_value"]

        # Apply transformation to working attributes
        for attr in working_attrs:
            if attr.get("type", "").upper() == attr_type:
                attr["value"]       = new_value
                attr["specificity"] = _new_specificity(attr_type, new_value)
                break

        # Patch the running sanitized text
        sanitized_text = _apply_text_patch(sanitized_text, original_value, new_value)

        # Record transformation
        applied_transformations.append({
            "attribute":      attr_type,
            "original":       original_value,
            "new_value":      new_value,
            "info_loss_cost": candidate["info_loss_cost"],
        })

        # Re-score
        current_risk = scorer(working_attrs)

    # -----------------------------------------------------------------------
    # Step 4: Build and return the result
    # -----------------------------------------------------------------------
    return _build_result(
        conversation_id=conversation_id,
        original_text=original_text,
        sanitized_text=sanitized_text,
        initial_risk=initial_risk,
        final_risk=current_risk,
        transformations=applied_transformations,
        original_attrs=original_attrs,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_specificity(attr_type: str, new_value: str) -> float:
    """
    After generalization, an attribute's specificity drops.
    These are reasonable defaults — Module 2 can override via scorer_fn.
    """
    if not new_value:
        return 0.0   # removed

    specificity_after: dict[str, float] = {
        "AGE":             0.3,
        "LOCATION":        0.3,
        "FACILITY":        0.2,
        "DATE":            0.2,
        "UNIQUENESS_WORD": 0.0,
        "HEALTH":          0.4,
        "GENDER":          0.4,
    }
    return specificity_after.get(attr_type.upper(), 0.3)


def _risk_level_label(score: float) -> str:
    if score <= 0.35:
        return "LOW"
    elif score <= 0.65:
        return "MEDIUM"
    else:
        return "HIGH"


def _build_result(
    conversation_id, original_text, sanitized_text,
    initial_risk, final_risk, transformations,
    original_attrs, threshold,
) -> dict:
    utility = utility_score(original_attrs, transformations)
    return {
        "conversation_id":    conversation_id,
        "original_text":      original_text,
        "sanitized_text":     sanitized_text,
        "initial_risk_score": round(initial_risk, 4),
        "final_risk_score":   round(final_risk, 4),
        "initial_risk_level": _risk_level_label(initial_risk),
        "final_risk_level":   _risk_level_label(final_risk),
        "risk_reduced":       round(initial_risk - final_risk, 4),
        "threshold_used":     threshold,
        "transformations":    transformations,
        "utility":            utility,
    }
