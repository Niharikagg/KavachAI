"""
utility.py
----------
Measures how much information was preserved / lost after transformation.

All scores are in [0.0, 1.0].

  information_loss      — fraction of attribute value that was degraded
  information_retained  — 1.0 - information_loss
  utility_score         — weighted retained value, penalising high-value
                          attributes (health, gender) more than low-value ones
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Analytical value weights — how important is each attribute for
# aggregate analysis (government analyst perspective).
# Higher weight = more valuable to retain.
# ---------------------------------------------------------------------------

_ATTRIBUTE_VALUE_WEIGHTS: dict[str, float] = {
    "HEALTH":           0.9,
    "GENDER":           0.7,
    "AGE":              0.7,
    "LOCATION":         0.6,
    "OCCUPATION":       0.5,
    "FACILITY":         0.4,
    "DATE":             0.3,
    "UNIQUENESS_WORD":  0.05,  # near-zero analytical value
    "NAME":             0.0,   # no analytical value — PII only
    "PHONE":            0.0,
    "EMAIL":            0.0,
    "AADHAAR":          0.0,
    "PAN":              0.0,
    "ADDRESS":          0.2,
}

_DEFAULT_WEIGHT = 0.3


def _weight(attr_type: str) -> float:
    return _ATTRIBUTE_VALUE_WEIGHTS.get(attr_type.upper(), _DEFAULT_WEIGHT)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def information_loss(transformations: list[dict]) -> float:
    """
    Weighted average information-loss cost across all applied transformations.

    Each transformation dict must contain:
      - "attribute":       str   (attribute type)
      - "info_loss_cost":  float (0.0 = no loss, 1.0 = full removal)

    Returns a float in [0.0, 1.0].
    Returns 0.0 if no transformations were applied.
    """
    if not transformations:
        return 0.0

    weighted_loss  = 0.0
    total_weight   = 0.0

    for t in transformations:
        attr_type = t.get("attribute", "UNKNOWN")
        cost      = float(t.get("info_loss_cost", 0.0))
        w         = _weight(attr_type)
        weighted_loss += cost * w
        total_weight  += w

    if total_weight == 0.0:
        return 0.0

    return round(weighted_loss / total_weight, 4)


def information_retained(transformations: list[dict]) -> float:
    """1.0 - information_loss(transformations)."""
    return round(1.0 - information_loss(transformations), 4)


def utility_score(
    original_attributes: list[dict],
    transformations: list[dict],
) -> dict:
    """
    Full utility breakdown.

    Parameters
    ----------
    original_attributes : list of attribute dicts from Module 1/2
    transformations     : list of transformation dicts applied by optimizer

    Returns
    -------
    {
        "information_loss":      float,
        "information_retained":  float,
        "attributes_total":      int,
        "attributes_changed":    int,
        "attributes_kept":       int,
        "attribute_detail":      list[dict]   # per-attribute breakdown
    }
    """
    transformed_types = {t["attribute"].upper() for t in transformations}
    detail = []

    for attr in original_attributes:
        attr_type = attr.get("type", "UNKNOWN").upper()
        changed   = attr_type in transformed_types

        # Find the matching transformation (if any) for cost
        cost = 0.0
        new_value = attr.get("value", "")
        if changed:
            match = next(
                (t for t in transformations if t["attribute"].upper() == attr_type),
                None
            )
            if match:
                cost      = match.get("info_loss_cost", 0.0)
                new_value = match.get("new_value", "")

        detail.append({
            "attribute":      attr_type,
            "original_value": attr.get("value", ""),
            "new_value":      new_value,
            "changed":        changed,
            "info_loss_cost": cost,
            "weight":         _weight(attr_type),
        })

    changed_count = sum(1 for d in detail if d["changed"])
    loss          = information_loss(transformations)

    return {
        "information_loss":     loss,
        "information_retained": round(1.0 - loss, 4),
        "attributes_total":     len(original_attributes),
        "attributes_changed":   changed_count,
        "attributes_kept":      len(original_attributes) - changed_count,
        "attribute_detail":     detail,
    }
