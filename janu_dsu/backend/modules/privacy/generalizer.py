"""
generalizer.py
--------------
Transformation rules for each attribute type.

Each transformation function returns:
  {
    "new_value": str,       # the generalized value
    "info_loss_cost": float # 0.0 (no loss) → 1.0 (complete removal)
  }

The cost table drives the optimizer: cheapest transformation is tried first.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Age
# ---------------------------------------------------------------------------

# Ordered list of (min_age, max_age, label) — inclusive on both ends
_AGE_BANDS = [
    (0,   12,  "a child"),
    (13,  17,  "a teenager"),
    (18,  25,  "a young adult"),
    (26,  35,  "an adult in their late twenties or thirties"),
    (36,  50,  "a middle-aged adult"),
    (51,  65,  "an adult in their fifties or sixties"),
    (66,  120, "a senior"),
]


def generalize_age(value: str) -> dict:
    """
    Exact age (e.g. '23') → age-band label (e.g. 'a young adult').
    If already a band or unparseable, return as-is with no cost.
    """
    try:
        age = int(value)
    except (ValueError, TypeError):
        return {"new_value": value, "info_loss_cost": 0.0}

    for lo, hi, label in _AGE_BANDS:
        if lo <= age <= hi:
            return {"new_value": label, "info_loss_cost": 0.25}

    return {"new_value": "an adult", "info_loss_cost": 0.3}


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------

# Generalization ladder: each step up loses more detail
_LOCATION_LADDER = ["village", "block", "district", "state", "region"]

# Cost increases as we move up the ladder
_LOCATION_COSTS = {
    "village":  0.35,   # village → district area
    "block":    0.25,
    "district": 0.15,
    "state":    0.10,
    "region":   0.05,
}

_LOCATION_LABEL = {
    "village":  "a rural area",
    "block":    "a local area",
    "district": "a district",
    "state":    "a state",
    "region":   "a region",
}


def generalize_location(value: str, granularity: str = "village") -> dict:
    """
    Replace a specific place name with a generic geographic descriptor
    one level up the granularity ladder.
    """
    granularity = granularity.lower()
    if granularity not in _LOCATION_LADDER:
        granularity = "village"

    label = _LOCATION_LABEL.get(granularity, "an area")
    cost  = _LOCATION_COSTS.get(granularity, 0.2)
    return {"new_value": label, "info_loss_cost": cost}


# ---------------------------------------------------------------------------
# Facility
# ---------------------------------------------------------------------------

_FACILITY_MAP = {
    "phc":        "a healthcare facility",
    "hospital":   "a healthcare facility",
    "clinic":     "a healthcare facility",
    "dispensary": "a healthcare facility",
    "anganwadi":  "a community centre",
    "school":     "an educational institution",
    "bank":       "a financial institution",
    "post office":"a government office",
    "panchayat":  "a local government office",
}


def generalize_facility(value: str) -> dict:
    """
    Map a specific facility name/type to a generic category.
    Matches on prefix keywords (case-insensitive).
    """
    v = value.lower()
    for keyword, label in _FACILITY_MAP.items():
        if keyword in v:
            return {"new_value": label, "info_loss_cost": 0.2}
    # Unknown facility — replace with generic
    return {"new_value": "a facility", "info_loss_cost": 0.2}


# ---------------------------------------------------------------------------
# Date / Time
# ---------------------------------------------------------------------------

_DATE_GENERALIZATIONS = [
    # (keywords_that_match, generalized_label, cost)
    (["today", "tonight"],                          "recently",        0.15),
    (["yesterday"],                                 "recently",        0.15),
    (["this morning", "this afternoon"],            "recently",        0.15),
    (["last week", "past week"],                    "in the past week", 0.20),
    (["last month", "past month"],                  "recently",        0.25),
    (["this year"],                                 "this year",       0.10),
    (["monday","tuesday","wednesday","thursday",
      "friday","saturday","sunday"],                "recently",        0.20),
    (["january","february","march","april","may",
      "june","july","august","september","october",
      "november","december"],                       "recently",        0.25),
]


def generalize_date(value: str) -> dict:
    """
    Coarsen a specific date/time reference to a vaguer expression.
    """
    v = value.lower()
    for keywords, label, cost in _DATE_GENERALIZATIONS:
        if any(kw in v for kw in keywords):
            return {"new_value": label, "info_loss_cost": cost}
    # Fallback: numeric date pattern (e.g. "6 September", "2024-09-06")
    return {"new_value": "recently", "info_loss_cost": 0.25}


# ---------------------------------------------------------------------------
# Uniqueness words  (always removed — they signal rare combination)
# ---------------------------------------------------------------------------

_UNIQUENESS_WORDS = {
    "only", "first", "last", "youngest", "oldest",
    "only one", "sole", "only person", "alone",
    "ek hi", "akeli", "pehli", "akhri",   # Hindi equivalents
}


def generalize_uniqueness_word(value: str) -> dict:
    """
    Uniqueness indicators are always removed; they add high re-ID risk
    for near-zero analytical value.
    """
    return {"new_value": "", "info_loss_cost": 0.05}


# ---------------------------------------------------------------------------
# PII (name / phone / aadhaar / pan / email) — always redact
# ---------------------------------------------------------------------------

_PII_TYPES = {"NAME", "PHONE", "EMAIL", "AADHAAR", "PAN", "ADDRESS"}


def redact_pii(attr_type: str) -> dict:
    """Hard redaction for explicit PII — no generalization alternative."""
    return {"new_value": "[REDACTED]", "info_loss_cost": 1.0}


# ---------------------------------------------------------------------------
# Public dispatch: given an attribute dict, return the best transformation
# ---------------------------------------------------------------------------

def get_transformation(attribute: dict) -> dict | None:
    """
    Given an attribute dict (with keys: type, value, specificity, ...),
    return a transformation dict:
      {
        "attribute":      str,   # attribute type
        "original":       str,   # original value
        "new_value":      str,   # generalized value
        "info_loss_cost": float
      }
    Returns None if no transformation is applicable (e.g. low-specificity
    health/gender attributes that should be kept).
    """
    attr_type  = attribute.get("type", "").upper()
    attr_value = attribute.get("value", "")
    specificity = attribute.get("specificity", 0.0)

    if attr_type in _PII_TYPES:
        result = redact_pii(attr_type)

    elif attr_type == "AGE":
        result = generalize_age(attr_value)

    elif attr_type == "LOCATION":
        granularity = attribute.get("granularity", "village")
        result = generalize_location(attr_value, granularity)

    elif attr_type == "FACILITY":
        result = generalize_facility(attr_value)

    elif attr_type == "DATE":
        result = generalize_date(attr_value)

    elif attr_type == "UNIQUENESS_WORD":
        result = generalize_uniqueness_word(attr_value)

    elif attr_type in {"HEALTH", "GENDER", "OCCUPATION", "RELIGION", "CASTE"}:
        # Keep these by default — high analytical value, low direct re-ID risk.
        # The optimizer will only touch them if risk is still too high after
        # cheaper transformations are exhausted.
        if specificity >= 0.9:
            result = {"new_value": f"a person with a {attr_type.lower()} attribute",
                      "info_loss_cost": 0.6}
        else:
            return None   # keep as-is

    else:
        # Unknown type — skip
        return None

    return {
        "attribute":      attr_type,
        "original":       attr_value,
        "new_value":      result["new_value"],
        "info_loss_cost": result["info_loss_cost"],
    }
