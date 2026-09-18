from typing import Any


ATTRIBUTE_COUNT_IMPORTANCE = 0.4
AVERAGE_SPECIFICITY_IMPORTANCE = 0.4
MAXIMUM_SPECIFICITY_IMPORTANCE = 0.6
HIGH_SPECIFICITY_IMPORTANCE = 0.6
SENSITIVE_CATEGORY_IMPORTANCE = 0.8
UNIQUENESS_IMPORTANCE = 0.8

LOW_RISK_MAX_SCORE = 3.33
MEDIUM_RISK_MAX_SCORE = 6.66
RISK_SCORE_SCALE = 10.0

TOTAL_IMPORTANCE = sum(
    (
        ATTRIBUTE_COUNT_IMPORTANCE,
        AVERAGE_SPECIFICITY_IMPORTANCE,
        MAXIMUM_SPECIFICITY_IMPORTANCE,
        HIGH_SPECIFICITY_IMPORTANCE,
        SENSITIVE_CATEGORY_IMPORTANCE,
        UNIQUENESS_IMPORTANCE,
    )
)


def _risk_level(risk_score: float) -> str:
    if risk_score <= LOW_RISK_MAX_SCORE:
        return "LOW"
    if risk_score <= MEDIUM_RISK_MAX_SCORE:
        return "MEDIUM"
    return "HIGH"


def score_risk(features: dict[str, float]) -> dict[str, Any]:
    weighted_sum = (
        ATTRIBUTE_COUNT_IMPORTANCE * features["attribute_count_normalized"]
        + AVERAGE_SPECIFICITY_IMPORTANCE * features["average_specificity"]
        + MAXIMUM_SPECIFICITY_IMPORTANCE * features["maximum_specificity"]
        + HIGH_SPECIFICITY_IMPORTANCE * features["high_specificity_ratio"]
        + SENSITIVE_CATEGORY_IMPORTANCE
        * features["sensitive_category_count_normalized"]
        + UNIQUENESS_IMPORTANCE * features["uniqueness_risk"]
    )
    raw_risk = weighted_sum / TOTAL_IMPORTANCE
    risk_score = raw_risk * RISK_SCORE_SCALE

    return {
        "risk_score": risk_score,
        "risk_level": _risk_level(risk_score),
    }
