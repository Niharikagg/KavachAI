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
from typing import Any

from backend.modules.privacy.generalizer import get_transformation
from backend.modules.privacy.utility import utility_score
from backend.modules.risk.features import build_features
from backend.modules.risk.scorer import score_risk

# Temporal adverbs produced by generalizer.generalize_date() that should NOT
# be preceded by a temporal preposition in the sanitized text.
# When one of these replaces a specific date, any stranded "on/in/at" before
# it is stripped to avoid output like "on recently" or "at this year".
_TEMPORAL_ADVERBS: frozenset[str] = frozenset({
    "recently",
    "in the past week",
    "in the past month",
    "this year",
})

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_RISK_THRESHOLD = 0.40   # risk must fall to or below this value

# M2 score values are on a 0-10 scale. M3 optimization works on a 0-1 scale,
# so the authoritative M2 score is normalized as: m2_score / 10.0.


def _normalize_m2_risk(risk_score: float) -> float:
    """Convert the authoritative M2 risk score (0-10) to the M3 internal scale (0-1)."""
    return round(float(risk_score) / 10.0, 4)


def _normalize_contextual_indicators(contextual_indicators: list[dict] | list[str] | None) -> list[dict]:
    """Normalize M3 contextual indicators to M2's ``indicator``/``attribute_type`` schema."""
    if not contextual_indicators:
        return []

    indicator_attribute_types = {
        "uniqueness_word": "UNIQUENESS_WORD",
        "uniqueness_indicator": "UNIQUENESS_WORD",
        "village_level_location": "LOCATION",
        "exact_age": "AGE",
        "exact_date": "DATE",
        "sensitive_health_attribute": "HEALTH",
        "recent_healthcare_visit": "FACILITY",
    }
    indicator_aliases = {
        "uniqueness_indicator": "uniqueness_word",
    }

    normalized: list[dict] = []
    for indicator in contextual_indicators:
        if isinstance(indicator, dict):
            # Preserve M2-shaped input while filling legacy partial payloads.
            item = dict(indicator)
            indicator_name = str(item.get("indicator", ""))
            item["indicator"] = indicator_aliases.get(indicator_name, indicator_name)
            if not item.get("attribute_type"):
                item["attribute_type"] = indicator_attribute_types.get(
                    indicator_name, "CONTEXTUAL"
                )
            normalized.append(item)
            continue

        indicator_name = str(indicator)
        mapped = indicator_aliases.get(indicator_name, indicator_name)
        normalized.append({
            "indicator": mapped,
            "attribute_type": indicator_attribute_types.get(indicator_name, "CONTEXTUAL"),
        })

    return normalized


def _m2_risk_for_attributes(attributes: list[dict], contextual_indicators: list[dict] | list[str] | None = None) -> float:
    """Compute a normalized M2 risk score for a working attribute set."""
    conversation = {
        "attributes": attributes,
        "contextual_indicators": _normalize_contextual_indicators(contextual_indicators),
    }
    features = build_features(conversation)
    m2_risk = float(score_risk(features)["risk_score"])
    return _normalize_m2_risk(m2_risk)


def _score_with_scorer(attributes: list[dict], contextual_indicators: list[dict] | list[str] | None = None, scorer_fn=None) -> float:
    """Use the configured risk scorer; default to the real M2 implementation."""
    if scorer_fn is not None:
        try:
            return float(scorer_fn(attributes, contextual_indicators or []))
        except TypeError:
            return float(scorer_fn(attributes))
    return _m2_risk_for_attributes(attributes, contextual_indicators)


# ---------------------------------------------------------------------------
# Text patching helper
# ---------------------------------------------------------------------------

def _apply_text_patch(text: str, original: str, new_value: str) -> str:
    """
    Replace `original` with `new_value` in `text` (case-insensitive).
    If new_value is empty (uniqueness word removal), also strip a
    leading/trailing comma or space around the removed word.

    Grammar fixes applied after the main substitution:
    * Orphaned temporal prepositions: when `new_value` is a temporal adverb
      (e.g. "recently"), any preceding "on"/"in"/"at"/"during"/"by" that
      was left behind by the date replacement is removed.
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
        # A generalized value can include its own indefinite article (for
        # example, HEALTH -> "a person with a health attribute").  When the
        # original value was already preceded by "a" or "an", replace that
        # complete noun phrase so we do not leave two articles behind.
        if re.match(r"^(?:a|an)\s+", new_value, flags=re.IGNORECASE):
            phrase_pattern = re.compile(
                rf"\b(?:a|an)\s+{re.escape(original)}",
                re.IGNORECASE,
            )
            text, replacements = phrase_pattern.subn(new_value, text)
            if replacements:
                return text
        # HEALTH categories are bare noun phrases (for example, "autoimmune
        # condition"). Replace a preceding indefinite article together with
        # the old phrase, selecting the appropriate article for the category.
        phrase_pattern = re.compile(
            rf"\b(?:a|an)\s+{re.escape(original)}",
            re.IGNORECASE,
        )
        article = "an" if re.match(r"^[aeiou]", new_value, re.IGNORECASE) else "a"
        text, replacements = phrase_pattern.subn(f"{article} {new_value}", text)
        if replacements:
            return text
        text = pattern.sub(new_value, text)

        # --- Fix B: strip orphaned temporal preposition before temporal adverbs ---
        # e.g. "visited on September 12, 2026" → "visited on recently" (bad)
        #      → "visited recently" (correct)
        nv_lower = new_value.lower()
        if nv_lower in _TEMPORAL_ADVERBS:
            preps = ["on", "at", "during", "by"]
            # Only strip "in" when the replacement does not itself start with "in"
            # (avoids stripping from "in the past week")
            if not nv_lower.startswith("in"):
                preps.append("in")
            prep_pat = r"\b(?:" + "|".join(preps) + r")\s+(?=" + re.escape(new_value) + r"\b)"
            text = re.sub(prep_pat, "", text, flags=re.IGNORECASE)
            text = re.sub(r"  +", " ", text).strip()

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
                Optional metadata is preserved in the result:
                    - contextual_indicators : list[str]
                    - risk_level             : str
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
    conversation_id       = conversation_input["conversation_id"]
    original_text         = conversation_input["original_text"]
    original_attrs        = conversation_input["attributes"]
    contextual_indicators = conversation_input.get("contextual_indicators", [])
    # Use pre-computed M2 risk if the adapter already injected it; otherwise
    # call M2 internally.  Backward-compatible: existing tests that do NOT
    # supply initial_risk_score continue to work unchanged.
    if 'initial_risk_score' in conversation_input:
        raw_score    = float(conversation_input['initial_risk_score'])
        initial_risk = raw_score / 10.0 if raw_score > 1.0 else raw_score
        risk_level   = (
            conversation_input.get('initial_risk_level')
            or conversation_input.get('risk_level')
            or _risk_level_label(initial_risk)
        )
    else:
        initial_risk = _m2_risk_for_attributes(original_attrs, contextual_indicators)
        # M2 is authoritative for the score and its corresponding risk level;
        # input metadata can be stale relative to the attributes being optimized.
        risk_level   = _risk_level_label(initial_risk)

    # Deep-copy attributes so we can mutate specificity without affecting input
    working_attrs = copy.deepcopy(original_attrs)
    sanitized_text = original_text

    # -----------------------------------------------------------------------
    # Step 1: If already below threshold, return immediately (no change)
    # -----------------------------------------------------------------------
    # When a pre-computed M2 score is supplied and no custom scorer is active,
    # reuse it directly to avoid a redundant M2 call.
    if 'initial_risk_score' in conversation_input and scorer_fn is None:
        current_risk = initial_risk
    else:
        current_risk = _score_with_scorer(working_attrs, contextual_indicators, scorer_fn)

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
            risk_level=risk_level,
            contextual_indicators=contextual_indicators,
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
        if attr_type == "AGE":
            sanitized_text = re.sub(
                rf"{re.escape(new_value)}\s+years?\s+old\b",
                new_value,
                sanitized_text,
                flags=re.IGNORECASE,
            )
            # Fix A: collapse "only a <age-band>" → "only <age-band>".
            # This artifact arises when an age-band label starting with "a/an"
            # is inserted right after the word "only" (e.g. "the only a young adult").
            sanitized_text = re.sub(
                r"\bonly\s+(?:a|an)\s+(?=[a-z])",
                "only ",
                sanitized_text,
                flags=re.IGNORECASE,
            )

        # Record transformation
        applied_transformations.append({
            "attribute":      attr_type,
            "original":       original_value,
            "new_value":      new_value,
            "info_loss_cost": candidate["info_loss_cost"],
        })

        # Re-score with the real M2 risk model normalized to the M3 0-1 scale
        current_risk = _score_with_scorer(working_attrs, contextual_indicators, scorer_fn)

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
        risk_level=risk_level,
        contextual_indicators=contextual_indicators,
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
    # M3 uses M2's 0-10 score normalized to 0-1.  Preserve M2's thresholds:
    # LOW <= 3.33 and MEDIUM <= 6.66.
    if score <= 0.333:
        return "LOW"
    elif score <= 0.666:
        return "MEDIUM"
    else:
        return "HIGH"


def _build_result(
    conversation_id, original_text, sanitized_text,
    initial_risk, final_risk, transformations,
    original_attrs, threshold, risk_level=None, contextual_indicators=None,
) -> dict:
    utility = utility_score(original_attrs, transformations)
    result = {
        "conversation_id":    conversation_id,
        "original_text":      original_text,
        "sanitized_text":     sanitized_text,
        "initial_risk_score": round(initial_risk, 4),
        "final_risk_score":   round(final_risk, 4),
        "initial_risk_level": risk_level or _risk_level_label(initial_risk),
        "final_risk_level":   _risk_level_label(final_risk),
        "risk_reduced":       round(initial_risk - final_risk, 4),
        "threshold_used":     threshold,
        "transformations":    transformations,
        "utility":            utility,
    }
    if risk_level is not None:
        result["risk_level"] = risk_level
    if contextual_indicators is not None:
        result["contextual_indicators"] = contextual_indicators
    return result
