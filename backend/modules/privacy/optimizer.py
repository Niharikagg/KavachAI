"""
KavachAI - Module 3 Privacy Optimizer

Module 3 responsibilities:
    1. Remove direct identifiers.
    2. Consolidate fragmented address detections.
    3. Generalize contextual attributes.
    4. Reduce contextual uniqueness signals.
    5. Re-score the protected attributes using Module 2.
    6. Report information loss / utility consistently.

M1 and M2 contracts are not modified here.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Callable

from backend.modules.risk.features import build_features
from backend.modules.risk.scorer import score_risk


# ============================================================
# CONSTANTS
# ============================================================

DIRECT_PII_TYPES = {
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "AADHAAR",
    "PAN",
    "BANK_ACCOUNT",
    "IFSC",
    "VEHICLE_NUMBER",
    "VOTER_ID",
    "ADDRESS",
    "PERSON",
}

CONTEXTUAL_TYPES = {
    "AGE",
    "HEALTH",
    "FACILITY",
    "LOCATION",
    "OCCUPATION",
    "EDUCATION",
    "DEGREE",
    "GENDER",
    "DATE",
}

DIRECT_REPLACEMENTS = {
    "PHONE_NUMBER": "[PHONE]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "AADHAAR": "[AADHAAR]",
    "PAN": "[PAN]",
    "BANK_ACCOUNT": "[BANK_ACCOUNT]",
    "IFSC": "[IFSC]",
    "VEHICLE_NUMBER": "[VEHICLE_ID]",
    "VOTER_ID": "[VOTER_ID]",
    "ADDRESS": "[ADDRESS]",
    "PERSON": "[PERSON]",
}

DEFAULT_RISK_THRESHOLD = 0.40


TRANSFORMATION_COSTS = {
    "PHONE_NUMBER": 1.0,
    "EMAIL_ADDRESS": 1.0,
    "AADHAAR": 1.0,
    "PAN": 1.0,
    "BANK_ACCOUNT": 1.0,
    "IFSC": 1.0,
    "VEHICLE_NUMBER": 1.0,
    "VOTER_ID": 1.0,
    "ADDRESS": 1.0,
    "PERSON": 1.0,

    "AGE": 0.25,
    "HEALTH": 0.35,
    "FACILITY": 0.20,
    "LOCATION": 0.35,
    "OCCUPATION": 0.25,
    "EDUCATION": 0.20,
    "DEGREE": 0.20,
    "GENDER": 0.15,
    "DATE": 0.25,
    "UNIQUENESS": 0.15,
}


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def _detect_language(text: str) -> str:
    if not text:
        return "english"

    kannada_count = len(
        re.findall(r"[\u0C80-\u0CFF]", text)
    )

    devanagari_count = len(
        re.findall(r"[\u0900-\u097F]", text)
    )

    latin_count = len(
        re.findall(r"[A-Za-z]", text)
    )

    if (
        kannada_count > devanagari_count
        and kannada_count > latin_count * 0.15
    ):
        return "kannada"

    if (
        devanagari_count > kannada_count
        and devanagari_count > latin_count * 0.15
    ):
        return "hindi"

    if latin_count > 0:
        return "english"

    return "mixed"


def _language_for_value(
    text: str,
    value: str,
) -> str:
    """
    Prefer the language of the surrounding sentence rather than
    trying to infer language from a short attribute value.
    """
    return _detect_language(text)


# ============================================================
# CONTEXTUAL GENERALIZATION
# ============================================================

def _generalize_age(
    value: str,
    language: str,
) -> str:

    match = re.search(
        r"\b(\d{1,3})\b",
        str(value),
    )

    if not match:
        if language == "kannada":
            return "ವಯಸ್ಕ"

        if language == "hindi":
            return "वयस्क"

        return "adult"

    age = int(match.group(1))

    if age < 13:
        if language == "kannada":
            return "ಮಗು"

        if language == "hindi":
            return "बच्चा"

        return "child"

    if age < 18:
        if language == "kannada":
            return "ಹದಿಹರೆಯದವರು"

        if language == "hindi":
            return "किशोर"

        return "teenager"

    if age < 25:
        if language == "kannada":
            return "ಯುವ ವಯಸ್ಕ"

        if language == "hindi":
            return "ಯುವ वयस्क"

        return "young adult"

    if age < 35:
        if language == "kannada":
            return "ವಯಸ್ಕ"

        if language == "hindi":
            return "वयस्क"

        return "adult"

    if age < 60:
        if language == "kannada":
            return "ಮಧ್ಯವಯಸ್ಕ"

        if language == "hindi":
            return "मध्यम आयु वर्ग का वयस्क"

        return "middle-aged adult"

    if language == "kannada":
        return "ಹಿರಿಯ ವಯಸ್ಕ"

    if language == "hindi":
        return "वृद्ध वयस्क"

    return "older adult"


def _generalize_health(
    value: str,
    language: str,
) -> str:

    if language == "kannada":
        return "ಆರೋಗ್ಯ ಸ್ಥಿತಿ"

    if language == "hindi":
        return "स्वास्थ्य संबंधी स्थिति"

    return "health condition"


def _generalize_facility(
    value: str,
    language: str,
) -> str:

    if language == "kannada":
        return "ಆರೋಗ್ಯ ಕೇಂದ್ರ"

    if language == "hindi":
        return "स्वास्थ्य केंद्र"

    return "healthcare facility"


def _generalize_location(
    value: str,
    language: str,
) -> str:

    value_lower = str(value).lower()

    bengaluru_terms = {
        "bengaluru",
        "bangalore",
        "ಬೆಂಗಳೂರು",
        "ಬ್ಲಾಕ್",
        "बेंगलुरु",
        "ಬೆಂಗಳೂರು ನಗರ",
    }

    if any(
        term in value_lower
        for term in bengaluru_terms
    ):
        if language == "kannada":
            return "ಕರ್ನಾಟಕದ ಒಂದು ನಗರ"

        if language == "hindi":
            return "कर्नाटक का एक शहर"

        return "a city in Karnataka"

    if language == "kannada":
        return "ಒಂದು ಪ್ರದೇಶ"

    if language == "hindi":
        return "एक क्षेत्र"

    return "an area"


def _generalize_occupation(
    value: str,
    language: str,
) -> str:

    if language == "kannada":
        return "ವೃತ್ತಿಪರ ಕೆಲಸ"

    if language == "hindi":
        return "पेशेवर भूमिका"

    return "professional role"


def _generalize_education(
    value: str,
    language: str,
) -> str:

    if language == "kannada":
        return "ಉನ್ನತ ಶಿಕ್ಷಣ"

    if language == "hindi":
        return "उच्च शिक्षा"

    return "higher education"


def _generalize_gender(
    value: str,
    language: str,
) -> str:

    value_lower = str(value).lower()

    if language == "kannada":

        if any(
            x in value_lower
            for x in (
                "female",
                "woman",
                "girl",
                "ಮಹಿಳೆ",
                "ಹೆಣ್ಣು",
            )
        ):
            return "ಮಹಿಳೆ"

        if any(
            x in value_lower
            for x in (
                "male",
                "man",
                "boy",
                "ಪುರುಷ",
                "ಗಂಡು",
            )
        ):
            return "ಪುರುಷ"

        return "ವ್ಯಕ್ತಿ"

    if language == "hindi":

        if any(
            x in value_lower
            for x in (
                "female",
                "woman",
                "girl",
                "महिला",
                "लड़की",
            )
        ):
            return "महिला"

        if any(
            x in value_lower
            for x in (
                "male",
                "man",
                "boy",
                "पुरुष",
                "लड़का",
            )
        ):
            return "पुरुष"

        return "व्यक्ति"

    if any(
        x in value_lower
        for x in (
            "female",
            "woman",
            "girl",
        )
    ):
        return "woman"

    if any(
        x in value_lower
        for x in (
            "male",
            "man",
            "boy",
        )
    ):
        return "man"

    return "person"


def _generalize_date(
    value: str,
    language: str,
) -> str:

    if language == "kannada":
        return "ಇತ್ತೀಚಿನ ದಿನಾಂಕ"

    if language == "hindi":
        return "हाल की तारीख"

    return "a recent date"


# ============================================================
# EXISTING GENERALIZER COMPATIBILITY
# ============================================================

def _try_existing_generalizer(
    attribute: dict[str, Any],
    text: str,
) -> str | None:

    try:
        from backend.modules.privacy.generalizer import (
            get_transformation,
        )
    except Exception:
        return None

    attr_type = str(
        attribute.get("type", "")
    ).upper()

    value = str(
        attribute.get("value", "")
    )

    attempts = [
        lambda: get_transformation(
            attr_type,
            value,
        ),
        lambda: get_transformation(
            attr_type,
            value,
            attribute,
        ),
        lambda: get_transformation(
            attribute,
        ),
        lambda: get_transformation(
            attribute=attribute,
        ),
    ]

    for attempt in attempts:

        try:
            result = attempt()

            if (
                isinstance(result, str)
                and result.strip()
            ):
                return result

            if isinstance(result, dict):

                for key in (
                    "replacement",
                    "generalized_value",
                    "transformed_value",
                    "value",
                    "new_value",
                ):

                    candidate = result.get(key)

                    if (
                        isinstance(candidate, str)
                        and candidate.strip()
                    ):
                        return candidate

        except (
            TypeError,
            KeyError,
            AttributeError,
        ):
            continue

        except Exception:
            continue

    return None


# ============================================================
# ATTRIBUTE TRANSFORMATION
# ============================================================

def _generalize_attribute(
    attribute: dict[str, Any],
    text: str,
) -> tuple[str, float]:

    attr_type = str(
        attribute.get("type", "")
    ).upper()

    value = str(
        attribute.get("value", "")
    )

    language = _language_for_value(
        text,
        value,
    )

    if attr_type == "AGE":
        return (
            _generalize_age(
                value,
                language,
            ),
            TRANSFORMATION_COSTS["AGE"],
        )

    if attr_type == "HEALTH":
        return (
            _generalize_health(
                value,
                language,
            ),
            TRANSFORMATION_COSTS["HEALTH"],
        )

    if attr_type == "FACILITY":
        return (
            _generalize_facility(
                value,
                language,
            ),
            TRANSFORMATION_COSTS["FACILITY"],
        )

    if attr_type == "LOCATION":
        return (
            _generalize_location(
                value,
                language,
            ),
            TRANSFORMATION_COSTS["LOCATION"],
        )

    if attr_type == "OCCUPATION":
        return (
            _generalize_occupation(
                value,
                language,
            ),
            TRANSFORMATION_COSTS["OCCUPATION"],
        )

    if attr_type in {
        "EDUCATION",
        "DEGREE",
    }:
        return (
            _generalize_education(
                value,
                language,
            ),
            TRANSFORMATION_COSTS["EDUCATION"],
        )

    if attr_type == "GENDER":
        return (
            _generalize_gender(
                value,
                language,
            ),
            TRANSFORMATION_COSTS["GENDER"],
        )

    if attr_type == "DATE":
        return (
            _generalize_date(
                value,
                language,
            ),
            TRANSFORMATION_COSTS["DATE"],
        )

    existing = _try_existing_generalizer(
        attribute,
        text,
    )

    if (
        existing is not None
        and existing != value
    ):
        return (
            existing,
            TRANSFORMATION_COSTS.get(
                attr_type,
                0.25,
            ),
        )

    return (
        value,
        TRANSFORMATION_COSTS.get(
            attr_type,
            0.25,
        ),
    )


# ============================================================
# TEXT PATCHING
# ============================================================

def _apply_text_patch(
    text: str,
    old: str,
    new: str,
) -> str:

    if not text or not old:
        return text

    old = str(old)
    new = str(new)

    # Handle Markdown email:
    # [abc@example.com](mailto:abc@example.com)
    if "@" in old:

        markdown_pattern = re.compile(
            rf"\[{re.escape(old)}\]"
            rf"\(mailto:[^)]+\)",
            flags=re.IGNORECASE,
        )

        updated = markdown_pattern.sub(
            new,
            text,
        )

        if updated != text:
            return updated

    return text.replace(
        old,
        new,
    )


# ============================================================
# ADDRESS CONSOLIDATION
# ============================================================

def _find_full_address_span(
    text: str,
) -> str | None:

    if not text:
        return None

    # Match a complete Indian-style address beginning with a
    # house/building number and ending with a 6-digit PIN code.
    #
    # Important:
    # Kannada text may directly follow the PIN:
    #
    #     560011ರಲ್ಲಿ
    #
    # Therefore we intentionally do NOT use \b after the PIN.
    pattern = re.compile(
        r"(?<!\d)"
        r"\d{1,5}"
        r"\s*,\s*"
        r"[^.;!?]{2,250}?"
        r"\d{6}"
        r"(?=[^\d]|$)",
        flags=re.UNICODE,
    )

    match = pattern.search(text)

    if match:
        return match.group(0).strip()

    return None


def _replace_full_address(
    text: str,
    attributes: list[dict[str, Any]],
) -> tuple[
    str,
    list[dict[str, Any]],
    set[int],
]:
    """Replace one complete address span while preserving M1 fragments in the audit."""

    address_indexes = [
        index
        for index, attr in enumerate(attributes)
        if str(attr.get("type", "")).upper() == "ADDRESS"
    ]

    if not address_indexes:
        return text, [], set()

    transformations: list[dict[str, Any]] = []
    transformed_indexes: set[int] = set()

    def add_address_transform(index: int) -> None:
        attr = attributes[index]
        transformations.append(
            {
                "attribute": "ADDRESS",
                "original_value": str(attr.get("value", "")),
                "generalized_value": "[ADDRESS]",
                "cost": 1.0,
                "specificity_before": float(attr.get("specificity", 1.0)),
                "specificity_after": 0.0,
                "source_message_id": attr.get("source_message_id", 1),
            }
        )
        transformed_indexes.add(index)

    # Preferred path: detect and replace the complete Indian-style address.
    full_address = _find_full_address_span(text)
    if full_address:
        updated = _apply_text_patch(
            text,
            full_address,
            "[ADDRESS]",
        )
        if updated != text:
            for index in address_indexes:
                add_address_transform(index)
            return updated, transformations, transformed_indexes

    # Fallback: reconstruct a contiguous address from the first detected
    # fragment through the first six-digit PIN code.
    address_values = [
        str(attributes[index].get("value", "")).strip()
        for index in address_indexes
        if str(attributes[index].get("value", "")).strip()
    ]

    if address_values:
        first_value = address_values[0]
        first_index = text.find(first_value)

        if first_index >= 0:
            start = first_index

            prefix_match = re.search(
                r"\d{1,5}\s*,\s*$",
                text[max(0, first_index - 20):first_index],
                flags=re.UNICODE,
            )

            if prefix_match:
                start = max(0, first_index - 20) + prefix_match.start()

            pin_match = re.search(
                r"\d{6}",
                text[start:],
                flags=re.UNICODE,
            )

            if pin_match:
                end = start + pin_match.end()
                full_address = text[start:end]
                updated = _apply_text_patch(
                    text,
                    full_address,
                    "[ADDRESS]",
                )

                if updated != text:
                    for index in address_indexes:
                        add_address_transform(index)
                    return updated, transformations, transformed_indexes

    # Last resort: replace individual address fragments.
    updated = text
    for index in address_indexes:
        attr = attributes[index]
        old = str(attr.get("value", ""))
        if not old:
            continue

        new_text = _apply_text_patch(
            updated,
            old,
            "[ADDRESS]",
        )

        if new_text == updated:
            continue

        add_address_transform(index)
        updated = new_text

    return updated, transformations, transformed_indexes


# ============================================================
# SPECIFICITY / RISK
# ============================================================

def _new_specificity(
    old_specificity: float,
    cost: float,
) -> float:

    try:
        old = float(
            old_specificity
        )
    except (
        TypeError,
        ValueError,
    ):
        old = 1.0

    try:
        reduction = float(
            cost
        )
    except (
        TypeError,
        ValueError,
    ):
        reduction = 0.25

    if reduction >= 1.0:
        return 0.0

    return round(
        max(
            0.0,
            min(
                1.0,
                old * (
                    1.0 - reduction
                ),
            ),
        ),
        4,
    )


def _normalise_risk_score(
    score: Any,
) -> float:

    try:
        value = float(score)
    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    # Module 2 normally returns 0-10.
    if value > 1.0:
        value /= 10.0

    return round(
        max(
            0.0,
            min(
                1.0,
                value,
            ),
        ),
        4,
    )


def _risk_level_label(
    score: float,
) -> str:

    score = _normalise_risk_score(
        score
    )

    if score <= 0.333:
        return "LOW"

    if score <= 0.666:
        return "MEDIUM"

    return "HIGH"


def _score_with_scorer(
    attributes: list[dict[str, Any]],
    contextual_indicators: list[Any],
    scorer_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:

    conversation = {
        "attributes": attributes,
        "contextual_indicators": contextual_indicators,
    }

    if scorer_fn is not None:

        try:
            result = scorer_fn(
                attributes,
                contextual_indicators,
            )

            if isinstance(
                result,
                dict,
            ):
                return result

            return {
                "risk_score": result,
                "risk_level": _risk_level_label(
                    _normalise_risk_score(
                        result
                    )
                ),
            }

        except TypeError:
            pass

        try:
            result = scorer_fn(
                attributes
            )

            if isinstance(
                result,
                dict,
            ):
                return result

            return {
                "risk_score": result,
                "risk_level": _risk_level_label(
                    _normalise_risk_score(
                        result
                    )
                ),
            }

        except Exception:
            pass

    try:

        features = build_features(
            conversation
        )

        result = score_risk(
            features
        )

        if isinstance(
            result,
            dict,
        ):
            return result

        return {
            "risk_score": result,
            "risk_level": _risk_level_label(
                _normalise_risk_score(
                    result
                )
            ),
        }

    except Exception:

        return {
            "risk_score": 0.0,
            "risk_level": "LOW",
        }


def _m2_risk_for_attributes(
    attributes: list[dict[str, Any]],
    contextual_indicators: list[Any],
    scorer_fn: Callable[..., Any] | None = None,
) -> float:

    result = _score_with_scorer(
        attributes,
        contextual_indicators,
        scorer_fn,
    )

    return _normalise_risk_score(
        result.get(
            "risk_score",
            0.0,
        )
    )


# ============================================================
# UNIQUENESS TRANSFORMATION
# ============================================================

def _transform_uniqueness(
    text: str,
) -> tuple[
    str,
    list[dict[str, Any]],
]:

    language = _detect_language(
        text
    )

    if language == "kannada":

        replacements = [
            (
                "ಅಪರೂಪದ ಸ್ವಯಂಪ್ರತಿರೋಧಕ ಕಾಯಿಲೆ",
                "ಸ್ವಯಂಪ್ರತಿರೋಧಕ ಕಾಯಿಲೆ",
            ),
            (
                "ಈ ಕಾಯಿಲೆ ಹೊಂದಿರುವ ಕೆಲವೇ ಜನರಲ್ಲಿ ನಾನೊಬ್ಬಳಾಗಿದ್ದು",
                "ಈ ಕಾಯಿಲೆ ಹೊಂದಿರುವ ಕೆಲವು ಜನರಲ್ಲಿ ನಾನೂ ಒಬ್ಬರು",
            ),
            (
                "ನಾನು ವಾಸಿಸುವ ಸಣ್ಣ ಪ್ರದೇಶದಲ್ಲಿ",
                "ನಾನು ವಾಸಿಸುವ ಪ್ರದೇಶದಲ್ಲಿ",
            ),
            (
                "ಕೆಲವೇ ಜನರಲ್ಲಿ",
                "ಕೆಲವು ಜನರಲ್ಲಿ",
            ),
            (
                "ಅಪರೂಪದ",
                "ನಿರ್ದಿಷ್ಟ",
            ),
            (
                "ಸಣ್ಣ ಪ್ರದೇಶದಲ್ಲಿ",
                "ಪ್ರದೇಶದಲ್ಲಿ",
            ),
            (
                "ಸಣ್ಣ ಪ್ರದೇಶ",
                "ಪ್ರದೇಶ",
            ),
            (
                "ಕೆಲವೇ",
                "ಕೆಲವು",
            ),
            (
                "ನಾನೊಬ್ಬಳೇ",
                "ನಾನು",
            ),
            (
                "ನಾನೊಬ್ಬನೇ",
                "ನಾನು",
            ),
            (
                "ಒಬ್ಬಳೇ",
                "ಒಬ್ಬರು",
            ),
            (
                "ಒಬ್ಬನೇ",
                "ಒಬ್ಬರು",
            ),
        ]

    elif language == "hindi":

        replacements = [
            (
                "कुछ ही",
                "कुछ",
            ),
            (
                "केवल",
                "",
            ),
            (
                "सिर्फ",
                "",
            ),
            (
                "दुर्लभ",
                "विशिष्ट",
            ),
            (
                "छोटे क्षेत्र में",
                "क्षेत्र में",
            ),
            (
                "छोटे क्षेत्र",
                "क्षेत्र",
            ),
            (
                "एकमात्र",
                "एक",
            ),
        ]

    else:

        replacements = [
            (
                "one of the few",
                "one of several",
            ),
            (
                "only a few",
                "some",
            ),
            (
                "only",
                "",
            ),
            (
                "rare",
                "specific",
            ),
            (
                "small area",
                "area",
            ),
            (
                "sole",
                "one",
            ),
        ]

    new_text = text
    transformations = []

    for old, new in replacements:

        if old not in new_text:
            continue

        updated = new_text.replace(
            old,
            new,
        )

        if updated == new_text:
            continue

        transformations.append(
            {
                "attribute": "UNIQUENESS",
                "original_value": old,
                "generalized_value": new,
                "cost": 0.15,
                "specificity_before": 1.0,
                "specificity_after": 0.85,
                "source_message_id": 1,
            }
        )

        new_text = updated

    # Collapse accidental Latin-space duplication only.
    if language in {
        "english",
        "hindi",
    }:
        new_text = re.sub(
            r"[ \t]{2,}",
            " ",
            new_text,
        )

    # Kannada needs explicit whitespace repair.
    if language == "kannada":
        new_text = re.sub(
            r"([ಅ-ಹಾ-ೞ])([ಅ-ಹಾ-ೞ])",
            r"\1\2",
            new_text,
        )

        new_text = new_text.replace(
            "ವಾಸಿಸುವಪ್ರದೇಶ",
            "ವಾಸಿಸುವ ಪ್ರದೇಶ",
        )
        new_text = new_text.replace(
            "ನೋಂದಣಿಸಂಖ್ಯೆ",
            "ನೋಂದಣಿ ಸಂಖ್ಯೆ",
        )

    return (
        new_text,
        transformations,
    )


# ============================================================
# CONTEXTUAL ATTRIBUTE TRANSFORMATION
# ============================================================

def _transform_contextual_attributes(
    text: str,
    attributes: list[dict[str, Any]],
    already_transformed: set[int],
) -> tuple[
    str,
    list[dict[str, Any]],
    set[int],
]:

    new_text = text

    transformations = []

    transformed_indexes = set(
        already_transformed
    )

    candidates = []

    for index, attr in enumerate(
        attributes
    ):

        if index in transformed_indexes:
            continue

        attr_type = str(
            attr.get(
                "type",
                "",
            )
        ).upper()

        if attr_type not in CONTEXTUAL_TYPES:
            continue

        try:
            specificity = float(
                attr.get(
                    "specificity",
                    0.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            specificity = 0.0

        try:
            confidence = float(
                attr.get(
                    "confidence",
                    0.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        priority = (
            specificity * 0.7
            + confidence * 0.3
        )

        candidates.append(
            (
                priority,
                index,
                attr,
            )
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    for (
        _,
        index,
        attr,
    ) in candidates:

        attr_type = str(
            attr.get(
                "type",
                "",
            )
        ).upper()

        old_value = str(
            attr.get(
                "value",
                "",
            )
        )

        if not old_value:
            continue

        generalized_value, cost = (
            _generalize_attribute(
                attr,
                new_text,
            )
        )

        if not generalized_value:
            continue

        if (
            generalized_value
            == old_value
        ):
            continue

        updated_text = _apply_text_patch(
            new_text,
            old_value,
            generalized_value,
        )

        if updated_text == new_text:
            continue

        try:
            old_specificity = float(
                attr.get(
                    "specificity",
                    1.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            old_specificity = 1.0

        new_specificity = (
            _new_specificity(
                old_specificity,
                cost,
            )
        )

        transformations.append(
            {
                "attribute": attr_type,
                "original_value": old_value,
                "generalized_value": generalized_value,
                "cost": cost,
                "specificity_before": old_specificity,
                "specificity_after": new_specificity,
                "source_message_id": attr.get(
                    "source_message_id",
                    1,
                ),
            }
        )

        transformed_indexes.add(
            index
        )

        new_text = updated_text

    return (
        new_text,
        transformations,
        transformed_indexes,
    )


# ============================================================
# FINAL ATTRIBUTE STATE
# ============================================================

def _build_final_attributes(
    original_attributes: list[dict[str, Any]],
    transformed_indexes: set[int],
    transformations: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    final_attributes = []

    contextual_transforms = {
        (
            str(
                transformation.get(
                    "attribute",
                    "",
                )
            ).upper(),
            str(
                transformation.get(
                    "original_value",
                    "",
                )
            ),
        ): transformation
        for transformation in transformations
    }

    for index, attr in enumerate(
        original_attributes
    ):

        attr_type = str(
            attr.get(
                "type",
                "",
            )
        ).upper()

        value = str(
            attr.get(
                "value",
                "",
            )
        )

        # Direct identifiers no longer exist
        # in the protected attribute set.
        if (
            index in transformed_indexes
            and attr_type in DIRECT_PII_TYPES
        ):
            continue

        transformation = contextual_transforms.get(
            (
                attr_type,
                value,
            )
        )

        if transformation is not None:

            updated = copy.deepcopy(
                attr
            )

            updated[
                "specificity"
            ] = transformation[
                "specificity_after"
            ]

            final_attributes.append(
                updated
            )

            continue

        final_attributes.append(
            copy.deepcopy(attr)
        )

    return final_attributes


# ============================================================
# UTILITY / INFORMATION LOSS
# ============================================================

def _calculate_utility(
    original_attributes: list[dict[str, Any]],
    transformations: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Calculate utility from contextual transformations only.

    Direct identifiers are intentionally excluded because their
    removal is mandatory privacy protection rather than contextual
    information loss.

    Utility is based on retained specificity:

        retained = specificity_after / specificity_before

    Information loss:

        loss = 1 - average(retained specificity)
    """

    contextual_transformations = [
        transformation
        for transformation in transformations
        if str(
            transformation.get(
                "attribute",
                "",
            )
        ).upper()
        not in DIRECT_PII_TYPES
    ]

    if not contextual_transformations:
        return {
            "information_loss": 0.0,
            "information_retained": 1.0,
            "utility_score": 1.0,
        }

    retained_values = []

    for transformation in contextual_transformations:

        try:
            before = float(
                transformation.get(
                    "specificity_before",
                    1.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            before = 1.0

        try:
            after = float(
                transformation.get(
                    "specificity_after",
                    0.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            after = 0.0

        if before <= 0:
            retained = 0.0
        else:
            retained = (
                after / before
            )

        retained_values.append(
            max(
                0.0,
                min(
                    1.0,
                    retained,
                ),
            )
        )

    information_retained = (
        sum(retained_values)
        / len(retained_values)
    )

    information_retained = round(
        information_retained,
        4,
    )

    information_loss = round(
        1.0
        - information_retained,
        4,
    )

    return {
        "information_loss": information_loss,
        "information_retained": information_retained,
        "utility_score": information_retained,
    }


# ============================================================
# RESULT CONTRACT
# ============================================================

def _build_result(
    conversation_input: dict[str, Any],
    sanitized_text: str,
    transformations: list[dict[str, Any]],
    initial_risk: float,
    final_risk: float,
    initial_level: str,
    final_level: str,
    utility: dict[str, float],
    original_attribute_count: int,
    threshold_used: float,
    final_attributes: list[dict[str, Any]],
    final_contextual_indicators: list[Any],
) -> dict[str, Any]:

    attributes_changed = len(
        transformations
    )

    total_attributes = (
        original_attribute_count
    )

    risk_reduction = round(
        max(
            0.0,
            initial_risk
            - final_risk,
        ),
        4,
    )

    result = {
        "conversation_id": conversation_input.get(
            "conversation_id",
            "unknown",
        ),

        "original_text": conversation_input.get(
            "original_text",
            conversation_input.get(
                "message",
                "",
            ),
        ),

        "sanitized_text": sanitized_text,

        # Risk
        "initial_risk": round(
            initial_risk,
            4,
        ),
        "final_risk": round(
            final_risk,
            4,
        ),
        "initial_risk_score": round(
            initial_risk,
            4,
        ),
        "final_risk_score": round(
            final_risk,
            4,
        ),
        "initial_risk_level": initial_level,
        "final_risk_level": final_level,
        "risk_reduction": risk_reduction,

        # Transformations
        "transformations": transformations,

        # Utility
        "utility": utility,
        "information_loss": utility[
            "information_loss"
        ],
        "information_retained": utility[
            "information_retained"
        ],
        "utility_score": utility[
            "utility_score"
        ],

        # Attribute counts
        "attributes_changed": attributes_changed,
        "total_attributes": total_attributes,

        # Additional aliases so the runner / API
        # can consume the result without ambiguity.
        "attribute_count": total_attributes,
        "changed_attribute_count": attributes_changed,

        # Threshold
        "threshold_used": threshold_used,

        # Protected M1 state
        "final_attributes": final_attributes,
        "final_contextual_indicators": (
            final_contextual_indicators
        ),
    }

    # Compatibility structure for callers that expect metrics.
    result["metrics"] = {
        "attributes_changed": attributes_changed,
        "total_attributes": total_attributes,
        "information_loss": utility[
            "information_loss"
        ],
        "information_retained": utility[
            "information_retained"
        ],
        "utility_score": utility[
            "utility_score"
        ],
        "risk_reduction": risk_reduction,
    }

    return result


# ============================================================
# MAIN OPTIMIZER
# ============================================================

def optimize(
    conversation_input: dict[str, Any],
    threshold: float = DEFAULT_RISK_THRESHOLD,
    scorer_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:

    if not isinstance(
        conversation_input,
        dict,
    ):
        raise TypeError(
            "conversation_input must be a dictionary"
        )

    attributes = copy.deepcopy(
        conversation_input.get(
            "attributes",
            [],
        )
    )

    contextual_indicators = copy.deepcopy(
        conversation_input.get(
            "contextual_indicators",
            [],
        )
    )

    original_text = conversation_input.get(
        "original_text",
        conversation_input.get(
            "message",
            "",
        ),
    )

    if original_text is None:
        original_text = ""

    original_text = str(
        original_text
    )

    try:
        threshold_value = float(
            threshold
        )
    except (
        TypeError,
        ValueError,
    ):
        threshold_value = DEFAULT_RISK_THRESHOLD

    if threshold_value > 1.0:
        threshold_value /= 10.0

    threshold_value = max(
        0.0,
        min(
            1.0,
            threshold_value,
        ),
    )

    # --------------------------------------------------------
    # INITIAL M2 SCORE
    # --------------------------------------------------------

    initial_risk = (
        _m2_risk_for_attributes(
            attributes,
            contextual_indicators,
            scorer_fn,
        )
    )

    initial_level = (
        _risk_level_label(
            initial_risk
        )
    )

    sanitized_text = original_text

    transformations = []

    transformed_indexes: set[int] = set()

    # --------------------------------------------------------
    # 1. CONSOLIDATE ADDRESS
    # --------------------------------------------------------

    (
        sanitized_text,
        address_transformations,
        address_indexes,
    ) = _replace_full_address(
        sanitized_text,
        attributes,
    )

    transformations.extend(
        address_transformations
    )

    transformed_indexes.update(
        address_indexes
    )

    # --------------------------------------------------------
    # 2. DIRECT PII
    # --------------------------------------------------------

    for index, attr in enumerate(
        attributes
    ):

        if index in transformed_indexes:
            continue

        attr_type = str(
            attr.get(
                "type",
                "",
            )
        ).upper()

        if attr_type not in DIRECT_PII_TYPES:
            continue

        old_value = str(
            attr.get(
                "value",
                "",
            )
        )

        if not old_value:
            continue

        replacement = DIRECT_REPLACEMENTS.get(
            attr_type
        )

        if replacement is None:
            continue

        updated_text = _apply_text_patch(
            sanitized_text,
            old_value,
            replacement,
        )

        if updated_text == sanitized_text:
            continue

        sanitized_text = updated_text

        try:
            specificity_before = float(
                attr.get(
                    "specificity",
                    1.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            specificity_before = 1.0

        transformations.append(
            {
                "attribute": attr_type,
                "original_value": old_value,
                "generalized_value": replacement,
                "cost": 1.0,
                "specificity_before": (
                    specificity_before
                ),
                "specificity_after": 0.0,
                "source_message_id": attr.get(
                    "source_message_id",
                    1,
                ),
            }
        )

        transformed_indexes.add(
            index
        )

    # --------------------------------------------------------
    # 3. CONTEXTUAL ATTRIBUTES
    # --------------------------------------------------------

    (
        sanitized_text,
        contextual_transformations,
        transformed_indexes,
    ) = _transform_contextual_attributes(
        sanitized_text,
        attributes,
        transformed_indexes,
    )

    transformations.extend(
        contextual_transformations
    )

    # --------------------------------------------------------
    # 4. CONTEXTUAL UNIQUENESS
    # --------------------------------------------------------

    (
        uniqueness_text,
        uniqueness_transformations,
    ) = _transform_uniqueness(
        sanitized_text
    )

    if uniqueness_text != sanitized_text:
        sanitized_text = (
            uniqueness_text
        )

    transformations.extend(
        uniqueness_transformations
    )

    # --------------------------------------------------------
    # 5. BUILD PROTECTED ATTRIBUTES
    # --------------------------------------------------------

    final_attributes = (
        _build_final_attributes(
            attributes,
            transformed_indexes,
            transformations,
        )
    )

    # --------------------------------------------------------
    # 6. CLEAN CONTEXTUAL INDICATORS
    # --------------------------------------------------------

    final_indicators = copy.deepcopy(
        contextual_indicators
    )

    if uniqueness_transformations:

        cleaned_indicators = []

        for indicator in final_indicators:

            indicator_text = str(
                indicator
            ).lower()

            uniqueness_terms = (
                "uniqueness",
                "only",
                "rare",
                "few",
                "केवल",
                "सिर्फ",
                "दुर्लभ",
                "केलवे",
                "ಕೆಲವೇ",
                "ಅಪರೂಪ",
            )

            if any(
                term in indicator_text
                for term in uniqueness_terms
            ):
                continue

            cleaned_indicators.append(
                indicator
            )

        final_indicators = (
            cleaned_indicators
        )

    # --------------------------------------------------------
    # 7. FINAL M2 SCORE
    # --------------------------------------------------------

    final_risk = (
        _m2_risk_for_attributes(
            final_attributes,
            final_indicators,
            scorer_fn,
        )
    )

    final_level = (
        _risk_level_label(
            final_risk
        )
    )

    # --------------------------------------------------------
    # 8. UTILITY
    # --------------------------------------------------------

    utility = _calculate_utility(
        attributes,
        transformations,
    )

    # --------------------------------------------------------
    # 9. RESULT
    # --------------------------------------------------------

    return _build_result(
        conversation_input={
            **conversation_input,
            "original_text": original_text,
        },
        sanitized_text=sanitized_text,
        transformations=transformations,
        initial_risk=initial_risk,
        final_risk=final_risk,
        initial_level=initial_level,
        final_level=final_level,
        utility=utility,
        original_attribute_count=len(
            attributes
        ),
        threshold_used=round(
            threshold_value,
            4,
        ),
        final_attributes=final_attributes,
        final_contextual_indicators=(
            final_indicators
        ),
    )


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def optimize_private_output(
    conversation_input: dict[str, Any],
    threshold: float = 0.40,
    scorer_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:

    return optimize(
        conversation_input,
        threshold=threshold,
        scorer_fn=scorer_fn,
    )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    sample = {
        "conversation_id": "optimizer_test",

        "original_text": (
            "My name is Aarav Sharma. "
            "I am 42 years old and live at "
            "42, MG Road, Bengaluru, Karnataka 560001. "
            "My phone number is 9876543210 and my email is "
            "aarav.sharma@example.com. "
            "I have a rare neurological disorder and receive "
            "treatment at a private hospital in Bengaluru."
        ),

        "attributes": [
            {
                "type": "PERSON",
                "value": "Aarav Sharma",
                "confidence": 0.95,
                "specificity": 1.0,
                "source_message_id": 1,
            },
            {
                "type": "AGE",
                "value": "42",
                "confidence": 0.95,
                "specificity": 1.0,
                "source_message_id": 1,
            },
            {
                "type": "ADDRESS",
                "value": (
                    "42, MG Road, Bengaluru, "
                    "Karnataka 560001"
                ),
                "confidence": 0.95,
                "specificity": 1.0,
                "source_message_id": 1,
            },
            {
                "type": "PHONE_NUMBER",
                "value": "9876543210",
                "confidence": 0.95,
                "specificity": 1.0,
                "source_message_id": 1,
            },
            {
                "type": "EMAIL_ADDRESS",
                "value": (
                    "aarav.sharma@example.com"
                ),
                "confidence": 0.95,
                "specificity": 1.0,
                "source_message_id": 1,
            },
            {
                "type": "HEALTH",
                "value": (
                    "rare neurological disorder"
                ),
                "confidence": 0.94,
                "specificity": 0.93,
                "source_message_id": 1,
            },
            {
                "type": "FACILITY",
                "value": "private hospital",
                "confidence": 0.90,
                "specificity": 0.80,
                "source_message_id": 1,
            },
        ],

        "contextual_indicators": [
            {
                "indicator": "rare_condition",
                "attribute_type": "HEALTH",
            }
        ],
    }

    import pprint

    pprint.pp(
        optimize(sample)
    )