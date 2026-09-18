"""Contextual and sensitive attribute extraction for Module 1."""

from __future__ import annotations

import re
from typing import Any

from gliner import GLiNER
from backend.modules.nlp.normalizer import is_garbage_span


GLINER_MODEL_NAME = "urchade/gliner_small-v2.1"
GLINER_LABELS = [
    "GENDER",
    "HEALTH",
    "OCCUPATION",
    "FACILITY",
    "RELIGION",
    "CASTE",
    "EDUCATION",
]

# Specificity is a deterministic M1 policy value, not a GLiNER probability.
# It represents how identifying a detection category is for downstream stages.
SPECIFICITY_BY_TYPE = {
    "GENDER": 0.70,
    "HEALTH": 0.92,
    "OCCUPATION": 0.75,
    "FACILITY": 0.98,
    "RELIGION": 0.80,
    "CASTE": 0.90,
    "EDUCATION": 0.60,
}

AGE_PATTERN = re.compile(
    r"\b(?:"
    r"(\d{1,3})\s*(?:-|–)?\s*(?:years?\s*old|yrs?\s*old|year-old|y/o)"
    r"|(?:(?:meri|mera|my)\s+)?(?:age|umar|aayu)\s*(?:is|hai|h)?\s*[:=-]?\s*(\d{1,3})"
    r"|(\d{1,3})\s*(?:years?|yrs?|saal)\s+hai"
    r"|(?:i\s*am|i'm|main)\s+(?:the\s+only\s+)?(\d{1,3})\b(?!\s*%)"
    r")\b",
    re.IGNORECASE,
)
AGE_RANGE_PATTERN = re.compile(
    r"\b(teen(?:ager)?|in (?:my|her|his) (?:twenties|thirties|forties|fifties|sixties)|elderly|middle-aged|young adult)\b",
    re.IGNORECASE,
)
UNIQUENESS_WORDS = ("only", "first", "last", "youngest", "eldest", "sole", "single")
# Characters after the uniqueness word within which an attribute value is
# considered to be semantically modified by that word.
_UNIQUENESS_PROXIMITY_WINDOW = 35
OCCUPATION_PHRASES = {
    "school teacher",
    "teacher",
    "software engineer",
    "engineer",
    "doctor",
    "nurse",
    "farmer",
    "driver",
    "student",
    "police officer",
    "data analyst",
    "analyst",
}
OCCUPATION_INDICATORS = re.compile(
    r"\b(?:work(?:s|ed|ing)?\s+as|job\s+as|employed\s+as|profession\s+(?:is|as)|occupation\s+(?:is|as)|career\s+as)\s+(?:a|an)?\s*",
    re.IGNORECASE,
)

_gliner_model: GLiNER | None = None


def _get_gliner_model() -> GLiNER:
    global _gliner_model
    if _gliner_model is None:
        _gliner_model = GLiNER.from_pretrained(GLINER_MODEL_NAME)
    return _gliner_model


def _find_age(text: str) -> list[dict[str, Any]]:
    found = []
    seen = set()
    for match in AGE_PATTERN.finditer(text):
        val = next((g for g in match.groups() if g is not None), None)
        if val and val not in seen:
            seen.add(val)
            found.append({"value": val, "specificity": 1.0, "indicator": "exact_age"})
    for match in AGE_RANGE_PATTERN.finditer(text):
        found.append({"value": match.group(0), "specificity": 0.4, "indicator": "age_range"})
    return found


def _find_gliner_attributes(text: str, language_context: dict[str, Any]) -> list[dict[str, Any]]:
    """Run GLiNER for contextual attributes only.

    MuRIL context is supplied here so extraction has access to language-aware
    processing metadata. GLiNER remains responsible only for its labels and
    is not used for direct PII.
    """
    del language_context
    predictions = _get_gliner_model().predict_entities(
        text,
        GLINER_LABELS,
        threshold=0.4,
    )
    attributes = []
    for prediction in predictions:
        val = prediction["text"].strip()
        if is_garbage_span(val):
            continue

        attribute_type = prediction["label"].upper()
        val_lower = val.lower()
        is_known_occupation = val_lower in OCCUPATION_PHRASES or any(
            val_lower.endswith(" " + term) for term in OCCUPATION_PHRASES
        )
        has_job_context = bool(
            OCCUPATION_INDICATORS.search(text)
            and re.search(
                rf"\b(?:work(?:s|ed|ing)?\s+as|job\s+as|employed\s+as)\s+(?:a|an)?\s+{re.escape(val)}\b",
                text,
                re.IGNORECASE,
            )
        )

        if attribute_type == "EDUCATION" and (is_known_occupation or has_job_context):
            attribute_type = "OCCUPATION"
        elif is_known_occupation and has_job_context:
            attribute_type = "OCCUPATION"
        if attribute_type not in SPECIFICITY_BY_TYPE:
            continue
        attributes.append({
            "type": attribute_type,
            "value": val,
            "confidence": round(float(prediction["score"]), 2),
            "specificity": SPECIFICITY_BY_TYPE[attribute_type],
        })
    return attributes


def _find_uniqueness_indicators(text: str) -> list[str]:
    lower = text.lower()
    hits = []
    for word in UNIQUENESS_WORDS:
        if word == "last" and re.search(r"\blast\b(?!\s+(year|month|week|night|time))", lower):
            hits.append(word)
        elif word != "last" and re.search(rf"\b{word}\b", lower):
            hits.append(word)
    return hits


def _uniqueness_targets(text: str, msg_attributes: list[dict]) -> set[str]:
    """
    Return the set of attribute types that the uniqueness word(s) in *text*
    semantically modify, using character-position proximity.

    Strategy
    --------
    For each uniqueness word found in *text*, collect attribute values that
    appear within ``_UNIQUENESS_PROXIMITY_WINDOW`` characters **after** the
    word.  Only those attribute types receive the ``uniqueness_word`` indicator.
    If no attribute falls within the window, fall back to the single nearest
    attribute after the word (so the indicator is never silently dropped).

    This prevents a single "only" at the start of a sentence from propagating
    to every detected attribute regardless of semantic relevance.
    """
    if not msg_attributes:
        return set()

    lower = text.lower()
    targeted: set[str] = set()

    for word in UNIQUENESS_WORDS:
        for match in re.finditer(rf"\b{re.escape(word)}\b", lower):
            # Apply existing "last" guard ("last year" is not a uniqueness word)
            if word == "last":
                context = lower[match.start(): match.start() + 25]
                if re.search(r"\blast\s+(?:year|month|week|night|time)\b", context):
                    continue

            word_end = match.end()
            in_window: list[tuple[int, str]] = []  # (distance, attr_type)

            for attr in msg_attributes:
                val_lower = str(attr["value"]).lower()
                # Search for the value starting at (or just before) the word end
                # to handle values that begin right at the uniqueness word boundary.
                pos = lower.find(val_lower, max(0, word_end - 3))
                if pos == -1:
                    pos = lower.find(val_lower)
                if pos != -1:
                    dist = pos - word_end
                    if 0 <= dist <= _UNIQUENESS_PROXIMITY_WINDOW:
                        in_window.append((dist, attr["type"]))

            if in_window:
                for _, attr_type in in_window:
                    targeted.add(attr_type)
            else:
                # Fallback: assign to the nearest attribute anywhere after the word
                best_type: str | None = None
                best_dist: float = float("inf")
                for attr in msg_attributes:
                    val_lower = str(attr["value"]).lower()
                    pos = lower.find(val_lower, word_end)
                    if pos != -1:
                        dist = float(pos - word_end)
                        if dist < best_dist:
                            best_dist = dist
                            best_type = attr["type"]
                if best_type:
                    targeted.add(best_type)

    return targeted


def extract_sensitive_data(
    messages: list[dict],
    language_contexts: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Extract contextual attributes and supported contextual indicators."""
    attributes = []
    contextual_indicators = []
    contexts = language_contexts or [{} for _ in messages]

    for message, language_context in zip(messages, contexts):
        message_id = message["id"]
        text = message["text"]

        for age in _find_age(text):
            attributes.append({
                "type": "AGE",
                "value": age["value"],
                "confidence": 0.98,
                "specificity": age["specificity"],
                "source_message_id": message_id,
            })
            contextual_indicators.append({
                "indicator": age["indicator"],
                "attribute_type": "AGE",
            })

        contextual_attributes = _find_gliner_attributes(text, language_context)
        for attribute in contextual_attributes:
            attribute["source_message_id"] = message_id
            attributes.append(attribute)

        if _find_uniqueness_indicators(text):
            msg_attrs = [
                a for a in attributes if a["source_message_id"] == message_id
            ]
            if msg_attrs:
                targeted_types = _uniqueness_targets(text, msg_attrs)
                contextual_indicators.extend(
                    {"indicator": "uniqueness_word", "attribute_type": attr_type}
                    for attr_type in sorted(targeted_types)
                )

    return {
        "attributes": attributes,
        "contextual_indicators": contextual_indicators,
    }
