"""Normalize detector and extractor results into the M1 schema."""

from __future__ import annotations

import re
from typing import Any

# Settlement/location words that indicate a value is a PLACE, not an occupation.
# Any GLiNER prediction labelled OCCUPATION whose value contains one of these
# words is dropped in normalize_attributes() as a false positive.
_OCCUPATION_LOCATION_BLOCKLIST = re.compile(
    r'\b(?:village|town|gram|tehsil|taluk|panchayat|gaon|mohalla)\b',
    re.IGNORECASE,
)


GARBAGE_TOKENS = {
    # Hindi auxiliary verbs & copulas
    "hai", "hain", "hoon", "hun", "ho", "tha", "thi", "the", "hoga", "hogi", "hoge",
    # Hindi postpositions & prepositions
    "mein", "me", "ke", "ki", "ka", "ko", "se", "par", "pe", "tak",
    # Hindi verbs / participles
    "liye", "jaana", "jana", "gaya", "gayi", "gaye", "rehta", "rehti", "rehte",
    "karna", "karta", "karti", "karte", "karo", "kare", "raha", "rahi", "rahe",
    # Hindi conjunctions & particles
    "aur", "ya", "bhi", "toh", "to", "hi", "na",
    # Hindi pronouns
    "main", "mera", "meri", "mere", "mujhe", "mujhko", "hum", "tum", "aap", "yeh", "woh",
    # Common English function words that appear in garbage fragments
    "and", "or", "in", "on", "at", "to", "for", "with", "from", "of", "is", "am", "are", "was", "were", "the", "a", "an",
}


def is_garbage_span(text: str) -> bool:
    """Identify if a span consists entirely of stop-words / grammatical particles."""
    if not text or not text.strip():
        return True
    if re.search(r"\d", text):
        return False
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not tokens:
        return True
    return all(token in GARBAGE_TOKENS for token in tokens)


def _attribute_key(attribute: dict[str, Any]) -> tuple[str, str, object]:
    normalized_type = str(attribute["type"]).upper().strip()
    normalized_value = " ".join(str(attribute["value"]).split()).casefold()
    return normalized_type, normalized_value, attribute["source_message_id"]


def normalize_attributes(attributes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    seen = set()
    for attribute in attributes:
        val = " ".join(str(attribute["value"]).split()).strip()
        if is_garbage_span(val):
            continue
        # Drop location-like phrases misclassified as OCCUPATION by GLiNER.
        # e.g. "village x", "town y" are places, not jobs.
        attr_type_raw = str(attribute["type"]).upper().strip()
        if attr_type_raw == "OCCUPATION" and _OCCUPATION_LOCATION_BLOCKLIST.search(val):
            continue
        normalized_attribute = {
            "type": str(attribute["type"]).upper().strip(),
            "value": val,
            "confidence": round(float(attribute["confidence"]), 2),
            "specificity": round(float(attribute["specificity"]), 2),
            "source_message_id": attribute["source_message_id"],
        }
        normalized_attribute["confidence"] = min(1.0, max(0.0, normalized_attribute["confidence"]))
        normalized_attribute["specificity"] = min(1.0, max(0.0, normalized_attribute["specificity"]))
        key = _attribute_key(normalized_attribute)
        if key not in seen:
            seen.add(key)
            normalized.append(normalized_attribute)
    return normalized


def normalize_indicators(indicators: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized = []
    seen = set()
    for indicator in indicators:
        normalized_indicator = {
            "indicator": str(indicator["indicator"]).strip(),
            "attribute_type": str(indicator["attribute_type"]).upper().strip(),
        }
        key = (normalized_indicator["indicator"], normalized_indicator["attribute_type"])
        if key not in seen:
            seen.add(key)
            normalized.append(normalized_indicator)
    return normalized