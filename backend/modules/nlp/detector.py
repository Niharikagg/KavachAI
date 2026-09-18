"""
detector.py — Explicit PII detection module.

Detects direct PII with Presidio and deterministic regexes. Contextual
attributes belong to extractor.py and are not detected here.
"""

import re

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from backend.modules.nlp.normalizer import is_garbage_span

# --- Custom recognizers for Indian IDs ---

aadhaar_pattern = Pattern(
    name="aadhaar_pattern",
    regex=r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    score=0.85,
)
aadhaar_recognizer = PatternRecognizer(
    supported_entity="AADHAAR",
    patterns=[aadhaar_pattern],
    context=["aadhaar", "aadhar", "uidai"],
)

pan_pattern = Pattern(
    name="pan_pattern",
    regex=r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
    score=0.85,
)
pan_recognizer = PatternRecognizer(
    supported_entity="PAN",
    patterns=[pan_pattern],
    context=["pan", "pan card", "income tax"],
)

CUSTOM_RECOGNIZERS = [
    aadhaar_recognizer,
    pan_recognizer,
]

# These patterns are also run explicitly so identifiers remain detectable
# when Presidio's contextual recognizer does not receive a nearby keyword.
REGEX_PATTERNS = {
    "PHONE_NUMBER": re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"),
    "EMAIL_ADDRESS": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "AADHAAR": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "PAN": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b"),
    "PINCODE": re.compile(
        r"\b(?:pin\s*code|pincode|postal\s*code)\s*[:#-]?\s*(\d{6})\b|(?<!\d)(\d{6})(?!\d)",
        re.IGNORECASE,
    ),
    "IFSC": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    "BANK_ACCOUNT": re.compile(
        r"\b(?:account|a/c|bank\s+account)\s*(?:number|no\.?|#)?\s*[:=-]?\s*(\d{9,18})\b",
        re.IGNORECASE,
    ),
    "VEHICLE_NUMBER": re.compile(
        r"\b[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4}\b",
        re.IGNORECASE,
    ),
    "VOTER_ID": re.compile(r"\b[A-Z]{3}\d{7}\b"),
    "DATE": re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+(?:[1-9]|[12]\d|3[01]),\s+\d{4}\b",
        re.IGNORECASE,
    ),
    "LOCATION": re.compile(r"\bnear\s+([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*)\b"),
}

# Confidence and specificity for deterministic patterns are M1 policy values.
REGEX_CONFIDENCE = {"DATE": 0.99, "LOCATION": 0.94}
REGEX_SPECIFICITY = {"DATE": 1.0, "LOCATION": 0.85}

# --- Build the analyzer once at module load ---

analyzer = AnalyzerEngine()
for recognizer in CUSTOM_RECOGNIZERS:
    analyzer.registry.add_recognizer(recognizer)

# Entities we care about for this project (explicit PII layer only —
# sensitive/contextual attributes like health/age are handled in extractor.py)
TARGET_ENTITIES = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "LOCATION",
    "AADHAAR",
    "PAN",
    "PINCODE",
    "IFSC",
    "BANK_ACCOUNT",
    "VEHICLE_NUMBER",
    "VOTER_ID",
]

# Words that Presidio's PERSON model sometimes misfires on
PERSON_FALSE_POSITIVES = {"aadhaar", "aadhar", "pan", "uidai"}


def _regex_entities(text: str) -> list[dict[str, object]]:
    detected = []
    for entity_type, pattern in REGEX_PATTERNS.items():
        for match in pattern.finditer(text):
            value = match.group(0).strip()
            if match.groups():
                captured = next((g for g in match.groups() if g is not None), None)
                if captured is not None:
                    value = captured.strip()
            if entity_type == "PINCODE" and not re.search(r"(?:pin|postal)", text[max(0, match.start() - 12):match.end()], re.IGNORECASE):
                continue
            if is_garbage_span(value):
                continue
            detected.append({
                "type": entity_type,
                "value": value,
                "confidence": REGEX_CONFIDENCE.get(entity_type, 0.95),
                "specificity": REGEX_SPECIFICITY.get(entity_type, 1.0),
            })
    return detected


def detect_entities(text: str) -> list[dict[str, object]]:
    """
    Detect explicit PII entities in a piece of text.

    Args:
        text: Raw citizen message text.

    Returns:
        A list of dicts, each describing one detected entity:
        [
            {"type": "PHONE_NUMBER", "value": "9876543210", "confidence": "0.90"},
            ...
        ]
    """
    if not text or not text.strip():
        return []

    results = analyzer.analyze(
        text=text,
        entities=TARGET_ENTITIES,
        language="en",
    )

    detected = _regex_entities(text)
    for r in results:
        value = text[r.start:r.end]
        if r.entity_type == "PERSON" and (value.lower() in PERSON_FALSE_POSITIVES or is_garbage_span(value)):
            continue  # skip known false positives
        if is_garbage_span(value):
            continue
        detected.append({
            "type": r.entity_type,
            "value": value,
            "confidence": round(float(r.score), 2),
            "specificity": 1.0,
        })

    return detected


# Quick manual test — run `python detector.py` directly to try it
if __name__ == "__main__":
    sample = "My name is Ravi Kumar, phone 9876543210, Aadhaar 1234 5678 9123, I live in Village X."
    for entity in detect_entities(sample):
        print(entity)
