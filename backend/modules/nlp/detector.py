"""
detector.py — Explicit PII detection module.

Detects standard PII entities (name, phone, email, location) using
Presidio's built-in analyzer, plus custom regex recognizers for
Indian identifiers (Aadhaar, PAN) that Presidio doesn't cover by default.
"""

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern

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

# --- Build the analyzer once at module load ---

analyzer = AnalyzerEngine()
analyzer.registry.add_recognizer(aadhaar_recognizer)
analyzer.registry.add_recognizer(pan_recognizer)

# Entities we care about for this project (explicit PII layer only —
# sensitive/contextual attributes like health/age are handled in extractor.py)
TARGET_ENTITIES = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "LOCATION",
    "AADHAAR",
    "PAN",
]

# Words that Presidio's PERSON model sometimes misfires on
PERSON_FALSE_POSITIVES = {"aadhaar", "aadhar", "pan", "uidai"}


def detect_entities(text: str) -> list[dict[str, str]]:
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

    detected = []
    for r in results:
        value = text[r.start:r.end]
        if r.entity_type == "PERSON" and value.lower() in PERSON_FALSE_POSITIVES:
            continue  # skip known false positives
        detected.append({
            "type": r.entity_type,
            "value": value,
            "confidence": f"{r.score:.2f}",
        })

    return detected


# Quick manual test — run `python detector.py` directly to try it
if __name__ == "__main__":
    sample = "My name is Ravi Kumar, phone 9876543210, Aadhaar 1234 5678 9123, I live in Village X."
    for entity in detect_entities(sample):
        print(entity)