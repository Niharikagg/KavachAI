"""
pipeline.py — NLP module orchestration.

Combines MuRIL processing context, direct PII detection, contextual
extraction, and normalization into the Module 1 contract.
"""

import re

from backend.modules.nlp.detector import detect_entities
from backend.modules.nlp.extractor import extract_sensitive_data
from backend.modules.nlp.muril import get_processing_contexts
from backend.modules.nlp.normalizer import normalize_attributes, normalize_indicators


def _original_text(messages: list[dict]) -> str:
    """Preserve submitted text without exposing processing metadata."""
    return "\n".join(str(message["text"]) for message in messages)


def _attribute_indicators(attributes: list[dict], messages: list[dict]) -> list[dict[str, str]]:
    """Derive explainable indicators from normalized M1 detections."""
    indicators = []
    for attribute in attributes:
        attribute_type = attribute["type"]
        value = attribute["value"]
        if attribute_type == "AGE" and attribute["specificity"] == 1.0:
            indicators.append({"indicator": "exact_age", "attribute_type": "AGE"})
        elif attribute_type == "FACILITY":
            indicators.append({"indicator": "specific_facility", "attribute_type": "FACILITY"})
        elif attribute_type == "DATE" and attribute["specificity"] == 1.0:
            indicators.append({"indicator": "exact_date", "attribute_type": "DATE"})
        elif attribute_type == "HEALTH" and "rare" in value.casefold():
            indicators.append({"indicator": "rare_condition", "attribute_type": "HEALTH"})
        elif attribute_type == "LOCATION":
            source_text = next(
                (str(message["text"]) for message in messages if message["id"] == attribute["source_message_id"]),
                "",
            )
            if re.search(
                rf"\bsmall\s+(?:village|town|city)\b.*?\bnear\s+{re.escape(value)}\b",
                source_text,
                re.IGNORECASE,
            ):
                indicators.append({"indicator": "small_location", "attribute_type": "LOCATION"})
    return indicators


def analyze_conversation(conversation_id: str, messages: list[dict]) -> dict:
    """
    Run the full NLP pipeline on a conversation.

    Args:
        conversation_id: Unique ID for this conversation.
        messages: list of {"id": int, "text": str}

    Returns:
        dict matching the locked API contract:
        {
          "conversation_id": str,
          "original_text": str,
          "attributes": [...],       # explicit PII + sensitive attributes, merged
          "contextual_indicators": [...]
        }
    """
    all_attributes = []
    language_contexts = get_processing_contexts([message["text"] for message in messages])

    # 1. Explicit PII detection (Presidio) — per message
    for msg in messages:
        msg_id = msg["id"]
        text = msg["text"]

        pii_entities = detect_entities(text)
        for entity in pii_entities:
            all_attributes.append({
                "type": entity["type"],
                "value": entity["value"],
                "confidence": float(entity["confidence"]),
                "specificity": float(entity.get("specificity", 1.0)),
                "source_message_id": msg_id,
            })

    extraction_result = extract_sensitive_data(messages, language_contexts)
    all_attributes.extend(extraction_result["attributes"])

    normalized_attributes = normalize_attributes(all_attributes)
    indicators = extraction_result["contextual_indicators"] + _attribute_indicators(
        normalized_attributes, messages
    )
    return {
        "conversation_id": conversation_id,
        "original_text": _original_text(messages),
        "attributes": normalized_attributes,
        "contextual_indicators": normalize_indicators(indicators),
    }


# Quick manual test
if __name__ == "__main__":
    import json

    sample_messages = [
        {"id": 1, "text": "I am the only 23-year-old woman in my family, phone 9876543210."},
        {"id": 2, "text": "I live in Village X, near the district hospital."},
        {"id": 3, "text": "I was diagnosed with diabetes last year. My Aadhaar is 1234 5678 9123."},
    ]
    result = analyze_conversation("conv_001", sample_messages)

    print(json.dumps(result, indent=2))
