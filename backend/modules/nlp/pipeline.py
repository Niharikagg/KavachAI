"""
pipeline.py — NLP module orchestration.

Combines detector.py (explicit PII), extractor.py (sensitive attributes +
contextual indicators), and embeddings.py (semantic similarity) into a
single function matching the locked API contract with the risk module.
"""

from backend.modules.nlp.detector import detect_entities
from backend.modules.nlp.extractor import extract_sensitive_data
from backend.modules.nlp.embeddings import create_embedding


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
          "attributes": [...],       # explicit PII + sensitive attributes, merged
          "contextual_indicators": [...],
          "embeddings": {msg_id: [floats], ...}   # optional, for downstream similarity checks
        }
    """
    all_attributes = []
    embeddings_by_message = {}

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
                "specificity": 1.0,  # explicit PII is always max specificity
                "source_message_id": msg_id,
            })

        # 3. Embeddings — per message, for downstream meaning-preservation checks
        embeddings_by_message[msg_id] = create_embedding(text)

    # 2. Sensitive attribute + contextual indicator extraction — across all messages
    extraction_result = extract_sensitive_data(messages)
    all_attributes.extend(extraction_result["attributes"])
    contextual_indicators = extraction_result["contextual_indicators"]

        # Deduplicate attributes that both detector.py and extractor.py may have found
    seen = set()
    deduped_attributes = []
    for attr in all_attributes:
        key = (attr["type"], attr["value"], attr["source_message_id"])
        if key not in seen:
            seen.add(key)
            deduped_attributes.append(attr)

    return {
        "conversation_id": conversation_id,
        "attributes": deduped_attributes,
        "contextual_indicators": contextual_indicators,
        "embeddings": embeddings_by_message,
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

    # Don't print full embeddings (too long) — just show their lengths
    printable = {**result, "embeddings": {k: f"[{len(v)} floats]" for k, v in result["embeddings"].items()}}
    print(json.dumps(printable, indent=2))