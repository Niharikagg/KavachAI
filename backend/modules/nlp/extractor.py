"""
extractor.py — Sensitive attribute + contextual indicator extraction.

Given a full conversation (list of messages), extracts sensitive attributes
(AGE, GENDER, LOCATION, HEALTH, FACILITY, DATE, OCCUPATION) aggregated across
the conversation, plus contextual indicators (e.g. "exact_age",
"village_level_location") tied back to the attribute that triggered them.

AGE uses regex (exact numbers are more reliable this way).
LOCATION and DATE use spaCy's built-in NER (already generalizes well).
GENDER, HEALTH, OCCUPATION, FACILITY use GLiNER — a zero-shot NER model
that generalizes to any term in these categories, not just a fixed keyword
list, since spaCy's base model was never trained on these categories.

Output matches the locked API contract with the risk module.
"""

import re
import spacy
from gliner import GLiNER

nlp = spacy.load("en_core_web_lg")

# Zero-shot NER model — labels are given at inference time, so it can
# recognize ANY health condition, occupation, or gender term, not just
# ones we hardcoded in a list.
gliner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

# GLiNER labels -> our schema's attribute types, plus a base specificity
# (specificity is still a judgment call we assign, since GLiNER doesn't
# output "how identifying is this" — just "what is this")
GLINER_LABELS = {
    "gender identity term such as man, woman, boy, girl, or transgender": ("GENDER", 0.7),
    "medical condition or disease": ("HEALTH", 0.9),
    "occupation or job title": ("OCCUPATION", 0.6),
    "facility such as hospital, school, or shelter": ("FACILITY", 0.8),
}

# Words that make an attribute more re-identifying because they imply uniqueness
UNIQUENESS_WORDS = ["only", "first", "last", "youngest", "eldest", "sole", "single"]

AGE_PATTERN = re.compile(r"\b(\d{1,3})\s*(?:years?\s*old|yrs?\s*old|-year-old|y/o)\b", re.IGNORECASE)
AGE_RANGE_PATTERN = re.compile(r"\b(teen(?:ager)?|in (?:my|her|his) (?:twenties|thirties|forties|fifties|sixties)|elderly|middle-aged|young adult)\b", re.IGNORECASE)

# Small/precise location indicators vs broad ones
GRANULAR_LOCATION_WORDS = ["village", "street", "colony", "ward", "block", "sector", "lane"]
BROAD_LOCATION_WORDS = ["state", "country", "region", "district"]


def _find_age(text: str) -> list[dict]:
    found = []
    for m in AGE_PATTERN.finditer(text):
        found.append({"value": m.group(1), "specificity": 1.0, "indicator": "exact_age"})
    for m in AGE_RANGE_PATTERN.finditer(text):
        found.append({"value": m.group(0), "specificity": 0.4, "indicator": "age_range"})
    return found


def _find_location(text: str, doc) -> list[dict]:
    found = []
    lower = text.lower()
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC", "FAC"):
            specificity = 1.0 if any(w in lower for w in GRANULAR_LOCATION_WORDS) else 0.6
            indicator = "village_level_location" if specificity == 1.0 else "broad_location"
            found.append({"value": ent.text, "specificity": specificity, "indicator": indicator})
    return found


def _find_date(doc, text: str) -> list[dict]:
    found = []
    for ent in doc.ents:
        if ent.label_ == "DATE" and not AGE_PATTERN.search(ent.text):
            found.append({"value": ent.text, "specificity": 0.7})
    return found


def _find_gliner_attributes(text: str) -> dict[str, list[dict]]:
    """
    Runs GLiNER once with all zero-shot labels, then buckets results
    by our schema's attribute types (GENDER, HEALTH, OCCUPATION, FACILITY).
    """
    results = {"GENDER": [], "HEALTH": [], "OCCUPATION": [], "FACILITY": []}

    predictions = gliner_model.predict_entities(text, list(GLINER_LABELS.keys()), threshold=0.4)

    for pred in predictions:
        attr_type, base_specificity = GLINER_LABELS[pred["label"]]
        results[attr_type].append({
            "value": pred["text"],
            "confidence": round(float(pred["score"]), 2),
            "specificity": base_specificity,
        })

    return results


def _find_uniqueness_indicators(text: str) -> list[str]:
    lower = text.lower()
    hits = []
    for w in UNIQUENESS_WORDS:
        if w == "last":
            # Only count "last" as uniqueness if NOT followed by a time word
            if re.search(r"\blast\b(?!\s+(year|month|week|night|time))", lower):
                hits.append(w)
        elif re.search(rf"\b{w}\b", lower):
            hits.append(w)
    return hits


def extract_sensitive_data(messages: list[dict]) -> dict:
    """
    Extract and aggregate sensitive attributes across an entire conversation.

    Args:
        messages: list of {"id": int, "text": str}

    Returns:
        dict matching the locked API contract:
        {
          "attributes": [...],
          "contextual_indicators": [...]
        }
    """
    attributes = []
    contextual_indicators = []

    for msg in messages:
        msg_id = msg["id"]
        text = msg["text"]
        doc = nlp(text)

        for a in _find_age(text):
            attributes.append({
                "type": "AGE", "value": a["value"], "confidence": 0.9,
                "specificity": a["specificity"], "source_message_id": msg_id,
            })
            contextual_indicators.append({"indicator": a["indicator"], "attribute_type": "AGE"})

        for loc in _find_location(text, doc):
            attributes.append({
                "type": "LOCATION", "value": loc["value"], "confidence": 0.85,
                "specificity": loc["specificity"], "source_message_id": msg_id,
            })
            contextual_indicators.append({"indicator": loc["indicator"], "attribute_type": "LOCATION"})

        for d in _find_date(doc, text):
            attributes.append({
                "type": "DATE", "value": d["value"], "confidence": 0.8,
                "specificity": d["specificity"], "source_message_id": msg_id,
            })

        # GLiNER-based: GENDER, HEALTH, OCCUPATION, FACILITY
        gliner_results = _find_gliner_attributes(text)
        for attr_type, hits in gliner_results.items():
            for h in hits:
                attributes.append({
                    "type": attr_type, "value": h["value"], "confidence": h["confidence"],
                    "specificity": h["specificity"], "source_message_id": msg_id,
                })
                if attr_type == "HEALTH":
                    contextual_indicators.append({"indicator": "sensitive_health_attribute", "attribute_type": "HEALTH"})

        # Uniqueness words apply to whatever attribute types were found in THIS message
        uniqueness_hits = _find_uniqueness_indicators(text)
        if uniqueness_hits:
            msg_attr_types = {a["type"] for a in attributes if a["source_message_id"] == msg_id}
            for attr_type in msg_attr_types:
                contextual_indicators.append({"indicator": "uniqueness_word", "attribute_type": attr_type})

    return {
        "attributes": attributes,
        "contextual_indicators": contextual_indicators,
    }


# Quick manual test
if __name__ == "__main__":
    sample_messages = [
        {"id": 1, "text": "I am the only 23-year-old woman in my family."},
        {"id": 2, "text": "I live in Village X, near the district hospital."},
        {"id": 3, "text": "I was diagnosed with lupus last year. I work as a data analyst."},
    ]
    result = extract_sensitive_data(sample_messages)
    import json
    print(json.dumps(result, indent=2))