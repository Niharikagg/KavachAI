"""
test_nlp_pipeline.py — Unit tests for the NLP module (detector, extractor,
embeddings, pipeline). Covers low/medium/high risk sample conversations.
"""

import pytest
from backend.modules.nlp.detector import detect_entities
from backend.modules.nlp.extractor import extract_sensitive_data
from backend.modules.nlp.embeddings import create_embedding, cosine_similarity
from backend.modules.nlp.pipeline import analyze_conversation


# --- Sample conversations (low / medium / high risk) ---

LOW_RISK_MESSAGES = [
    {"id": 1, "text": "The weather has been nice lately."},
    {"id": 2, "text": "I enjoy reading books in my free time."},
]

MEDIUM_RISK_MESSAGES = [
    {"id": 1, "text": "I am a teacher in my village."},
    {"id": 2, "text": "I have been feeling unwell recently."},
]

HIGH_RISK_MESSAGES = [
    {"id": 1, "text": "I am the only 23-year-old woman in my family, phone 9876543210."},
    {"id": 2, "text": "I live in Village X, near the district hospital."},
    {"id": 3, "text": "I was diagnosed with diabetes last year. My Aadhaar is 1234 5678 9123."},
]


# --- detector.py tests ---

class TestDetector:
    def test_empty_text_returns_empty_list(self):
        assert detect_entities("") == []
        assert detect_entities("   ") == []

    def test_detects_aadhaar(self):
        result = detect_entities("My Aadhaar is 1234 5678 9123.")
        types = [r["type"] for r in result]
        assert "AADHAAR" in types

    def test_detects_phone_number(self):
        result = detect_entities("Call me at 9876543210.")
        types = [r["type"] for r in result]
        assert "PHONE_NUMBER" in types

    def test_no_false_positive_person_for_aadhaar_word(self):
        result = detect_entities("My Aadhaar is 1234 5678 9123.")
        for r in result:
            assert not (r["type"] == "PERSON" and r["value"].lower() == "aadhaar")

    def test_low_risk_text_has_minimal_detections(self):
        result = detect_entities(LOW_RISK_MESSAGES[0]["text"])
        assert len(result) == 0


# --- extractor.py tests ---

class TestExtractor:
    def test_empty_messages_returns_empty_attributes(self):
        result = extract_sensitive_data([])
        assert result["attributes"] == []
        assert result["contextual_indicators"] == []

    def test_detects_age(self):
        result = extract_sensitive_data([{"id": 1, "text": "I am 23 years old."}])
        types = [a["type"] for a in result["attributes"]]
        assert "AGE" in types

    def test_detects_health_attribute(self):
        result = extract_sensitive_data([{"id": 1, "text": "I have diabetes."}])
        types = [a["type"] for a in result["attributes"]]
        assert "HEALTH" in types

    def test_uniqueness_word_triggers_contextual_indicator(self):
        result = extract_sensitive_data([{"id": 1, "text": "I am the only 23-year-old in my village."}])
        indicators = [c["indicator"] for c in result["contextual_indicators"]]
        assert "uniqueness_word" in indicators

    def test_last_year_does_not_trigger_uniqueness(self):
        result = extract_sensitive_data([{"id": 1, "text": "I was diagnosed last year."}])
        indicators = [c["indicator"] for c in result["contextual_indicators"]]
        assert "uniqueness_word" not in indicators

    def test_source_message_id_is_tracked(self):
        result = extract_sensitive_data(MEDIUM_RISK_MESSAGES)
        for attr in result["attributes"]:
            assert attr["source_message_id"] in [1, 2]


# --- embeddings.py tests ---

class TestEmbeddings:
    def test_empty_text_returns_empty_list(self):
        assert create_embedding("") == []

    def test_embedding_has_correct_dimension(self):
        vec = create_embedding("Sample text.")
        assert len(vec) == 384

    def test_similar_sentences_score_higher_than_unrelated(self):
        vec1 = create_embedding("I have diabetes and live in a village.")
        vec2 = create_embedding("A person has a health condition and lives rurally.")
        vec3 = create_embedding("The stock market rose today.")
        sim_related = cosine_similarity(vec1, vec2)
        sim_unrelated = cosine_similarity(vec1, vec3)
        assert sim_related > sim_unrelated


# --- pipeline.py integration tests ---

class TestPipeline:
    def test_output_matches_locked_schema_keys(self):
        result = analyze_conversation("test_conv", HIGH_RISK_MESSAGES)
        assert set(result.keys()) == {"conversation_id", "attributes", "contextual_indicators", "embeddings"}

    def test_no_duplicate_attributes(self):
        result = analyze_conversation("test_conv", HIGH_RISK_MESSAGES)
        seen = set()
        for attr in result["attributes"]:
            key = (attr["type"], attr["value"], attr["source_message_id"])
            assert key not in seen, f"Duplicate attribute found: {key}"
            seen.add(key)

    def test_high_risk_conversation_detects_multiple_attribute_types(self):
        result = analyze_conversation("test_conv", HIGH_RISK_MESSAGES)
        types_found = {a["type"] for a in result["attributes"]}
        # Should detect at least these across the 3 high-risk messages
        assert {"AGE", "LOCATION", "HEALTH"}.issubset(types_found)

    def test_low_risk_conversation_detects_few_or_no_attributes(self):
        result = analyze_conversation("test_conv", LOW_RISK_MESSAGES)
        assert len(result["attributes"]) == 0

    def test_every_message_has_an_embedding(self):
        result = analyze_conversation("test_conv", MEDIUM_RISK_MESSAGES)
        assert set(result["embeddings"].keys()) == {1, 2}
        for emb in result["embeddings"].values():
            assert len(emb) == 384