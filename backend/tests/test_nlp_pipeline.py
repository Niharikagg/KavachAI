"""Focused tests for the Module 1 extraction contract."""

import pytest

from backend.modules.nlp import extractor, pipeline
from backend.modules.nlp.detector import detect_entities
from backend.modules.nlp.muril import _script_context


class FakeGLiNER:
    def predict_entities(self, text, labels, threshold):
        del labels, threshold
        lower = text.lower()
        candidates = [
            ("gender", ("woman", "man", "transgender")),
            ("health", ("diabetes", "rare autoimmune condition", "rare neurological disorder", "asthma")),
            ("occupation", ("school teacher", "teacher", "data analyst", "engineer")),
            ("facility", ("st. john's medical college hospital", "apollo hospitals", "apollo hospital", "hospital", "school")),
            ("religion", ("hindu", "muslim", "sikh")),
            ("caste", ("obc", "scheduled caste", "dalit")),
            ("education", ("graduate", "bachelor", "phd")),
        ]
        predictions = []
        for label, terms in candidates:
            for term in terms:
                if term in lower:
                    start = lower.index(term)
                    predictions.append({"label": label.upper(), "text": text[start:start + len(term)], "score": 0.94})
                    break
        return predictions


@pytest.fixture(autouse=True)
def stub_large_models(monkeypatch):
    monkeypatch.setattr(extractor, "_get_gliner_model", lambda: FakeGLiNER())
    monkeypatch.setattr(
        pipeline,
        "get_processing_contexts",
        lambda texts: [{"observed_scripts": ["en"]} for _ in texts],
    )


def test_detector_handles_indian_pii():
    result = detect_entities(
        "Aadhaar 1234 5678 9123, PAN ABCDE1234F, +91 9876543210, "
        "mail citizen@example.com, PIN code 560001, vehicle KA 01 AB 1234."
    )
    types = {entity["type"] for entity in result}
    assert {"AADHAAR", "PAN", "PHONE_NUMBER", "EMAIL_ADDRESS", "PINCODE", "VEHICLE_NUMBER"}.issubset(types)


def test_detector_handles_additional_indian_identifiers():
    result = detect_entities("IFSC SBIN0001234, account 123456789012, voter ABC1234567")
    types = {entity["type"] for entity in result}
    assert {"IFSC", "BANK_ACCOUNT", "VOTER_ID"}.issubset(types)


def test_english_contextual_attributes():
    result = pipeline.analyze_conversation(
        "english",
        [{"id": 1, "text": "I am 31 years old and visited Apollo Hospitals in Bengaluru."}],
    )
    types = {attribute["type"] for attribute in result["attributes"]}
    assert {"AGE", "FACILITY", "LOCATION"}.issubset(types)


def test_pipeline_002_contract_and_target_detections():
    text = (
        "I am 24 years old and visited St. John's Medical College Hospital in Bengaluru "
        "on September 12, 2026 for treatment of a rare neurological disorder. I live "
        "in a small village near Ramanagara and work as a school teacher."
    )
    result = pipeline.analyze_conversation("test_pipeline_002", [{"id": 1, "text": text}])

    assert set(result) == {"conversation_id", "original_text", "attributes", "contextual_indicators"}
    assert result["conversation_id"] == "test_pipeline_002"
    assert result["original_text"] == text
    detected = {(item["type"], item["value"]) for item in result["attributes"]}
    assert {
        ("AGE", "24"),
        ("LOCATION", "Ramanagara"),
        ("FACILITY", "St. John's Medical College Hospital"),
        ("DATE", "September 12, 2026"),
        ("HEALTH", "rare neurological disorder"),
        ("OCCUPATION", "school teacher"),
    }.issubset(detected)
    assert all(item["source_message_id"] == 1 for item in result["attributes"])
    assert {
        ("exact_age", "AGE"),
        ("specific_facility", "FACILITY"),
        ("exact_date", "DATE"),
        ("rare_condition", "HEALTH"),
        ("small_location", "LOCATION"),
    }.issubset({(item["indicator"], item["attribute_type"]) for item in result["contextual_indicators"]})


def test_all_configured_contextual_attributes():
    result = pipeline.analyze_conversation(
        "contextual",
        [{
            "id": 1,
            "text": "I am a woman with a rare autoimmune condition. I work as a data analyst. "
            "I am Hindu, OBC, and a graduate.",
        }],
    )
    types = {attribute["type"] for attribute in result["attributes"]}
    assert {"GENDER", "HEALTH", "OCCUPATION", "RELIGION", "CASTE", "EDUCATION"}.issubset(types)


def test_code_mixed_messages_are_processed():
    result = pipeline.analyze_conversation(
        "mixed",
        [
            {"id": 1, "text": "Mujhe Apollo hospital mein treatment ke liye jaana hai."},
            {"id": 2, "text": "Meri age 31 years hai aur main Bengaluru mein rehta hoon."},
        ],
    )
    values = {attribute["value"].lower() for attribute in result["attributes"]}
    assert "apollo hospital" in values or "hospital" in values
    assert "31" in values


def test_muril_processing_context_records_code_mixed_scripts_without_language_claim():
    context = _script_context("Meri age 31 years hai")
    assert context["is_code_mixed"] is True
    assert set(context["observed_scripts"]) == {"en", "hi"}
    assert "language" not in context


def test_duplicate_detections_are_normalized_once():
    result = pipeline.analyze_conversation(
        "dedup",
        [{"id": 1, "text": "My Aadhaar is 1234 5678 9123."}],
    )
    aadhaar = [attribute for attribute in result["attributes"] if attribute["type"] == "AADHAAR"]
    assert len(aadhaar) == 1


def test_contextual_indicators_only_reflect_supported_input():
    result = pipeline.analyze_conversation(
        "indicators",
        [{"id": 1, "text": "I am the only 31-year-old woman."}],
    )
    assert {"indicator": "exact_age", "attribute_type": "AGE"} in result["contextual_indicators"]
    assert {"indicator": "uniqueness_word", "attribute_type": "AGE"} in result["contextual_indicators"]
    assert {"indicator": "uniqueness_word", "attribute_type": "GENDER"} in result["contextual_indicators"]


def test_final_m1_schema_has_no_downstream_fields():
    result = pipeline.analyze_conversation("schema", [{"id": 1, "text": "I am 31 years old."}])
    assert set(result) == {"conversation_id", "original_text", "attributes", "contextual_indicators"}
    for attribute in result["attributes"]:
        assert set(attribute) == {
            "type",
            "value",
            "confidence",
            "specificity",
            "source_message_id",
        }
