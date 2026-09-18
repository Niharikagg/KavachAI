"""
test_m1_m2_integration.py — Integration tests connecting M1 NLP to M2 Risk.

Validates:
  1. Full end-to-end flow: Citizen conversation -> M1 -> Adapter -> M2.
  2. Test 1 (test_pipeline_002): exact input from specifications.
  3. Architectural separation: M1 has no risk fields; M2 produces risk fields.
  4. M2 is fed directly from M1 output (no duplicate extraction).
  5. Decoupling: M1 confidence changes do not affect M2 risk calculation.
  6. Direct authoritative derivation: features built by M2 build_features(),
     scored by M2 score_risk().
"""

import pytest

from backend.modules.nlp import extractor, pipeline
from backend.modules.risk.features import build_features
from backend.modules.risk.m1_m2_adapter import adapt_m1_to_m2, analyze_with_m2
from backend.modules.risk.scorer import score_risk


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
    """Stub heavy ML models to keep tests fast, deterministic, and offline-capable."""
    monkeypatch.setattr(extractor, "_get_gliner_model", lambda: FakeGLiNER())
    monkeypatch.setattr(
        pipeline,
        "get_processing_contexts",
        lambda texts: [{"observed_scripts": ["en"]} for _ in texts],
    )


def test_pipeline_002_end_to_end():
    """
    Test 1: test_pipeline_002
    Input:
      'I am 24 years old and visited St. John's Medical College Hospital in Bengaluru
       on September 12, 2026 for treatment of a rare neurological disorder. I live
       in a small village near Ramanagara and work as a school teacher.'
    """
    raw_text = (
        "I am 24 years old and visited St. John's Medical College Hospital in Bengaluru "
        "on September 12, 2026 for treatment of a rare neurological disorder. I live "
        "in a small village near Ramanagara and work as a school teacher."
    )
    messages = [{"id": 1, "text": raw_text}]
    conversation_id = "test_pipeline_002"

    # 1. Step 1: Run M1 extraction
    m1_result = pipeline.analyze_conversation(conversation_id, messages)

    # Verify M1 output schema and attributes
    assert set(m1_result) == {"conversation_id", "original_text", "attributes", "contextual_indicators"}
    assert m1_result["conversation_id"] == conversation_id
    assert m1_result["original_text"] == raw_text

    detected_pairs = {(attr["type"], attr["value"]) for attr in m1_result["attributes"]}
    expected_pairs = {
        ("AGE", "24"),
        ("FACILITY", "St. John's Medical College Hospital"),
        ("LOCATION", "Bengaluru"),
        ("LOCATION", "Ramanagara"),
        ("DATE", "September 12, 2026"),
        ("HEALTH", "rare neurological disorder"),
        ("OCCUPATION", "school teacher"),
    }
    assert expected_pairs.issubset(detected_pairs)

    # Verify confidence and specificity for every attribute
    for attr in m1_result["attributes"]:
        assert "confidence" in attr
        assert isinstance(attr["confidence"], float)
        assert 0.0 <= attr["confidence"] <= 1.0
        assert "specificity" in attr
        assert isinstance(attr["specificity"], float)
        assert 0.0 <= attr["specificity"] <= 1.0
        assert "source_message_id" in attr

    # Verify M1 has NO downstream risk fields
    assert "risk_score" not in m1_result
    assert "risk_level" not in m1_result

    # 2. Step 2: Pass M1 output to M2 via adapter
    m2_result = analyze_with_m2(m1_result)

    # 3. Step 3: Verify M2 receives attributes, builds features, and produces risk
    assert "features" in m2_result
    assert "risk_score" in m2_result
    assert "risk_level" in m2_result

    features = m2_result["features"]
    assert "attribute_count_normalized" in features
    assert "average_specificity" in features
    assert "maximum_specificity" in features
    assert "high_specificity_ratio" in features
    assert "sensitive_category_count_normalized" in features
    assert "uniqueness_risk" in features

    # Verify authoritative calculation matching M2 build_features + score_risk
    adapted_conv = adapt_m1_to_m2(m1_result)
    expected_features = build_features(adapted_conv)
    expected_risk = score_risk(expected_features)

    assert features == expected_features
    assert m2_result["risk_score"] == expected_risk["risk_score"]
    assert m2_result["risk_level"] == expected_risk["risk_level"]

    # Verify risk_score is in valid range [0, 10] and risk_level is one of LOW, MEDIUM, HIGH
    assert 0.0 <= m2_result["risk_score"] <= 10.0
    assert m2_result["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert m2_result["risk_level"] == "HIGH"


def test_architectural_separation_m1_vs_m2():
    """Prove strict separation of responsibilities between M1 and M2."""
    messages = [{"id": 1, "text": "My phone is 9876543210."}]
    m1_result = pipeline.analyze_conversation("sep_001", messages)

    # M1 only owns extraction
    assert "conversation_id" in m1_result
    assert "original_text" in m1_result
    assert "attributes" in m1_result
    assert "contextual_indicators" in m1_result
    assert "risk_score" not in m1_result
    assert "risk_level" not in m1_result
    assert "features" not in m1_result

    # M2 only owns risk
    m2_result = analyze_with_m2(m1_result)
    assert "risk_score" in m2_result
    assert "risk_level" in m2_result
    assert "features" in m2_result


def test_m2_uses_m1_output_without_re_extracting():
    """Verify M2 does not independently re-parse or re-extract attributes from text."""
    # Synthetic M1 result passed directly
    m1_synthetic = {
        "conversation_id": "mock_001",
        "original_text": "I am an anonymous citizen.",
        "attributes": [
            {"type": "AGE", "value": "24", "confidence": 0.98, "specificity": 1.0, "source_message_id": 1},
            {"type": "HEALTH", "value": "rare condition", "confidence": 0.95, "specificity": 0.9, "source_message_id": 1},
        ],
        "contextual_indicators": [
            {"indicator": "exact_age", "attribute_type": "AGE"},
            {"indicator": "rare_condition", "attribute_type": "HEALTH"},
        ],
    }

    # Pass to adapter
    adapted = adapt_m1_to_m2(m1_synthetic)
    # The attributes must originate directly from m1_synthetic
    assert len(adapted["attributes"]) == 2
    assert adapted["attributes"][0]["type"] == "AGE"
    assert adapted["attributes"][0]["value"] == "24"
    assert adapted["attributes"][1]["type"] == "HEALTH"

    # Indicator 'rare_condition' should be mapped to 'sensitive_health_attribute'
    ind_names = [ind["indicator"] for ind in adapted["contextual_indicators"]]
    assert "exact_age" in ind_names
    assert "sensitive_health_attribute" in ind_names

    # Compute risk using authoritative functions
    features = build_features(adapted)
    risk = score_risk(features)
    assert risk["risk_score"] > 0.0
    assert risk["risk_level"] in {"LOW", "MEDIUM", "HIGH"}


def test_confidence_decoupled_from_m2_risk():
    """Verify changing M1 attribute confidence has zero impact on M2 features and risk score."""
    base_m1 = {
        "conversation_id": "test_conf",
        "original_text": "I am 30 years old.",
        "attributes": [
            {"type": "AGE", "value": "30", "confidence": 0.99, "specificity": 1.0, "source_message_id": 1},
            {"type": "HEALTH", "value": "diabetes", "confidence": 0.95, "specificity": 0.9, "source_message_id": 1},
        ],
        "contextual_indicators": [{"indicator": "exact_age", "attribute_type": "AGE"}],
    }
    result_high_conf = analyze_with_m2(base_m1)

    low_conf_m1 = {
        "conversation_id": "test_conf",
        "original_text": "I am 30 years old.",
        "attributes": [
            {"type": "AGE", "value": "30", "confidence": 0.10, "specificity": 1.0, "source_message_id": 1},
            {"type": "HEALTH", "value": "diabetes", "confidence": 0.05, "specificity": 0.9, "source_message_id": 1},
        ],
        "contextual_indicators": [{"indicator": "exact_age", "attribute_type": "AGE"}],
    }
    result_low_conf = analyze_with_m2(low_conf_m1)

    # Features and risk must be identical regardless of confidence
    assert result_high_conf["features"] == result_low_conf["features"]
    assert result_high_conf["risk_score"] == result_low_conf["risk_score"]
    assert result_high_conf["risk_level"] == result_low_conf["risk_level"]


def test_low_risk_conversation():
    """Verify a conversation with low identifying information scores LOW risk."""
    m1_low = {
        "conversation_id": "low_001",
        "original_text": "I live in India.",
        "attributes": [
            {"type": "LOCATION", "value": "India", "confidence": 0.95, "specificity": 0.2, "source_message_id": 1}
        ],
        "contextual_indicators": [],
    }
    result = analyze_with_m2(m1_low)
    assert result["risk_level"] == "LOW"
    assert result["risk_score"] <= 3.33
