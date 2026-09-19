from __future__ import annotations

from backend.modules.privacy.optimizer import optimize


def _input(text, attributes, indicators=None, risk=0.0):
    return {
        "conversation_id": "direct_pii_m3",
        "original_text": text,
        "attributes": attributes,
        "contextual_indicators": indicators or [],
        "initial_risk_score": risk,
        "initial_risk_level": "HIGH" if risk > 0.66 else "LOW",
    }


def test_english_person_phone_and_email_are_redacted_even_when_contextual_risk_is_low():
    text = "Aarav Sharma can be reached at 9876543210 or aarav.sharma@example.com."
    result = optimize(_input(text, [
        {"type": "PERSON", "value": "Aarav Sharma", "specificity": 1.0},
        {"type": "PHONE_NUMBER", "value": "9876543210", "specificity": 1.0},
        {"type": "EMAIL_ADDRESS", "value": "aarav.sharma@example.com", "specificity": 1.0},
    ]))

    assert result["sanitized_text"] == "[PERSON] can be reached at [PHONE] or [EMAIL]."
    assert {item["attribute"] for item in result["transformations"]} == {
        "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS",
    }


def test_hindi_person_phone_and_email_are_redacted():
    text = "आरव शर्मा का मोबाइल 9876543210 है और email aarav.sharma@example.com है।"
    result = optimize(_input(text, [
        {"type": "PERSON", "value": "आरव शर्मा", "specificity": 1.0},
        {"type": "PHONE_NUMBER", "value": "9876543210", "specificity": 1.0},
        {"type": "EMAIL_ADDRESS", "value": "aarav.sharma@example.com", "specificity": 1.0},
    ]))

    assert result["sanitized_text"] == "[PERSON] का मोबाइल [PHONE] है और email [EMAIL] है।"
    assert "आरव शर्मा" not in result["sanitized_text"]
    assert "9876543210" not in result["sanitized_text"]
    assert "aarav.sharma@example.com" not in result["sanitized_text"]


def test_kannada_person_and_phone_are_redacted():
    text = "ಆರವ್ ಶರ್ಮಾ ಅವರ ಮೊಬೈಲ್ 9876543210 ಆಗಿದೆ."
    result = optimize(_input(text, [
        {"type": "PERSON", "value": "ಆರವ್ ಶರ್ಮಾ", "specificity": 1.0},
        {"type": "PHONE_NUMBER", "value": "9876543210", "specificity": 1.0},
    ]))

    assert result["sanitized_text"] == "[PERSON] ಅವರ ಮೊಬೈಲ್ [PHONE] ಆಗಿದೆ."
    assert "ಆರವ್ ಶರ್ಮಾ" not in result["sanitized_text"]
    assert "9876543210" not in result["sanitized_text"]


def test_vehicle_registration_identifier_is_redacted():
    text = "The registered vehicle is KA-09-AB-4721."
    result = optimize(_input(text, [
        {"type": "VEHICLE_NUMBER", "value": "KA-09-AB-4721", "specificity": 1.0},
    ]))

    assert result["sanitized_text"] == "The registered vehicle is [VEHICLE_ID]."
    assert result["transformations"][0]["new_value"] == "[VEHICLE_ID]"


def test_mixed_direct_pii_and_contextual_attributes_are_both_sanitized():
    text = (
        "आरव शर्मा, उम्र 24, रामनगर के पास एक छोटे गाँव में रहता है। "
        "उसका मोबाइल 9876543210 है और email aarav.sharma@example.com है।"
    )
    result = optimize(_input(
        text,
        [
            {"type": "PERSON", "value": "आरव शर्मा", "specificity": 1.0},
            {"type": "PHONE_NUMBER", "value": "9876543210", "specificity": 1.0},
            {"type": "EMAIL_ADDRESS", "value": "aarav.sharma@example.com", "specificity": 1.0},
            {"type": "AGE", "value": "24", "specificity": 1.0},
            {"type": "LOCATION", "value": "छोटे गाँव", "specificity": 0.55},
        ],
        [{"indicator": "small_location", "attribute_type": "LOCATION"}],
        risk=0.9,
    ), threshold=0.0, scorer_fn=lambda attrs: max(
        (
            float(attribute.get("specificity", 0.0))
            for attribute in attrs
            if attribute.get("type") in {"AGE", "LOCATION", "HEALTH", "DATE", "FACILITY"}
        ),
        default=0.0,
    ))

    sanitized = result["sanitized_text"]
    assert "[PERSON]" in sanitized
    assert "[PHONE]" in sanitized
    assert "[EMAIL]" in sanitized
    assert "आरव शर्मा" not in sanitized
    assert "9876543210" not in sanitized
    assert "aarav.sharma@example.com" not in sanitized
    assert "24" not in sanitized
    assert "छोटे गाँव" not in sanitized
    assert "young adult" not in sanitized
    assert result["final_risk_score"] < result["initial_risk_score"]
