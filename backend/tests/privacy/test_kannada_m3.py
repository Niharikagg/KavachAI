from __future__ import annotations

from backend.modules.privacy.optimizer import optimize


KANNADA_TEXT = (
    "ನಾನು Ramanagaraದ ಹತ್ತಿರದ ಸಣ್ಣ ಹಳ್ಳಿಯಲ್ಲಿ ವಾಸಿಸುತ್ತೇನೆ. "
    "ಒಬ್ಬನೇ 23 ವರ್ಷದ ಮಹಿಳೆ ನಾನು ಮತ್ತು school teacher ಆಗಿ ಕೆಲಸ ಮಾಡುತ್ತೇನೆ. "
    "ನನಗೆ ಅಪರೂಪದ neurological disorder ಇದೆ ಮತ್ತು Bengaluruದ ಆಸ್ಪತ್ರೆಯಲ್ಲಿ "
    "12 ಸೆಪ್ಟೆಂಬರ್ 2026 ರಂದು ಚಿಕಿತ್ಸೆ ಪಡೆದಿದ್ದೇನೆ."
)


def _high_risk_input() -> dict:
    return {
        "conversation_id": "kannada_m3_regression",
        "original_text": KANNADA_TEXT,
        "attributes": [
            {"type": "AGE", "value": "23", "specificity": 1.0},
            {"type": "LOCATION", "value": "ಸಣ್ಣ ಹಳ್ಳಿ", "specificity": 0.85},
            {"type": "LOCATION", "value": "Ramanagara", "specificity": 0.85},
            {"type": "HEALTH", "value": "neurological disorder", "specificity": 0.92},
            {"type": "DATE", "value": "12 ಸೆಪ್ಟೆಂಬರ್ 2026", "specificity": 1.0},
        ],
        "contextual_indicators": [
            {"indicator": "exact_age", "attribute_type": "AGE"},
            {"indicator": "small_location", "attribute_type": "LOCATION"},
            {"indicator": "rare_condition", "attribute_type": "HEALTH"},
            {"indicator": "exact_date", "attribute_type": "DATE"},
        ],
        "initial_risk_score": 9.0,
        "initial_risk_level": "HIGH",
    }


def _force_all_candidates(attrs, indicators):
    del indicators
    return max((float(attribute.get("specificity", 0.0)) for attribute in attrs), default=0.0)


def test_kannada_generalization_removes_uniqueness_and_uses_grammar_safe_age():
    result = optimize(_high_risk_input(), threshold=0.0, scorer_fn=_force_all_candidates)
    text = result["sanitized_text"]

    assert "ಒಬ್ಬನೇ" not in text
    assert "23" not in text
    assert "a young adult" not in text
    assert "ವರ್ಷ" not in text
    assert "ಯುವ ವಯಸ್ಕ" in text


def test_kannada_health_generalization_removes_rare_signal():
    result = optimize(_high_risk_input(), threshold=0.0, scorer_fn=_force_all_candidates)
    text = result["sanitized_text"]

    assert "ಅಪರೂಪದ" not in text
    assert "rare" not in text.casefold()
    assert "neurological condition" not in text
    assert "ನರವೈಜ್ಞಾನಿಕ ಸ್ಥಿತಿ" in text


def test_kannada_date_and_location_generalization_preserve_grammar_and_city():
    result = optimize(_high_risk_input(), threshold=0.0, scorer_fn=_force_all_candidates)
    text = result["sanitized_text"]

    assert "12 ಸೆಪ್ಟೆಂಬರ್ 2026" not in text
    assert "ಇತ್ತೀಚೆಗೆ" in text
    assert "ಇತ್ತೀಚೆಗೆ ರಂದು" not in text
    assert "ಸಣ್ಣ ಹಳ್ಳಿ" not in text
    assert "ಗ್ರಾಮೀಣ ಪ್ರದೇಶ" in text
    assert "Ramanagara" in text


def test_mixed_kannada_english_does_not_insert_english_age_or_location_phrases():
    result = optimize(_high_risk_input(), threshold=0.0, scorer_fn=_force_all_candidates)
    text = result["sanitized_text"]

    assert "a rural area" not in text
    assert "a young adult" not in text
    assert "recently" not in text
    assert "school teacher" in text


def _language_case(text: str, age: str, village: str, health: str, date: str, facility: str) -> dict:
    return {
        "conversation_id": "m3_language_regression",
        "original_text": text,
        "attributes": [
            {"type": "AGE", "value": age, "specificity": 1.0},
            {"type": "LOCATION", "value": village, "specificity": 0.85},
            {"type": "LOCATION", "value": "Ramanagara", "specificity": 0.85},
            {"type": "HEALTH", "value": health, "specificity": 0.92},
            {"type": "DATE", "value": date, "specificity": 1.0},
            {"type": "FACILITY", "value": facility, "specificity": 0.98},
        ],
        "initial_risk_score": 9.0,
        "initial_risk_level": "HIGH",
    }


def test_hindi_final_sanitized_sentence_is_grammatical():
    data = _language_case(
        "मैं 23 साल की हूँ और रामनगर के पास एक छोटे से गाँव में रहती हूँ। "
        "मैं उस गाँव की अकेली 23 वर्षीय महिला हूँ और स्कूल टीचर के रूप में काम करती हूँ। "
        "मुझे एक दुर्लभ न्यूरोलॉजिकल बीमारी है और मैंने 12 सितंबर 2026 को "
        "बेंगलुरु के St. John's Medical College Hospital में इलाज कराया था।",
        "23", "छोटे से गाँव", "दुर्लभ न्यूरोलॉजिकल बीमारी", "12 सितंबर 2026",
        "St. John's Medical College Hospital",
    )
    result = optimize(data, threshold=0.0, scorer_fn=_force_all_candidates)
    text = result["sanitized_text"]

    assert text == (
        "मैं एक युवा वयस्क हूँ और रामनगर के पास एक ग्रामीण क्षेत्र में रहती हूँ। "
        "मैं उस गाँव की एक युवा वयस्क महिला हूँ और स्कूल टीचर के रूप में काम करती हूँ। "
        "मुझे एक न्यूरोलॉजिकल बीमारी है और मैंने हाल ही में बेंगलुरु के एक स्वास्थ्य केंद्र में इलाज कराया था।"
    )
    assert result["final_risk_score"] < result["initial_risk_score"]


def test_kanglish_final_sanitized_sentence_preserves_roman_grammar():
    data = _language_case(
        "Nanage 23 varsha, nanu Ramanagara hatra iro ondu sanna halliyalli vasistiddene. "
        "Nanu aa halliyalliro obbane 23 varshada mahile mattu school teacher aagi kelasa madtini. "
        "Nanage aparoopada neurological disorder ide mattu 12 September 2026 randu "
        "Bengaluru St. John's Medical College Hospital nalli treatment padediddene.",
        "23", "sanna halli", "aparoopada neurological disorder", "12 September 2026",
        "St. John's Medical College Hospital",
    )
    result = optimize(data, threshold=0.0, scorer_fn=_force_all_candidates)
    text = result["sanitized_text"]

    assert text == (
        "Naanu young adult, nanu Ramanagara hatra iro ondu grameena pradeshadalli vasistiddene. "
        "Nanu aa halliyalliro young adult mahile mattu school teacher aagi kelasa madtini. "
        "Nanage neurological condition ide mattu recently Bengaluru nalli ondu healthcare facility nalli treatment padediddene."
    )
    assert "a young adult" not in text
    assert "recently randu" not in text
    assert result["final_risk_score"] < result["initial_risk_score"]


def test_hinglish_final_sanitized_sentence_preserves_hindi_grammar():
    data = _language_case(
        "Main 23 saal ki hoon aur Ramanagara ke paas ek chhote gaon mein rehti hoon. "
        "Main us gaon ki eklauti 23 saal ki mahila hoon aur school teacher ke roop mein kaam karti hoon. "
        "Mujhe ek rare neurological disorder hai aur maine 12 September 2026 ko Bengaluru ke "
        "St. John's Medical College Hospital mein treatment liya tha.",
        "23", "chhote gaon", "rare neurological disorder", "12 September 2026",
        "St. John's Medical College Hospital",
    )
    result = optimize(data, threshold=0.0, scorer_fn=_force_all_candidates)
    text = result["sanitized_text"]

    assert text == (
        "Main young adult hoon aur Ramanagara ke paas ek rural area mein rehti hoon. "
        "Main us gaon ki young adult mahila hoon aur school teacher ke roop mein kaam karti hoon. "
        "Mujhe ek neurological condition hai aur maine recently Bengaluru ke ek healthcare facility mein treatment liya tha."
    )
    assert "eklauti" not in text.casefold()
    assert "rare" not in text.casefold()
    assert "recently ko" not in text.casefold()
    assert result["final_risk_score"] < result["initial_risk_score"]


def test_english_age_uniqueness_and_village_are_realized_as_complete_phrases():
    data = {
        "conversation_id": "english_m3_realization",
        "original_text": (
            "I am 23 years old and live in a small village near Ramanagara. "
            "I am the only 23-year-old woman in my village and work as a school teacher."
        ),
        "attributes": [
            {"type": "AGE", "value": "23", "specificity": 1.0},
            {"type": "LOCATION", "value": "Ramanagara", "specificity": 0.85},
        ],
        "contextual_indicators": [
            {"indicator": "exact_age", "attribute_type": "AGE"},
            {"indicator": "uniqueness_word", "attribute_type": "AGE"},
            {"indicator": "uniqueness_word", "attribute_type": "GENDER"},
            {"indicator": "small_location", "attribute_type": "LOCATION"},
        ],
        "initial_risk_score": 9.0,
        "initial_risk_level": "HIGH",
    }
    result = optimize(data, threshold=0.0, scorer_fn=_force_all_candidates)
    text = result["sanitized_text"]

    assert text == (
        "I am a young adult and live in a rural area near Ramanagara. "
        "I am a young adult woman in my village and work as a school teacher."
    )
    assert "only" not in text.casefold()
    assert "young adult-year-old" not in text.casefold()
    assert result["final_risk_score"] < result["initial_risk_score"]