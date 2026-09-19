from __future__ import annotations

import pytest

from backend.modules.nlp import extractor, pipeline
from backend.modules.nlp.muril import _script_context

EXPECTED_TYPES = {
    "AGE",
    "LOCATION",
    "GENDER",
    "HEALTH",
    "OCCUPATION",
    "DATE",
    "FACILITY",
}
EXPECTED_INDICATORS = {
    ("exact_age", "AGE"),
    ("exact_date", "DATE"),
    ("small_location", "LOCATION"),
    ("rare_condition", "HEALTH"),
    ("specific_facility", "FACILITY"),
    ("uniqueness_word", "AGE"),
    ("uniqueness_word", "GENDER"),
}

CASES = {
    "english": "I am 23 years old and live in a small village near Ramanagara. I am the only 23-year-old woman in my village and I work as a school teacher. I have a rare neurological disorder and received treatment at St. John's Medical College Hospital in Bengaluru on September 12, 2026.",
    "hindi": "मेरी उम्र 23 साल है और मैं रामनगर के पास एक छोटे से गांव में रहती हूं। मैं अपने गांव की इकलौती 23 वर्षीय महिला हूं और स्कूल शिक्षिका के रूप में काम करती हूं। मुझे एक दुर्लभ न्यूरोलॉजिकल विकार है और 12 सितंबर, 2026 को बेंगलुरु के सेंट जॉन्स मेडिकल कॉलेज अस्पताल में इलाज कराया था।",
    "kannada": "ನನಗೆ 23 ವರ್ಷ ವಯಸ್ಸಾಗಿದೆ ಮತ್ತು ನಾನು ರಾಮನಗರದ ಬಳಿಯಿರುವ ಒಂದು ಸಣ್ಣ ಹಳ್ಳಿಯಲ್ಲಿ ವಾಸಿಸುತ್ತಿದ್ದೇನೆ. ನಾನು ನಮ್ಮ ಹಳ್ಳಿಯ ಏಕೈಕ 23 ವರ್ಷದ ಮಹಿಳೆ ಮತ್ತು ಶಾಲಾ ಶಿಕ್ಷಕಿಯಾಗಿ ಕೆಲಸ ಮಾಡುತ್ತೇನೆ. ನನಗೆ ಅಪರೂಪದ ನರವೈಜ್ಞಾನಿಕ ಕಾಯಿಲೆ ಇದೆ ಮತ್ತು 12 ಸೆಪ್ಟೆಂಬರ್ 2026 ರಂದು ಬೆಂಗಳೂರಿನ ಸೇಂಟ್ ಜಾನ್ಸ್ ಮೆಡಿಕಲ್ ಕಾಲೇಜು ಆಸ್ಪತ್ರೆಯಲ್ಲಿ ಚಿಕಿತ್ಸೆ ಪಡೆದಿದ್ದೇನೆ.",
    "hinglish": "Meri umar 23 saal hai aur main Ramanagara ke paas ek chhote gaon mein rehti hoon. Main apne gaon ki eklauti 23 saal ki mahila hoon aur school teacher ke roop mein kaam karti hoon. Mujhe ek rare neurological disorder hai aur 12 September 2026 ko Bengaluru ke St. John's Medical College Hospital mein treatment mila.",
    "kanglish": "Nanna vayassu 23 varsha mattu naanu Ramanagara hatra ondu sanna halliyalli vaasisuttiddene. Naanu nanna halliya obba 23 varshada mahile mattu school teacher aagi kelasa maaduttiddene. Nanage aparoopada neurological disorder ide mattu 12 September 2026 randu Bengalurina St. John's Medical College Hospital nalli treatment padediddene.",
}


class EmptyGLiNER:
    def predict_entities(self, text, labels, threshold):
        del text, labels, threshold
        return []


def _assert_case(monkeypatch: pytest.MonkeyPatch, language: str) -> None:
    monkeypatch.setattr(extractor, "_get_gliner_model", lambda: EmptyGLiNER())
    monkeypatch.setattr(
        pipeline,
        "get_processing_contexts",
        lambda texts: [_script_context(text) for text in texts],
    )

    result = pipeline.analyze_conversation(
        language,
        [{"id": 1, "text": CASES[language]}],
    )

    detected_types = {attribute["type"] for attribute in result["attributes"]}
    detected_indicators = {
        (indicator["indicator"], indicator["attribute_type"])
        for indicator in result["contextual_indicators"]
    }
    assert EXPECTED_TYPES <= detected_types
    assert EXPECTED_INDICATORS <= detected_indicators


def test_english_m1_extraction(monkeypatch):
    _assert_case(monkeypatch, "english")


def test_hindi_m1_extraction(monkeypatch):
    _assert_case(monkeypatch, "hindi")


def test_kannada_m1_extraction(monkeypatch):
    _assert_case(monkeypatch, "kannada")


def test_hinglish_m1_extraction(monkeypatch):
    _assert_case(monkeypatch, "hinglish")


def test_kanglish_m1_extraction(monkeypatch):
    _assert_case(monkeypatch, "kanglish")
