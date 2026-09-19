"""
extractor.py — Contextual attribute extraction module.

Extracts contextual, potentially re-identifying attributes from text.

detector.py
    -> Direct PII
    -> PERSON, PHONE, EMAIL, Aadhaar, PAN, etc.

extractor.py
    -> AGE
    -> HEALTH
    -> OCCUPATION
    -> EDUCATION
    -> FACILITY
    -> contextual uniqueness indicators

The extractor must never classify duration as a person's age.
"""

from __future__ import annotations

import re
from typing import Any


# ============================================================
# AGE PATTERNS
# ============================================================

MULTILINGUAL_AGE_PATTERNS = [
    # --------------------------------------------------------
    # English
    # --------------------------------------------------------
    re.compile(
        r"\b(?:i'm|i\s+am|i\s+was)"
        r"\s+(\d{1,3})"
        r"\s+(?:years?|yrs?)"
        r"(?:\s+old)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:my\s+)?age"
        r"\s*(?:is|:|=|-)"
        r"\s*(\d{1,3})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\baged\s+(\d{1,3})\b",
        re.IGNORECASE,
    ),

    # --------------------------------------------------------
    # Hindi / Devanagari
    # --------------------------------------------------------
    re.compile(
        r"(?:"
        r"मेरी?\s+उम्र"
        r"|"
        r"मेरी?\s+आयु"
        r"|"
        r"उम्र"
        r"|"
        r"आयु"
        r")"
        r"\s*(?:है|हैं|होने)?"
        r"\s*[:=-]?\s*"
        r"(\d{1,3})"
        r"\s*"
        r"(?:वर्ष|साल)"
        r"(?:\s+(?:है|हैं))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"मैं"
        r"\s+(\d{1,3})"
        r"\s*"
        r"(?:वर्ष|साल)"
        r"\s*"
        r"(?:का|की|के)?"
        r"\s*"
        r"(?:हूँ|हुन|है|हैं|था|थी)?",
        re.IGNORECASE,
    ),

    # --------------------------------------------------------
    # Roman Hindi / Hinglish
    # --------------------------------------------------------
    re.compile(
        r"(?:"
        r"meri\s+umar"
        r"|"
        r"meri\s+age"
        r"|"
        r"umar"
        r"|"
        r"age"
        r")"
        r"\s*(?:hai|h)?"
        r"\s*[:=-]?\s*"
        r"(\d{1,3})"
        r"\s*"
        r"(?:saal|sal|years?)?"
        r"\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"main"
        r"\s+(\d{1,3})"
        r"\s*"
        r"(?:saal|sal|years?)"
        r"\s*"
        r"(?:ka|ki|ke)?"
        r"\s*"
        r"(?:hoon|hun|hai|hain)?",
        re.IGNORECASE,
    ),

    # --------------------------------------------------------
    # Kannada
    # --------------------------------------------------------
    re.compile(
        r"(?:"
        r"ನನ್ನ\s+ವಯಸ್ಸು"
        r"|"
        r"ನನ್ನ\s+ಆಯುಸ್ಸು"
        r"|"
        r"ವಯಸ್ಸು"
        r"|"
        r"ಆಯುಸ್ಸು"
        r")"
        r"\s*"
        r"[:=-]?\s*"
        r"(\d{1,3})"
        r"\s*"
        r"(?:ವರ್ಷ|ವರ್ಷದ)"
        r"(?:\s*(?:ವಯಸ್ಸು|ವಯಸ್ಸಾಗಿದೆ|ವಯಸ್ಸು\s*ಆಗಿದೆ))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"ನನಗೆ"
        r"\s+(\d{1,3})"
        r"\s*"
        r"ವರ್ಷ"
        r"\s*"
        r"(?:"
        r"ವಯಸ್ಸಾಗಿದೆ"
        r"|"
        r"ವಯಸ್ಸು"
        r"|"
        r"ವಯಸ್ಸು\s*ಆಗಿದೆ"
        r")",
        re.IGNORECASE,
    ),
    re.compile(
        r"(\d{1,3})"
        r"\s*"
        r"ವರ್ಷ"
        r"\s*"
        r"ವಯಸ್ಸಾಗಿದೆ",
        re.IGNORECASE,
    ),

    # --------------------------------------------------------
    # Roman Kannada / Kanglish
    # --------------------------------------------------------
    re.compile(
        r"(?:"
        r"nanna\s+vayassu"
        r"|"
        r"nanna\s+age"
        r"|"
        r"vayassu"
        r"|"
        r"age"
        r")"
        r"\s*[:=-]?\s*"
        r"(\d{1,3})"
        r"\s*"
        r"(?:varsha|varshada|years?)"
        r"(?:\s*(?:vayassagide|vayassu|vayassu\s*agide))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"nanage"
        r"\s+(\d{1,3})"
        r"\s*"
        r"varsha"
        r"\s*"
        r"(?:"
        r"vayassagide"
        r"|"
        r"vayassu"
        r"|"
        r"vayassu\s*agide"
        r")",
        re.IGNORECASE,
    ),
]


# ============================================================
# AGE DURATION / EXPERIENCE FILTERS
# ============================================================

AGE_DURATION_PATTERNS = [
    # English
    re.compile(
        r"\b"
        r"(?:for|since|over|more\s+than|nearly|about)?"
        r"\s*\d{1,3}\s+"
        r"(?:years?|yrs?)"
        r"\s+"
        r"(?:"
        r"of\s+(?:experience|teaching|work|service)"
        r"|"
        r"experience"
        r"|"
        r"of\s+teaching"
        r"|"
        r"since\s+graduation"
        r")",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d{1,3}\s+(?:years?|yrs?)\s+"
        r"(?:ago|later|before|after)\b",
        re.IGNORECASE,
    ),

    # Hindi
    re.compile(
        r"\d{1,3}\s*(?:वर्ष|साल)"
        r"\s*"
        r"(?:से|का\s+अनुभव|का\s+अध्यापन|से\s+अध्यापन)",
        re.IGNORECASE,
    ),

    # Roman Hindi
    re.compile(
        r"\d{1,3}\s*"
        r"(?:saal|sal)"
        r"\s*"
        r"(?:se|ka\s+anubhav|kaam\s+ka\s+anubhav)",
        re.IGNORECASE,
    ),

    # Kannada
    re.compile(
        r"\d{1,3}\s*"
        r"ವರ್ಷ"
        r"\s*"
        r"(?:"
        r"ಗಳಿಂದ"
        r"|"
        r"ಅನುಭವ"
        r"|"
        r"ಅಧ್ಯಾಪನ"
        r"|"
        r"ಕೆಲಸ"
        r")",
        re.IGNORECASE,
    ),

    # Kanglish
    re.compile(
        r"\d{1,3}\s*"
        r"varsha"
        r"\s*"
        r"(?:"
        r"inda"
        r"|"
        r"anubhava"
        r"|"
        r"kelasa"
        r"|"
        r"adhyapana"
        r")",
        re.IGNORECASE,
    ),
]


# ============================================================
# CONTEXTUAL ATTRIBUTE PATTERNS
# ============================================================

HEALTH_PATTERNS = [
    # English
    re.compile(
        r"\b(?:"
        r"rare\s+(?:neurological|medical|genetic)\s+(?:disorder|condition|disease)"
        r"|"
        r"(?:diabetes|hypertension|high\s+blood\s+pressure)"
        r"|"
        r"(?:neurological|chronic|genetic)\s+(?:disorder|condition|disease)"
        r")\b",
        re.IGNORECASE,
    ),

    # Hindi
    re.compile(
        r"(?:"
        r"दुर्लभ\s+न्यूरोलॉजिकल\s+(?:बीमारी|रोग|विकार)"
        r"|"
        r"उच्च\s+रक्तचाप"
        r"|"
        r"मधुमेह"
        r"|"
        r"न्यूरोलॉजिकल\s+(?:बीमारी|रोग|विकार)"
        r")",
        re.IGNORECASE,
    ),

    # Kannada
    re.compile(
        r"(?:"
        r"ಅಪರೂಪದ\s+ನರವೈಜ್ಞಾನಿಕ\s+(?:ಕಾಯಿಲೆ|ರೋಗ|ಸ್ಥಿತಿ)"
        r"|"
        r"ಮಧುಮೇಹ"
        r"|"
        r"ಅಧಿಕ\s+ರಕ್ತದೊತ್ತಡ"
        r"|"
        r"ನರವೈಜ್ಞಾನಿಕ\s+(?:ಕಾಯಿಲೆ|ರೋಗ|ಸ್ಥಿತಿ)"
        r")",
        re.IGNORECASE,
    ),
]


OCCUPATION_PATTERNS = [
    # English
    re.compile(
        r"\b(?:"
        r"senior\s+software\s+engineer"
        r"|"
        r"software\s+engineer"
        r"|"
        r"doctor"
        r"|"
        r"teacher"
        r"|"
        r"professor"
        r"|"
        r"engineer"
        r"|"
        r"developer"
        r"|"
        r"lawyer"
        r")\b",
        re.IGNORECASE,
    ),

    # Hindi
    re.compile(
        r"(?:"
        r"वरिष्ठ\s+गणित\s+शिक्षक"
        r"|"
        r"गणित\s+शिक्षक"
        r"|"
        r"शिक्षक"
        r"|"
        r"डॉक्टर"
        r"|"
        r"इंजीनियर"
        r")",
        re.IGNORECASE,
    ),

    # Kannada
    re.compile(
        r"(?:"
        r"ಹಿರಿಯ\s+ಸಾಫ್ಟ್‌ವೇರ್\s+ಎಂಜಿನಿಯರ್"
        r"|"
        r"ಸಾಫ್ಟ್‌ವೇರ್\s+ಎಂಜಿನಿಯರ್"
        r"|"
        r"ಶಿಕ್ಷಕ"
        r"|"
        r"ಅಧ್ಯಾಪಕ"
        r"|"
        r"ವೈದ್ಯ"
        r"|"
        r"ಎಂಜಿನಿಯರ್"
        r")",
        re.IGNORECASE,
    ),
]


EDUCATION_PATTERNS = [
    # English
    re.compile(
        r"\b(?:"
        r"master'?s\s+degree"
        r"|"
        r"bachelor'?s\s+degree"
        r"|"
        r"phd"
        r"|"
        r"doctorate"
        r")\b",
        re.IGNORECASE,
    ),

    # Hindi
    re.compile(
        r"(?:"
        r"स्नातकोत्तर"
        r"|"
        r"स्नातक\s+की\s+पढ़ाई"
        r"|"
        r"मास्टर'?s\s+डिग्री"
        r")",
        re.IGNORECASE,
    ),

    # Kannada
    re.compile(
        r"(?:"
        r"ಮಾಸ್ಟರ್ಸ್\s+ಪದವಿ"
        r"|"
        r"ಸ್ನಾತಕೋತ್ತರ"
        r"|"
        r"ಸ್ನಾತಕ\s+ಪದವಿ"
        r")",
        re.IGNORECASE,
    ),
]


FACILITY_PATTERNS = [
    # English
    re.compile(
        r"\b(?:"
        r"private\s+hospital"
        r"|"
        r"government\s+hospital"
        r"|"
        r"district\s+hospital"
        r"|"
        r"healthcare\s+facility"
        r"|"
        r"hospital"
        r"|"
        r"clinic"
        r")\b",
        re.IGNORECASE,
    ),

    # Hindi
    re.compile(
        r"(?:"
        r"निजी\s+अस्पताल"
        r"|"
        r"सरकारी\s+अस्पताल"
        r"|"
        r"जिला\s+अस्पताल"
        r"|"
        r"अस्पताल"
        r"|"
        r"क्लिनिक"
        r")",
        re.IGNORECASE,
    ),

    # Kannada
    re.compile(
        r"(?:"
        r"ಖಾಸಗಿ\s+ಆಸ್ಪತ್ರೆ"
        r"|"
        r"ಸರ್ಕಾರಿ\s+ಆಸ್ಪತ್ರೆ"
        r"|"
        r"ಜಿಲ್ಲಾ\s+ಆಸ್ಪತ್ರೆ"
        r"|"
        r"ಆಸ್ಪತ್ರೆ"
        r"|"
        r"ಆರೋಗ್ಯ\s+ಕೇಂದ್ರ"
        r")",
        re.IGNORECASE,
    ),
]


# ============================================================
# UNIQUENESS INDICATORS
# ============================================================

UNIQUENESS_PATTERNS = [
    # English
    (
        re.compile(
            r"\b(?:"
            r"rare|only|unique|few|very\s+few|"
            r"one\s+of\s+a\s+kind|the\s+only"
            r")\b",
            re.IGNORECASE,
        ),
        "uniqueness",
    ),

    # Hindi
    (
        re.compile(
            r"(?:"
            r"दुर्लभ|केवल|एकमात्र|बहुत\s+कम|"
            r"कुछ\s+ही|इकलौता|इकलौती"
            r")",
            re.IGNORECASE,
        ),
        "uniqueness",
    ),

    # Kannada
    (
        re.compile(
            r"(?:"
            r"ಅಪರೂಪದ|ಒಬ್ಬನೇ|ಒಬ್ಬಳೇ|"
            r"ಒಂದೇ|ಕೆಲವೇ|ಬಹಳ\s+ಕಡಿಮೆ|ಏಕೈಕ"
            r")",
            re.IGNORECASE,
        ),
        "uniqueness",
    ),

    # Kanglish
    (
        re.compile(
            r"\b(?:"
            r"rare|only|unique|few|"
            r"ekai|obbaney|obbale|kelave"
            r")\b",
            re.IGNORECASE,
        ),
        "uniqueness",
    ),
]


# ============================================================
# AGE VALIDATION
# ============================================================

def _valid_age(value: str) -> bool:
    """
    Conservative human-age range: 1–120.
    """
    try:
        age = int(value)
    except (TypeError, ValueError):
        return False

    return 1 <= age <= 120


def _is_duration_context(
    text: str,
    start: int,
    end: int,
) -> bool:
    """
    Determine whether an age candidate is actually part of
    a duration / experience expression.
    """
    window_start = max(0, start - 100)
    window_end = min(len(text), end + 120)

    window = text[window_start:window_end]
    folded = window.casefold()

    duration_words = [
        # English
        "years of teaching",
        "years of experience",
        "years experience",
        "years of work",
        "years of service",
        "years since",
        "years ago",
        "years later",
        "for years",

        # Hindi
        "वर्षों से",
        "वर्ष से",
        "सालों से",
        "साल से",
        "वर्ष का अनुभव",
        "साल का अनुभव",
        "वर्षों का अनुभव",
        "सालों का अनुभव",
        "अध्यापन के क्षेत्र में",

        # Kannada
        "ವರ್ಷಗಳಿಂದ",
        "ವರ್ಷದಿಂದ",
        "ವರ್ಷಗಳ",
        "ಅನುಭವ",
        "ಅಧ್ಯಾಪನ",
        "ಕೆಲಸ",

        # Kanglish
        "varshagalinda",
        "varshagalu",
        "varshadinda",
        "anubhava",
        "adhyapana",
        "kelasa",
    ]

    for marker in duration_words:
        if marker.casefold() in folded:
            return True

    for pattern in AGE_DURATION_PATTERNS:
        if pattern.search(window):
            return True

    # English duration forms
    if re.search(
        r"\bfor\s+\d{1,3}\s+(?:years?|yrs?)\b",
        window,
        re.IGNORECASE,
    ):
        return True

    if re.search(
        r"\b\d{1,3}\s+(?:years?|yrs?)\s+of\s+"
        r"(?:teaching|experience|work|service)\b",
        window,
        re.IGNORECASE,
    ):
        return True

    # Hindi duration
    if re.search(
        r"\d{1,3}\s*(?:वर्ष|साल)"
        r"\s*(?:से|का\s+अनुभव|का\s+अध्यापन)",
        window,
        re.IGNORECASE,
    ):
        return True

    # Kannada duration
    if re.search(
        r"\d{1,3}\s*ವರ್ಷ"
        r"\s*(?:ಗಳಿಂದ|ದಿಂದ|ಗಳ|ಅನುಭವ|ಅಧ್ಯಾಪನ|ಕೆಲಸ)",
        window,
        re.IGNORECASE,
    ):
        return True

    return False


# ============================================================
# AGE EXTRACTION
# ============================================================

def _find_age(text: str) -> dict[str, Any] | None:
    """
    Find one explicit age expression.
    """
    if not text or not text.strip():
        return None

    for pattern in MULTILINGUAL_AGE_PATTERNS:
        for match in pattern.finditer(text):
            groups = [
                group
                for group in match.groups()
                if group is not None
            ]

            if not groups:
                continue

            value = groups[0].strip()

            if not _valid_age(value):
                continue

            if _is_duration_context(
                text,
                match.start(),
                match.end(),
            ):
                continue

            return {
                "value": value,
                "confidence": 0.98,
                "specificity": 1.0,
                "indicator": "exact_age",
            }

    return None


# ============================================================
# GENERIC CONTEXTUAL ATTRIBUTE HELPER
# ============================================================

def _extract_pattern_attributes(
    text: str,
    patterns: list[re.Pattern],
    attribute_type: str,
    confidence: float,
    specificity: float,
    source_message_id: int,
) -> list[dict[str, Any]]:
    attributes: list[dict[str, Any]] = []

    for pattern in patterns:
        for match in pattern.finditer(text):
            value = match.group(0).strip()

            if not value:
                continue

            attributes.append(
                {
                    "type": attribute_type,
                    "value": value,
                    "confidence": confidence,
                    "specificity": specificity,
                    "source_message_id": source_message_id,
                }
            )

    return attributes


# ============================================================
# CONTEXTUAL UNIQUENESS INDICATORS
# ============================================================

def _find_contextual_indicators(
    text: str,
) -> list[dict[str, str]]:
    indicators: list[dict[str, str]] = []

    if not text:
        return indicators

    for pattern, indicator_type in UNIQUENESS_PATTERNS:
        for match in pattern.finditer(text):
            indicators.append(
                {
                    "indicator": indicator_type,
                    "attribute_type": "CONTEXTUAL",
                    "value": match.group(0),
                }
            )

    return indicators


# ============================================================
# CONTEXTUAL ATTRIBUTE EXTRACTION
# ============================================================

def extract_contextual_attributes(
    text: str,
    source_message_id: int = 1,
) -> list[dict[str, Any]]:
    """
    Extract contextual attributes from one message.

    Every returned attribute follows the M1 normalization contract.
    """

    if not text or not text.strip():
        return []

    attributes: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    age = _find_age(text)

    if age is not None:
        attributes.append(
            {
                "type": "AGE",
                "value": age["value"],
                "confidence": age["confidence"],
                "specificity": age["specificity"],
                "source_message_id": source_message_id,
            }
        )

    # --------------------------------------------------------
    # HEALTH
    # --------------------------------------------------------

    attributes.extend(
        _extract_pattern_attributes(
            text,
            HEALTH_PATTERNS,
            "HEALTH",
            confidence=0.94,
            specificity=0.93,
            source_message_id=source_message_id,
        )
    )

    # --------------------------------------------------------
    # OCCUPATION
    # --------------------------------------------------------

    attributes.extend(
        _extract_pattern_attributes(
            text,
            OCCUPATION_PATTERNS,
            "OCCUPATION",
            confidence=0.90,
            specificity=0.75,
            source_message_id=source_message_id,
        )
    )

    # --------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------

    attributes.extend(
        _extract_pattern_attributes(
            text,
            EDUCATION_PATTERNS,
            "EDUCATION",
            confidence=0.90,
            specificity=0.70,
            source_message_id=source_message_id,
        )
    )

    # --------------------------------------------------------
    # FACILITY
    # --------------------------------------------------------

    attributes.extend(
        _extract_pattern_attributes(
            text,
            FACILITY_PATTERNS,
            "FACILITY",
            confidence=0.90,
            specificity=0.80,
            source_message_id=source_message_id,
        )
    )

    return attributes


# ============================================================
# SENSITIVE DATA EXTRACTION
# ============================================================

def extract_sensitive_data(
    messages: list[dict[str, Any]],
    language_contexts: Any = None,
) -> dict[str, Any]:
    """
    Main API used by backend.modules.nlp.pipeline.

    Expected input:

        messages = [
            {"id": 1, "text": "..."},
            {"id": 2, "text": "..."}
        ]

    Returns:

        {
            "attributes": [...],
            "contextual_indicators": [...]
        }

    Direct PII remains the responsibility of detector.py.
    """

    if not messages:
        return {
            "attributes": [],
            "contextual_indicators": [],
        }

    all_attributes: list[dict[str, Any]] = []
    all_indicators: list[dict[str, str]] = []

    for index, message in enumerate(messages, start=1):

        if isinstance(message, dict):
            text = message.get("text", "")
            source_message_id = message.get("id", index)
        else:
            text = str(message)
            source_message_id = index

        if not isinstance(text, str):
            continue

        if not text.strip():
            continue

        # Contextual attributes
        attributes = extract_contextual_attributes(
            text=text,
            source_message_id=source_message_id,
        )

        all_attributes.extend(attributes)

        # Contextual indicators
        indicators = _find_contextual_indicators(text)
        all_indicators.extend(indicators)

    return {
        "attributes": all_attributes,
        "contextual_indicators": all_indicators,
    }


# ============================================================
# BACKWARD-COMPATIBILITY API
# ============================================================

def extract_attributes(
    text: str,
    source_message_id: int = 1,
) -> list[dict[str, Any]]:
    """
    Backward-compatible single-message API.

    Unlike extract_sensitive_data(), this function returns
    only the attribute list.
    """

    return extract_contextual_attributes(
        text=text,
        source_message_id=source_message_id,
    )


# ============================================================
# QUICK MANUAL TEST
# ============================================================

if __name__ == "__main__":

    test_cases = [
        (
            "English",
            "I am 42 years old and work as a software engineer.",
        ),
        (
            "Hindi",
            "मेरी उम्र 52 वर्ष है और मैं बेंगलुरु में रहता हूँ।",
        ),
        (
            "Hinglish",
            "Meri umar 37 saal hai aur main Bengaluru mein rehta hoon.",
        ),
        (
            "Kannada",
            "ನನಗೆ 42 ವರ್ಷ ವಯಸ್ಸಾಗಿದೆ ಮತ್ತು ನಾನು ಬೆಂಗಳೂರಿನಲ್ಲಿ ವಾಸಿಸುತ್ತಿದ್ದೇನೆ.",
        ),
        (
            "Kanglish",
            "Nanage 42 varsha vayassagide mattu naanu Bengaluru alli iddini.",
        ),
        (
            "Duration — must NOT be AGE",
            "He has been teaching for 25 years.",
        ),
        (
            "Duration — must NOT be AGE",
            "He has 25 years of teaching experience.",
        ),
        (
            "Kannada duration — must NOT be AGE",
            "ಅವರು ಕಳೆದ 25 ವರ್ಷಗಳಿಂದ ಅಧ್ಯಾಪನ ಮಾಡುತ್ತಿದ್ದಾರೆ.",
        ),
        (
            "Kannada health",
            "ನನಗೆ ಅಪರೂಪದ ನರವೈಜ್ಞಾನಿಕ ಕಾಯಿಲೆ ಇದೆ.",
        ),
    ]

    print("\nCONTEXTUAL EXTRACTION TESTS\n")

    for label, text in test_cases:

        result = extract_contextual_attributes(text)

        print(f"{label}:")
        print(f"  TEXT   : {text}")
        print(f"  RESULT : {result}")
        print()