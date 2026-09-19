"""
detector.py — Explicit PII detection module.

Detects direct PII with Presidio and deterministic regexes.

Contextual attributes belong to extractor.py and are not detected here.

ADDRESS is treated separately from LOCATION:

- ADDRESS = street-level / house-level address information
- LOCATION = city, state, country, village, region, etc.
"""

import re

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

from backend.modules.nlp.normalizer import (
    is_garbage_span,
    is_plausible_person_name,
)


# ============================================================
# Custom recognizers for Indian IDs
# ============================================================

aadhaar_pattern = Pattern(
    name="aadhaar_pattern",
    regex=r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    score=0.85,
)

aadhaar_recognizer = PatternRecognizer(
    supported_entity="AADHAAR",
    patterns=[aadhaar_pattern],
    context=[
        "aadhaar",
        "aadhar",
        "uidai",
        "uid",
        "आधार",
        "ಆಧಾರ್",
    ],
)


pan_pattern = Pattern(
    name="pan_pattern",
    regex=r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
    score=0.85,
)

pan_recognizer = PatternRecognizer(
    supported_entity="PAN",
    patterns=[pan_pattern],
    context=[
        "pan",
        "pan card",
        "income tax",
        "पैन",
        "पैन कार्ड",
        "ಪ್ಯಾನ್",
        "ಪ್ಯಾನ್ ಕಾರ್ಡ್",
    ],
)


CUSTOM_RECOGNIZERS = [
    aadhaar_recognizer,
    pan_recognizer,
]


# ============================================================
# Identifier context
# ============================================================

AADHAAR_CONTEXT = re.compile(
    r"(?:"
    r"\b(?:aadhaar|aadhar|uidai|uid)\b"
    r"|आधार"
    r"|आधार\s+नंबर"
    r"|आधार\s+संख्या"
    r"|ಆಧಾರ್"
    r"|ಆಧಾರ್\s+ನಂಬರ್"
    r"|ಆಧಾರ್\s+ಸಂಖ್ಯೆ"
    r")",
    re.IGNORECASE,
)


BANK_ACCOUNT_CONTEXT = re.compile(
    r"(?:"
    # English
    r"\b(?:my\s+bank|mybank|bank)\s+account\b"
    r"|\baccount\s+(?:number|no\.?|#)\b"
    r"|\baccount\b"
    r"|\ba/c\b"
    r"|\ba/c\s+(?:number|no\.?|#)\b"
    r"|\bbank\s+a/c\b"
    r"|\bbank\s+a/c\s+(?:number|no\.?|#)\b"

    # Hindi
    r"|बैंक\s+खाता"
    r"|बैंक\s+खाता\s+नंबर"
    r"|बैंक\s+खाता\s+संख्या"
    r"|खाता\s+नंबर"
    r"|खाता\s+संख्या"

    # Kannada
    r"|ಬ್ಯಾಂಕ್\s+ಖಾತೆ"
    r"|ಬ್ಯಾಂಕ್\s+ಖಾತೆ\s+ನಂಬರ್"
    r"|ಬ್ಯಾಂಕ್\s+ಖಾತೆ\s+ಸಂಖ್ಯೆ"
    r"|ಖಾತೆ\s+ನಂಬರ್"
    r"|ಖಾತೆ\s+ಸಂಖ್ಯೆ"

    # Romanized Kannada / Kanglish
    r"|bank\s+khaate"
    r"|bank\s+khaate\s+number"
    r"|bank\s+khaate\s+no\.?"
    r"|khaate\s+number"
    r"|khaate\s+no\.?"
    r")",
    re.IGNORECASE,
)


# ============================================================
# Aadhaar Verhoeff validation
# ============================================================

def _verhoeff_validate(number: str) -> bool:
    """
    Validate a 12-digit Aadhaar candidate using the Verhoeff checksum.

    This is an additional signal and is not used alone
    to determine whether arbitrary numeric data is Aadhaar.
    """

    number = re.sub(r"\D", "", number)

    if len(number) != 12:
        return False

    multiplication_table = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
        [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
        [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
        [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
        [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
        [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
        [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
        [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    ]

    permutation_table = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
        [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
        [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    ]

    checksum = 0

    for i, digit in enumerate(reversed(number)):
        checksum = multiplication_table[checksum][
            permutation_table[i % 4][int(digit)]
        ]

    return checksum == 0


# ============================================================
# Shared Devanagari fragments for Hindi PERSON patterns
# ============================================================

_DEV = r"\u0900-\u0963\u0970-\u097F"

_HI_TERMINATOR_WORDS = r"(?:है|हैं|था|थी|और|तथा|ने|को|के|की|का)"

_HI_NAME_TOKEN = (
    rf"(?!{_HI_TERMINATOR_WORDS}(?![{_DEV}]))"
    rf"[{_DEV}]+"
)


# ============================================================
# Deterministic regex patterns
# ============================================================

REGEX_PATTERNS = {

    # --------------------------------------------------------
    # Direct PII
    # --------------------------------------------------------

    "PHONE_NUMBER": re.compile(
        r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"
    ),

    "EMAIL_ADDRESS": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),

    "AADHAAR": re.compile(
        r"\b\d{4}\s?\d{4}\s?\d{4}\b"
    ),

    "PAN": re.compile(
        r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b"
    ),

    "PINCODE": re.compile(
        r"\b(?:pin\s*code|pincode|postal\s*code)"
        r"\s*[:#-]?\s*(\d{6})\b"
        r"|(?<!\d)(\d{6})(?!\d)",
        re.IGNORECASE,
    ),

    "IFSC": re.compile(
        r"\b[A-Z]{4}0[A-Z0-9]{6}\b"
    ),

    # --------------------------------------------------------
    # BANK ACCOUNT
    #
    # Explicit context must appear BEFORE the number.
    # --------------------------------------------------------

    "BANK_ACCOUNT": re.compile(
        r"(?:"

        # English
        r"\b(?:my\s+bank|mybank|bank)\s+account\b"
        r"\s*(?:number|no\.?|#)?"
        r"\s*[:=-]?\s*(\d{9,18})\b"
        r"|"
        r"\baccount\s+(?:number|no\.?|#)"
        r"\s*[:=-]?\s*(\d{9,18})\b"
        r"|"
        r"\baccount\b"
        r"\s*[:=-]?\s*(\d{9,18})\b"
        r"|"
        r"\ba/c\s+(?:number|no\.?|#)"
        r"\s*[:=-]?\s*(\d{9,18})\b"
        r"|"
        r"\ba/c\b"
        r"\s*[:=-]?\s*(\d{9,18})\b"
        r"|"
        r"\bbank\s+a/c\s+(?:number|no\.?|#)"
        r"\s*[:=-]?\s*(\d{9,18})\b"

        # Hindi
        r"|"
        r"\bबैंक\s+खाता(?:\s+(?:नंबर|संख्या))?"
        r"\s*[:=-]?\s*(\d{9,18})\b"
        r"|"
        r"\bखाता\s+(?:नंबर|संख्या)"
        r"\s*[:=-]?\s*(\d{9,18})\b"

        # Kannada
        r"|"
        r"\bಬ್ಯಾಂಕ್\s+ಖಾತೆ(?:\s+(?:ನಂಬರ್|ಸಂಖ್ಯೆ))?"
        r"\s*[:=-]?\s*(\d{9,18})\b"
        r"|"
        r"\bಖಾತೆ\s+(?:ನಂಬರ್|ಸಂಖ್ಯೆ)"
        r"\s*[:=-]?\s*(\d{9,18})\b"

        # Romanized Kannada / Kanglish
        r"|"
        r"\b(?:bank\s+)?khaate"
        r"(?:\s+(?:number|no\.?))?"
        r"\s*[:=-]?\s*(\d{9,18})\b"

        r")",
        re.IGNORECASE,
    ),

    "VEHICLE_NUMBER": re.compile(
        r"\b[A-Z]{2}[-\s]?\d{1,2}[-\s]?[A-Z]{1,3}"
        r"[-\s]?\d{4}\b",
        re.IGNORECASE,
    ),

    "VOTER_ID": re.compile(
        r"\b[A-Z]{3}\d{7}\b"
    ),

    # --------------------------------------------------------
    # Street-level address
    # --------------------------------------------------------

    "ADDRESS": re.compile(
        r"""
        (?:

            # English street-level addresses
            (?:\#\s*)?
            \d{1,5}(?:st|nd|rd|th)?
            (?:\s*[-/]\s*\d{1,5})?
            \s*,?\s*
            (?:
                (?:[A-Za-z][A-Za-z0-9.'-]*\s+){0,5}
                (?:Main\s+Road|Main\s+Rd|Cross\s+Road|Cross|Road|Rd|Street|St|
                   Lane|Ln|Avenue|Ave|Boulevard|Blvd|Highway|Hwy|Layout|Nagar|
                   Colony|Block|Phase|Sector|Extension|Enclave|Residency|
                   Apartments?|Society|Circle)
            )
            (?:
                \s*,\s*
                (?:
                    [A-Za-z][A-Za-z0-9.'-]*
                    |[\u0C80-\u0CFF]+
                )
                (?:
                    \s+
                    (?:
                        [A-Za-z][A-Za-z0-9.'-]*
                        |[\u0C80-\u0CFF]+
                    )
                ){0,2}
            ){0,2}
            (?:\s*,?\s*\d{6})?

            |

            # Kannada and mixed-script block addresses
            \d{1,5}(?:ನೇ|ನೇ)?\s+ಬ್ಲಾಕ್
            (?:\s*,?\s*[\u0C80-\u0CFFA-Za-z]+)?
            (?:\s*,?\s*\d{6})?

            |

            # Hindi block addresses
            ब्लॉक\s+\d{1,5}
            (?:\s*,?\s*[\u0900-\u097FA-Za-z]+)?
            (?:\s*,?\s*\d{6})?

            |

            # Kannada / mixed-script street-level addresses
            (?:[\u0C80-\u0CFF]+(?:\s+[\u0C80-\u0CFF]+){0,2})?
            (?:ರಸ್ತೆ|ರೋಡ್|ಬೀದಿ|ಲೇನ್|ನಗರ|ನಗರಿ|ಬಡಾವಣೆ|ಕಾಲೋನಿ|ಲೇಔಟ್|ಬ್ಲಾಕ್|ಹಂತ|ವಿಸ್ತರಣೆ|ವೃತ್ತ)
            (?:
                \s*,\s*
                (?:
                    [A-Za-z][A-Za-z0-9.'-]*
                    |
                    [\u0C80-\u0CFF]+
                )
                (?:
                    \s+
                    (?:
                        [A-Za-z][A-Za-z0-9.'-]*
                        |
                        [\u0C80-\u0CFF]+
                    )
                ){0,5}
            ){0,2}
            (?:\s*,?\s*\d{6})?

        )
        """,
        re.IGNORECASE | re.VERBOSE,
    ),

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    "DATE": re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)"
        r"\s+(?:[1-9]|[12]\d|3[01]),\s+\d{4}\b",
        re.IGNORECASE,
    ),

    # --------------------------------------------------------
    # Contextual location
    # --------------------------------------------------------

    "LOCATION": re.compile(
        r"\bnear\s+"
        r"([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*)\b"
    ),

    # --------------------------------------------------------
    # English person names
    #
    # IMPORTANT:
    # We require an explicit name cue.
    #
    # We intentionally do NOT use:
    #     [A-Z][a-z]+ [A-Z][a-z]+
    #
    # as a generic PERSON detector because that would create
    # false positives such as:
    #     Bengaluru Karnataka
    #     Senior Software
    #     Private Technology
    # --------------------------------------------------------

    "PERSON_ENGLISH": re.compile(
        r"(?:"
        r"\bmy\s+name\s+is"
        r"|\bmy\s+father'?s\s+name\s+is"
        r"|\bmy\s+mother'?s\s+name\s+is"
        r"|\bhis\s+name\s+is"
        r"|\bher\s+name\s+is"
        r"|\btheir\s+name\s+is"
        r"|\bthe\s+name\s+is"
        r")"
        r"\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})"
        r"(?=\s*[,.!?;:]"
        r"|\s+(?:and|is|was|were|lives?|works?|has|have)\b"
        r"|$)",
        re.IGNORECASE,
    ),

    # --------------------------------------------------------
    # Hindi person names (Devanagari)
    #
    # Requires an explicit cue:
    #   नाम
    #   मेरा नाम / पिता का नाम / उनका नाम ...
    #   honorifics
    #   छोटी बहन
    # --------------------------------------------------------

    "PERSON_HINDI": re.compile(
        rf"(?<![{_DEV}])"
        rf"(?:नाम|श्रीमती|श्रीमान|श्री|डॉ\.?|छोटी\s+बहन)"
        rf"(?![{_DEV}])"
        rf"\s+(?:है\s+)?\*{{0,2}}"
        rf"({_HI_NAME_TOKEN}(?:\s+{_HI_NAME_TOKEN}){{1,2}})"
        rf"(?=\s*{_HI_TERMINATOR_WORDS}(?![{_DEV}])|\s*[।,.]|\s*$)"
    ),

    # --------------------------------------------------------
    # Roman Hindi person names
    # --------------------------------------------------------

    "PERSON_HINDI_ROMAN": re.compile(
        r"(?:mera\s+naam|uska\s+naam|unka\s+naam|"
        r"mere\s+(?:pita|father)\s+ka\s+naam|"
        r"meri\s+(?:maa|mother)\s+ka\s+naam)\s+"
        r"\*{0,2}"
        r"(?-i:([A-Z][a-z]+(?:\s+[A-Z][a-z]+)))"
        r"(?=\s+(?:hai|aur|ka|ki)|[,.]|$)",
        re.IGNORECASE,
    ),

    # --------------------------------------------------------
    # Kannada + Kanglish person-name patterns
    # --------------------------------------------------------

    "PERSON_KANNADA": re.compile(
        r"(?:"
        r"ನನ್ನ\s+ತಂದೆಯ\s+ಹೆಸರು"
        r"|ನನ್ನ\s+ತಾಯಿಯ\s+ಹೆಸರು"
        r"|ನನ್ನ\s+ಹೆಸರು"
        r"|ತಂದೆಯ\s+ಹೆಸರು"
        r"|ತಾಯಿಯ\s+ಹೆಸರು"
        r"|ಹೆಸರು"

        # Kanglish / Romanized Kannada
        r"|nanna\s+tandeya\s+hesaru"
        r"|nanna\s+taayiya\s+hesaru"
        r"|nanna\s+hesaru"
        r"|tandeya\s+hesaru"
        r"|taayiya\s+hesaru"
        r"|hesaru"
        r")"
        r"\s+"
        r"("
        r"[\u0C80-\u0CFF]+(?:\s+[\u0C80-\u0CFF]+)?"
        r"|"
        r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}"
        r")"
        r"(?=\s+(?:ಮತ್ತು|ನಾನು|ಇವರು|ಅವರು|mattu|naanu|ivaru|avaru)|[,.]|$)"
    ),
}


PERSON_PATTERN_NAMES = {
    "PERSON_ENGLISH",
    "PERSON_HINDI",
    "PERSON_HINDI_ROMAN",
    "PERSON_KANNADA",
}


# ============================================================
# Confidence and specificity
# ============================================================

REGEX_CONFIDENCE = {
    "DATE": 0.99,
    "LOCATION": 0.94,
    "ADDRESS": 0.95,
    "PERSON_ENGLISH": 0.95,
    "PERSON_HINDI": 0.95,
    "PERSON_HINDI_ROMAN": 0.95,
    "PERSON_KANNADA": 0.95,
}

REGEX_SPECIFICITY = {
    "DATE": 1.0,
    "LOCATION": 0.85,
    "ADDRESS": 1.0,
    "PERSON_ENGLISH": 1.0,
    "PERSON_HINDI": 1.0,
    "PERSON_HINDI_ROMAN": 1.0,
    "PERSON_KANNADA": 1.0,
}


# ============================================================
# Build Presidio analyzer once
# ============================================================

analyzer = AnalyzerEngine()

for recognizer in CUSTOM_RECOGNIZERS:
    analyzer.registry.add_recognizer(recognizer)


# ============================================================
# Presidio entities used by KavachAI
# ============================================================

TARGET_ENTITIES = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "LOCATION",
    "AADHAAR",
    "PAN",
    "PINCODE",
    "IFSC",
    "BANK_ACCOUNT",
    "VEHICLE_NUMBER",
    "VOTER_ID",
]


# ============================================================
# False-positive filtering
# ============================================================

PERSON_FALSE_POSITIVES = {
    "aadhaar",
    "aadhar",
    "pan",
    "uidai",
}

NON_ENTITY_SPANS = {
    "में",
    "ಮತ್ತು",
    "ಯಲ್ಲಿ",
    "yalli",
    "nalli",
    "meri",
    "nanna",
}

ROMAN_HINDI_MARKERS = {
    "mera",
    "meri",
    "mujhe",
    "saal",
    "gaon",
    "eklauti",
    "hoon",
    "kaam",
    "naam",
    "umar",
    "phone",
    "number",
    "pita",
    "maa",
}

ROMAN_KANNADA_MARKERS = {
    "nanna",
    "nanu",
    "nanage",
    "varsha",
    "halli",
    "halliyalli",
    "hatra",
    "mattu",
    "hesaru",
    "tandeya",
    "kelasa",
    "madtini",
    "ide",
    "phone",
}


# ============================================================
# Identifier context helpers
# ============================================================

def _identifier_context(
    text: str,
    start: int,
    end: int,
) -> str:
    """
    Return ONLY the text immediately before the identifier.

    Context after the candidate is intentionally excluded.
    """

    return text[max(0, start - 80):start]


# ============================================================
# Aadhaar / Bank-account classifier
# ============================================================

def _classify_numeric_identifier(
    text: str,
    match: re.Match,
    entity_type: str,
) -> str | None:
    """
    Resolve Aadhaar vs bank-account overlap.

    Priority:

    1. Explicit bank-account context BEFORE the number
    2. Explicit Aadhaar context BEFORE the number
    3. Aadhaar Verhoeff validation
    4. Otherwise reject
    """

    value = match.group(0).strip()

    context = _identifier_context(
        text,
        match.start(),
        match.end(),
    )

    has_aadhaar_context = bool(
        AADHAAR_CONTEXT.search(context)
    )

    has_bank_context = bool(
        BANK_ACCOUNT_CONTEXT.search(context)
    )

    if entity_type == "BANK_ACCOUNT":
        if has_bank_context:
            return "BANK_ACCOUNT"

        return None

    if entity_type == "AADHAAR":
        if has_bank_context:
            return "BANK_ACCOUNT"

        if has_aadhaar_context:
            return "AADHAAR"

        digits = re.sub(r"\D", "", value)

        if _verhoeff_validate(digits):
            return "AADHAAR"

        return None

    return entity_type


# ============================================================
# Deterministic regex detection
# ============================================================

def _regex_entities(
    text: str,
) -> list[dict[str, object]]:

    detected = []

    for pattern_name, pattern in REGEX_PATTERNS.items():

        for match in pattern.finditer(text):

            value = match.group(0).strip()

            # For patterns with capturing groups, use the
            # captured value.
            if match.groups():

                captured = next(
                    (
                        group
                        for group in match.groups()
                        if group is not None
                    ),
                    None,
                )

                if captured is not None:
                    value = captured.strip()

            # ------------------------------------------------
            # ADDRESS cleanup
            # ------------------------------------------------

            if pattern_name == "ADDRESS":

                block_address = re.match(
                    r"^((?:\d+(?:st|nd|rd|th)?\s+Block"
                    r"|\d+(?:ನೇ|ನೇ)?\s+ಬ್ಲಾಕ್)"
                    r"(?:\s*,\s*[A-Za-z\u0900-\u097F\u0C80-\u0CFF]+)?)",
                    value,
                    re.IGNORECASE,
                )

                if block_address:
                    value = block_address.group(1).strip()

            # ------------------------------------------------
            # Resolve Aadhaar vs bank-account overlap
            # ------------------------------------------------

            if pattern_name in {
                "AADHAAR",
                "BANK_ACCOUNT",
            }:

                resolved_type = _classify_numeric_identifier(
                    text,
                    match,
                    pattern_name,
                )

                if resolved_type is None:
                    continue

                entity_type = resolved_type

            elif pattern_name in PERSON_PATTERN_NAMES:

                entity_type = "PERSON"

            else:

                entity_type = pattern_name

            # ------------------------------------------------
            # PINCODE needs nearby context
            # ------------------------------------------------

            if entity_type == "PINCODE":

                context = text[
                    max(0, match.start() - 12):
                    match.end()
                ]

                if not re.search(
                    r"(?:pin|postal)",
                    context,
                    re.IGNORECASE,
                ):
                    continue

            # ------------------------------------------------
            # PERSON-specific filtering
            # ------------------------------------------------

            if entity_type == "PERSON":

                if (
                    value.casefold()
                    in PERSON_FALSE_POSITIVES

                    or value.casefold()
                    in NON_ENTITY_SPANS

                    or is_garbage_span(value)

                    or not is_plausible_person_name(value)
                ):
                    continue

            # ------------------------------------------------
            # Ignore known garbage spans
            # ------------------------------------------------

            if is_garbage_span(value):
                continue

            detected.append(
                {
                    "type": entity_type,
                    "value": value,
                    "confidence": REGEX_CONFIDENCE.get(
                        pattern_name,
                        0.95,
                    ),
                    "specificity": REGEX_SPECIFICITY.get(
                        pattern_name,
                        1.0,
                    ),
                }
            )

    return detected


# ============================================================
# Main detector
# ============================================================

def detect_entities(
    text: str,
    language_context: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """
    Detect explicit PII entities in a piece of text.

    ADDRESS is detected separately from LOCATION.

    Aadhaar vs bank account:

    - Bank-account context BEFORE the number wins.
    - Aadhaar context identifies Aadhaar.
    - Without context, Aadhaar requires Verhoeff validation.
    """

    if not text or not text.strip():
        return []

    observed_scripts = (
        set(
            language_context.get(
                "observed_scripts",
                [],
            )
        )
        if language_context
        else set()
    )

    indic_or_mixed = (
        bool(
            observed_scripts
            & {"hi", "kn"}
        )

        or bool(
            re.search(
                r"[\u0900-\u097F\u0C80-\u0CFF]",
                text,
            )
        )

        or len(
            set(
                re.findall(
                    r"\b[a-zA-Z]+\b",
                    text.casefold(),
                )
            )
            & ROMAN_HINDI_MARKERS
        ) >= 2

        or len(
            set(
                re.findall(
                    r"\b[a-zA-Z]+\b",
                    text.casefold(),
                )
            )
            & ROMAN_KANNADA_MARKERS
        ) >= 2

        or bool(
            language_context
            and language_context.get(
                "is_code_mixed"
            )
        )
    )

    # --------------------------------------------------------
    # Presidio detection
    # --------------------------------------------------------

    results = analyzer.analyze(
        text=text,
        entities=TARGET_ENTITIES,
        language="en",
    )

    # --------------------------------------------------------
    # Deterministic detection
    # --------------------------------------------------------

    detected = _regex_entities(text)

    deterministic_types = {
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",
        "AADHAAR",
        "PAN",
        "PINCODE",
        "IFSC",
        "BANK_ACCOUNT",
        "VEHICLE_NUMBER",
        "VOTER_ID",
    }

    for r in results:

        value = text[r.start:r.end]

        # Deterministic detector is authoritative for
        # direct PII types handled by regex.
        if r.entity_type in deterministic_types:
            continue

        # For Indic/code-mixed text, avoid Presidio's
        # unreliable English PERSON/LOCATION model results.
        if (
            indic_or_mixed
            and r.entity_type in {
                "PERSON",
                "LOCATION",
            }
        ):
            continue

        if r.entity_type == "PERSON":

            if (
                value.lower()
                in PERSON_FALSE_POSITIVES

                or value.casefold()
                in NON_ENTITY_SPANS

                or is_garbage_span(value)

                or any(
                    token in NON_ENTITY_SPANS
                    for token in re.findall(
                        r"\b[\w']+\b",
                        value.casefold(),
                    )
                )
            ):
                continue

        if r.entity_type == "LOCATION":

            # A PAN-shaped value must never become LOCATION.
            if re.fullmatch(
                r"[A-Z]{5}[0-9]{4}[A-Z]",
                value.strip(),
                re.IGNORECASE,
            ):
                continue

            if value.casefold() in NON_ENTITY_SPANS:
                continue

            if is_garbage_span(value):
                continue

        if is_garbage_span(value):
            continue

        detected.append(
            {
                "type": r.entity_type,
                "value": value,
                "confidence": round(
                    float(r.score),
                    2,
                ),
                "specificity": 1.0,
            }
        )

    # --------------------------------------------------------
    # Final deduplication
    # --------------------------------------------------------

    unique_detected = []

    seen = set()

    for entity in detected:

        key = (
            entity["type"],
            str(
                entity["value"]
            ).strip().casefold(),
        )

        if key in seen:
            continue

        seen.add(key)
        unique_detected.append(entity)

    return unique_detected


# ============================================================
# Quick manual test
# ============================================================

if __name__ == "__main__":

    sample = (
        "My name is Aarav Sharma, and I live at "
        "42, MG Road, Bengaluru, Karnataka 560001. "
        "My phone number is 9876543210 and my email is "
        "aarav.sharma@example.com. "
        "My Aadhaar number is 2345 6789 0123 and "
        "my bank account number is 123456789012. "
        "My PAN is ABCDE1234F and my IFSC is SBIN0001234. "
        "My vehicle registration number is KA-09-AB-4721 "
        "and my voter ID is ABC1234567. "
        "My father's name is Ramesh Sharma."
    )

    print("\nDetected entities:\n")

    for entity in detect_entities(sample):
        print(entity)