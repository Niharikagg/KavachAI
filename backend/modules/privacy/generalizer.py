"""
generalizer.py

Transformation rules for each attribute type.

Each transformation returns:

{
    "new_value": str,
    "info_loss_cost": float
}

The optimizer uses these costs to choose transformations.
"""

from __future__ import annotations


# ============================================================
# LANGUAGE HELPERS
# ============================================================

def _is_kannada(text: str) -> bool:
    if not text:
        return False

    return any("\u0C80" <= char <= "\u0CFF" for char in text)


def _is_hindi(text: str) -> bool:
    if not text:
        return False

    return any("\u0900" <= char <= "\u097F" for char in text)


# ============================================================
# AGE
# ============================================================

_AGE_BANDS_EN = [
    (0, 12, "a child"),
    (13, 17, "a teenager"),
    (18, 25, "a young adult"),
    (26, 35, "an adult in their late twenties or thirties"),
    (36, 50, "a middle-aged adult"),
    (51, 65, "an adult in their fifties or sixties"),
    (66, 120, "a senior"),
]

_AGE_BANDS_KN = [
    (0, 12, "ಮಗು"),
    (13, 17, "ಹದಿಹರೆಯದವರು"),
    (18, 25, "ಯುವ ವಯಸ್ಕ"),
    (26, 35, "ಇಪ್ಪತ್ತರ ಕೊನೆಯ ಅಥವಾ ಮೂವತ್ತರ ವಯಸ್ಸಿನ ವಯಸ್ಕ"),
    (36, 50, "ಮಧ್ಯವಯಸ್ಕ"),
    (51, 65, "ಐವತ್ತು ಅಥವಾ ಅರವತ್ತು ವಯಸ್ಸಿನ ವಯಸ್ಕ"),
    (66, 120, "ಹಿರಿಯ ವ್ಯಕ್ತಿ"),
]

_AGE_BANDS_HI = [
    (0, 12, "एक बच्चा"),
    (13, 17, "एक किशोर"),
    (18, 25, "एक युवा वयस्क"),
    (26, 35, "बीस या तीस वर्ष की आयु का वयस्क"),
    (36, 50, "एक मध्यम आयु का वयस्क"),
    (51, 65, "पचास या साठ वर्ष की आयु का वयस्क"),
    (66, 120, "एक वरिष्ठ व्यक्ति"),
]


def _age_label(age: int, bands: list[tuple[int, int, str]]) -> str:
    for lo, hi, label in bands:
        if lo <= age <= hi:
            return label

    return bands[-1][2]


def generalize_age(
    value: str,
    language: str = "en",
) -> dict:
    """
    Convert exact age into an age band.

    language:
        en
        kn
        hi
    """

    try:
        age = int(str(value).strip())
    except (ValueError, TypeError):
        return {
            "new_value": value,
            "info_loss_cost": 0.0,
        }

    if language == "kn":
        label = _age_label(age, _AGE_BANDS_KN)
    elif language == "hi":
        label = _age_label(age, _AGE_BANDS_HI)
    else:
        label = _age_label(age, _AGE_BANDS_EN)

    return {
        "new_value": label,
        "info_loss_cost": 0.25,
    }


# ============================================================
# LOCATION
# ============================================================

_LOCATION_COSTS = {
    "village": 0.35,
    "block": 0.25,
    "district": 0.15,
    "state": 0.10,
    "region": 0.05,
}

_LOCATION_LABELS_EN = {
    "village": "a rural area",
    "block": "a local area",
    "district": "a district",
    "state": "a state",
    "region": "a region",
}

_LOCATION_LABELS_KN = {
    "village": "ಗ್ರಾಮೀಣ ಪ್ರದೇಶ",
    "block": "ಸ್ಥಳೀಯ ಪ್ರದೇಶ",
    "district": "ಜಿಲ್ಲೆ",
    "state": "ರಾಜ್ಯ",
    "region": "ಪ್ರದೇಶ",
}

_LOCATION_LABELS_HI = {
    "village": "ग्रामीण क्षेत्र",
    "block": "स्थानीय क्षेत्र",
    "district": "एक जिला",
    "state": "एक राज्य",
    "region": "एक क्षेत्र",
}


def generalize_location(
    value: str,
    granularity: str = "village",
    language: str = "en",
) -> dict:

    granularity = str(granularity).lower()

    if granularity not in _LOCATION_COSTS:
        granularity = "village"

    if language == "kn":
        label = _LOCATION_LABELS_KN[granularity]
    elif language == "hi":
        label = _LOCATION_LABELS_HI[granularity]
    else:
        label = _LOCATION_LABELS_EN[granularity]

    return {
        "new_value": label,
        "info_loss_cost": _LOCATION_COSTS[granularity],
    }


# ============================================================
# FACILITY
# ============================================================

_FACILITY_MAP_EN = {
    "phc": "a healthcare facility",
    "hospital": "a healthcare facility",
    "clinic": "a healthcare facility",
    "dispensary": "a healthcare facility",
    "anganwadi": "a community centre",
    "school": "an educational institution",
    "bank": "a financial institution",
    "post office": "a government office",
    "panchayat": "a local government office",
}

_FACILITY_MAP_KN = {
    "ಆಸ್ಪತ್ರೆ": "ಆರೋಗ್ಯ ಕೇಂದ್ರ",
    "ಚಿಕಿತ್ಸಾಲಯ": "ಆರೋಗ್ಯ ಕೇಂದ್ರ",
    "ಆರೋಗ್ಯ ಕೇಂದ್ರ": "ಆರೋಗ್ಯ ಕೇಂದ್ರ",
    "ಶಾಲೆ": "ಶೈಕ್ಷಣಿಕ ಸಂಸ್ಥೆ",
    "ಬ್ಯಾಂಕ್": "ಹಣಕಾಸು ಸಂಸ್ಥೆ",
    "ಅಂಚೆ ಕಚೇರಿ": "ಸರ್ಕಾರಿ ಕಚೇರಿ",
}

_FACILITY_MAP_HI = {
    "अस्पताल": "स्वास्थ्य केंद्र",
    "क्लिनिक": "स्वास्थ्य केंद्र",
    "स्वास्थ्य केंद्र": "स्वास्थ्य केंद्र",
    "स्कूल": "शैक्षणिक संस्थान",
    "बैंक": "वित्तीय संस्थान",
}


def generalize_facility(
    value: str,
    language: str = "en",
) -> dict:

    if not value:
        return {
            "new_value": value,
            "info_loss_cost": 0.0,
        }

    if language == "kn":
        for keyword, replacement in _FACILITY_MAP_KN.items():
            if keyword in value:
                return {
                    "new_value": replacement,
                    "info_loss_cost": 0.20,
                }

        return {
            "new_value": "ಆರೋಗ್ಯ ಕೇಂದ್ರ",
            "info_loss_cost": 0.20,
        }

    if language == "hi":
        for keyword, replacement in _FACILITY_MAP_HI.items():
            if keyword in value:
                return {
                    "new_value": replacement,
                    "info_loss_cost": 0.20,
                }

        return {
            "new_value": "स्वास्थ्य केंद्र",
            "info_loss_cost": 0.20,
        }

    normalized = value.casefold()

    for keyword, replacement in _FACILITY_MAP_EN.items():
        if keyword in normalized:
            return {
                "new_value": replacement,
                "info_loss_cost": 0.20,
            }

    return {
        "new_value": "a facility",
        "info_loss_cost": 0.20,
    }


# ============================================================
# DATE / TIME
# ============================================================

_DATE_GENERALIZATIONS = [
    (
        ["today", "tonight"],
        "recently",
        0.15,
    ),
    (
        ["yesterday"],
        "recently",
        0.15,
    ),
    (
        ["this morning", "this afternoon"],
        "recently",
        0.15,
    ),
    (
        ["last week", "past week"],
        "in the past week",
        0.20,
    ),
    (
        ["last month", "past month"],
        "recently",
        0.25,
    ),
    (
        ["this year"],
        "this year",
        0.10,
    ),
    (
        [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ],
        "recently",
        0.20,
    ),
    (
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ],
        "recently",
        0.25,
    ),
]


def generalize_date(
    value: str,
    language: str = "en",
) -> dict:

    normalized = str(value).casefold()

    for keywords, label, cost in _DATE_GENERALIZATIONS:
        if any(keyword in normalized for keyword in keywords):

            if language == "kn" and label == "recently":
                label = "ಇತ್ತೀಚೆಗೆ"

            elif language == "hi" and label == "recently":
                label = "हाल ही में"

            return {
                "new_value": label,
                "info_loss_cost": cost,
            }

    if language == "kn":
        label = "ಇತ್ತೀಚೆಗೆ"
    elif language == "hi":
        label = "हाल ही में"
    else:
        label = "recently"

    return {
        "new_value": label,
        "info_loss_cost": 0.25,
    }


# ============================================================
# HEALTH
# ============================================================

_HEALTH_KEYWORDS_EN = {
    "neurological": "a neurological health condition",
    "autoimmune": "an autoimmune condition",
    "diabetes": "a metabolic health condition",
    "hypertension": "a cardiovascular health condition",
    "high blood pressure": "a cardiovascular health condition",
}

_HEALTH_KEYWORDS_KN = {
    "ನರವೈಜ್ಞಾನಿಕ": "ನರವೈಜ್ಞಾನಿಕ ಆರೋಗ್ಯ ಸ್ಥಿತಿ",
    "ಸ್ವಯಂಪ್ರತಿರೋಧಕ": "ಸ್ವಯಂಪ್ರತಿರೋಧಕ ಆರೋಗ್ಯ ಸ್ಥಿತಿ",
    "ಮಧುಮೇಹ": "ಚಯಾಪಚಯ ಆರೋಗ್ಯ ಸ್ಥಿತಿ",
    "ರಕ್ತದೊತ್ತಡ": "ಹೃದಯರಕ್ತನಾಳದ ಆರೋಗ್ಯ ಸ್ಥಿತಿ",
    "ಅಪರೂಪದ": "ಆರೋಗ್ಯ ಸ್ಥಿತಿ",
}

_HEALTH_KEYWORDS_HI = {
    "न्यूरोलॉजिकल": "न्यूरोलॉजिकल स्वास्थ्य स्थिति",
    "स्वप्रतिरक्षी": "स्वप्रतिरक्षी स्वास्थ्य स्थिति",
    "ऑटोइम्यून": "स्वप्रतिरक्षी स्वास्थ्य स्थिति",
    "मधुमेह": "चयापचय संबंधी स्वास्थ्य स्थिति",
    "रक्तचाप": "हृदय संबंधी स्वास्थ्य स्थिति",
}


def generalize_health(
    value: str,
    language: str = "en",
) -> dict:

    if not value:
        return {
            "new_value": value,
            "info_loss_cost": 0.0,
        }

    if language == "kn":
        for keyword, replacement in _HEALTH_KEYWORDS_KN.items():
            if keyword in value:
                return {
                    "new_value": replacement,
                    "info_loss_cost": 0.60,
                }

        return {
            "new_value": "ಆರೋಗ್ಯ ಸ್ಥಿತಿ",
            "info_loss_cost": 0.60,
        }

    if language == "hi":
        for keyword, replacement in _HEALTH_KEYWORDS_HI.items():
            if keyword in value:
                return {
                    "new_value": replacement,
                    "info_loss_cost": 0.60,
                }

        return {
            "new_value": "एक स्वास्थ्य स्थिति",
            "info_loss_cost": 0.60,
        }

    normalized = value.casefold()

    for keyword, replacement in _HEALTH_KEYWORDS_EN.items():
        if keyword in normalized:
            return {
                "new_value": replacement,
                "info_loss_cost": 0.60,
            }

    return {
        "new_value": "a health condition",
        "info_loss_cost": 0.60,
    }


# ============================================================
# GENERIC CONTEXTUAL ATTRIBUTES
# ============================================================

def generalize_contextual_attribute(
    attr_type: str,
    value: str,
    language: str = "en",
) -> dict:

    attr_type = attr_type.upper()

    if language == "kn":
        labels = {
            "GENDER": "ಲಿಂಗ ಮಾಹಿತಿ ಹೊಂದಿರುವ ವ್ಯಕ್ತಿ",
            "OCCUPATION": "ವೃತ್ತಿಪರ ಹಿನ್ನೆಲೆ ಹೊಂದಿರುವ ವ್ಯಕ್ತಿ",
            "RELIGION": "ಧಾರ್ಮಿಕ ಹಿನ್ನೆಲೆ ಹೊಂದಿರುವ ವ್ಯಕ್ತಿ",
            "CASTE": "ಸಾಮಾಜಿಕ ಹಿನ್ನೆಲೆ ಹೊಂದಿರುವ ವ್ಯಕ್ತಿ",
        }
    elif language == "hi":
        labels = {
            "GENDER": "लिंग संबंधी जानकारी वाला व्यक्ति",
            "OCCUPATION": "पेशेवर पृष्ठभूमि वाला व्यक्ति",
            "RELIGION": "धार्मिक पृष्ठभूमि वाला व्यक्ति",
            "CASTE": "सामाजिक पृष्ठभूमि वाला व्यक्ति",
        }
    else:
        labels = {
            "GENDER": "a person with gender information",
            "OCCUPATION": "a person with a professional background",
            "RELIGION": "a person with a religious background",
            "CASTE": "a person with a social background",
        }

    return {
        "new_value": labels.get(
            attr_type,
            "a person with contextual information",
        ),
        "info_loss_cost": 0.60,
    }


# ============================================================
# UNIQUENESS
# ============================================================

def generalize_uniqueness_word(value: str) -> dict:
    return {
        "new_value": "",
        "info_loss_cost": 0.05,
    }


# ============================================================
# DIRECT PII
# ============================================================

_PII_PLACEHOLDERS = {
    "NAME": "[PERSON]",
    "PERSON": "[PERSON]",
    "PHONE": "[PHONE]",
    "PHONE_NUMBER": "[PHONE]",
    "EMAIL": "[EMAIL]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "VEHICLE": "[VEHICLE_ID]",
    "VEHICLE_ID": "[VEHICLE_ID]",
    "VEHICLE_NUMBER": "[VEHICLE_ID]",
    "REGISTRATION_ID": "[REGISTRATION_ID]",
    "AADHAAR": "[AADHAAR]",
    "PAN": "[PAN]",
    "GOVERNMENT_ID": "[GOVERNMENT_ID]",
    "ADDRESS": "[ADDRESS]",
    "PINCODE": "[PINCODE]",
    "IFSC": "[IFSC]",
    "BANK_ACCOUNT": "[BANK_ACCOUNT]",
    "VOTER_ID": "[VOTER_ID]",
}

_PII_TYPES = set(_PII_PLACEHOLDERS)


def redact_pii(attr_type: str) -> dict:
    return {
        "new_value": _PII_PLACEHOLDERS.get(
            attr_type.upper(),
            "[REDACTED]",
        ),
        "info_loss_cost": 1.0,
    }


# ============================================================
# PUBLIC DISPATCH
# ============================================================

def get_transformation(
    attribute: dict,
) -> dict | None:

    attr_type = str(
        attribute.get("type", "")
    ).upper()

    attr_value = str(
        attribute.get("value", "")
    )

    specificity = float(
        attribute.get("specificity", 0.0)
    )

    language = str(
        attribute.get("language", "")
    ).lower()

    if language not in {"en", "kn", "hi"}:
        if _is_kannada(attr_value):
            language = "kn"
        elif _is_hindi(attr_value):
            language = "hi"
        else:
            language = "en"

    # --------------------------------------------------------
    # Direct PII
    # --------------------------------------------------------

    if attr_type in _PII_TYPES:
        result = redact_pii(attr_type)

    # --------------------------------------------------------
    # Age
    # --------------------------------------------------------

    elif attr_type == "AGE":
        result = generalize_age(
            attr_value,
            language,
        )

    # --------------------------------------------------------
    # Location
    # --------------------------------------------------------

    elif attr_type == "LOCATION":
        result = generalize_location(
            attr_value,
            attribute.get("granularity", "village"),
            language,
        )

    # --------------------------------------------------------
    # Facility
    # --------------------------------------------------------

    elif attr_type == "FACILITY":
        result = generalize_facility(
            attr_value,
            language,
        )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    elif attr_type == "DATE":
        result = generalize_date(
            attr_value,
            language,
        )

    # --------------------------------------------------------
    # Uniqueness
    # --------------------------------------------------------

    elif attr_type == "UNIQUENESS_WORD":
        result = generalize_uniqueness_word(
            attr_value,
        )

    # --------------------------------------------------------
    # Contextual sensitive attributes
    # --------------------------------------------------------

    elif attr_type in {
        "HEALTH",
        "GENDER",
        "OCCUPATION",
        "RELIGION",
        "CASTE",
    }:

        if specificity < 0.5:
            return None

        if attr_type == "HEALTH":
            result = generalize_health(
                attr_value,
                language,
            )
        else:
            result = generalize_contextual_attribute(
                attr_type,
                attr_value,
                language,
            )

    else:
        return None

    return {
        "attribute": attr_type,
        "original": attr_value,
        "new_value": result["new_value"],
        "info_loss_cost": result["info_loss_cost"],
    }