import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.modules.risk.features import build_features
from backend.modules.risk.scorer import score_risk


SEED = 20260914
TARGET_COUNTS = {"LOW": 167, "MEDIUM": 166, "HIGH": 167}
OUTPUT_PATH = Path("data/synthetic/risk_ml_dataset.csv")
ATTRIBUTE_VALUES = {
    "AGE": ["adult", "young adult", "29", "34", "41", "57"],
    "GENDER": ["woman", "man", "nonbinary adult"],
    "LOCATION": [
        "India", "Karnataka", "Bengaluru", "Mysuru", "Village X",
        "Village Y", "small village near Pune", "Ward 14 in Jaipur",
    ],
    "HEALTH": [
        "health problems", "diabetes", "asthma", "rare neurological disorder",
        "chronic kidney disease", "a rare blood disorder",
    ],
    "FACILITY": ["health centre", "PHC Y", "City Hospital", "district clinic"],
    "DATE": ["this week", "yesterday", "14 March 2026", "2 January 2025"],
    "OCCUPATION": [
        "student", "teacher", "software engineer", "shopkeeper", "farm worker",
    ],
    "FINANCIAL": [
        "financial difficulty", "medical debt", "₹50 lakh debt", "a small loan",
    ],
    "FAMILY": ["parent", "single parent", "two children", "elderly dependent"],
    "PHONE_NUMBER": ["a phone ending in 4421", "a mobile number"],
    "GOVERNMENT_ID": ["a ration card", "a government identification number"],
}

ATTRIBUTE_TYPES = tuple(ATTRIBUTE_VALUES)
SENSITIVE_TYPES = {"HEALTH", "FINANCIAL", "FAMILY"}
LOW_FILLERS = [
    "I spent the morning reading and then made lunch at home.",
    "The bus was late, so I listened to music while waiting.",
    "After work I walked through the park and called a friend.",
    "I attended two classes, shared lunch with classmates, and came home.",
    "It was a quiet day with errands, tea, and a little television.",
]
HIGH_FILLERS = [
    "The appointment was recorded in the local clinic register.",
    "Only a few people in the area know these details about me.",
    "I had already discussed the matter with the district office.",
]
TEXT_PATTERNS = [
    "Here are some details: {details}.",
    "For context, {details}.",
    "Yesterday I mentioned that {details}.",
    "My situation is that {details}.",
    "A few details about me: {details}.",
]


def make_attributes(
    raw_attributes: list[tuple[str, str, float]], rng: random.Random
) -> list[dict[str, Any]]:
    attributes = []
    for index, raw_attribute in enumerate(raw_attributes, start=1):
        attribute_type, value, specificity = raw_attribute
        attribute = {
            "type": attribute_type,
            "value": value,
            "specificity": specificity,
        }
        attribute["confidence"] = round(rng.uniform(0.88, 0.99), 2)
        attribute["source_message_id"] = rng.randint(1, min(index, 4))
        attributes.append(attribute)
    return attributes


def choose_specificity(profile: str, rng: random.Random) -> float:
    if profile == "LOW":
        value = rng.uniform(0.05, 0.6)
    elif profile == "MEDIUM":
        value = rng.uniform(0.35, 0.85)
    else:
        value = rng.uniform(0.65, 1.0)
    return round(value, 2)


def generate_raw_attributes(
    profile: str, rng: random.Random
) -> list[tuple[str, str, float]]:
    if profile == "LOW":
        count = rng.choices([0, 1, 2, 3], weights=[0.2, 0.4, 0.3, 0.1])[0]
    elif profile == "MEDIUM":
        count = rng.choices([2, 3, 4, 5], weights=[0.2, 0.35, 0.3, 0.15])[0]
    else:
        count = rng.choices([2, 3, 4, 5, 6, 7, 8], weights=[0.08, 0.15, 0.2, 0.25, 0.18, 0.1, 0.04])[0]

    selected_types = rng.sample(ATTRIBUTE_TYPES, count)
    return [
        (attribute_type, rng.choice(ATTRIBUTE_VALUES[attribute_type]), choose_specificity(profile, rng))
        for attribute_type in selected_types
    ]


def build_indicators(
    raw_attributes: list[tuple[str, str, float]],
    uniqueness: bool,
) -> list[dict[str, str]]:
    indicators = []
    for attribute_type, value, specificity in raw_attributes:
        if attribute_type == "AGE" and specificity >= 0.85:
            indicators.append({"indicator": "exact_age", "attribute_type": "AGE"})
        if attribute_type == "LOCATION" and "village" in value.lower():
            indicators.append({
                "indicator": "village_level_location",
                "attribute_type": "LOCATION",
            })
        if attribute_type == "DATE" and specificity >= 0.85:
            indicators.append({"indicator": "exact_date", "attribute_type": "DATE"})
        if attribute_type == "HEALTH" and specificity >= 0.75:
            indicators.append({
                "indicator": "sensitive_health_attribute",
                "attribute_type": "HEALTH",
            })
    if uniqueness:
        uniqueness_type = next(
            (attribute_type for attribute_type, _, _ in raw_attributes if attribute_type in {"AGE", "LOCATION", "HEALTH"}),
            "AGE",
        )
        indicators.append({"indicator": "uniqueness_word", "attribute_type": uniqueness_type})
    return indicators


def attribute_phrase(attribute_type: str, value: str) -> str:
    phrases = {
        "AGE": f"I am {value}",
        "GENDER": f"I identify as {value}",
        "LOCATION": f"I live in {value}",
        "HEALTH": f"I have {value}",
        "FACILITY": f"I visited {value}",
        "DATE": f"the relevant date is {value}",
        "OCCUPATION": f"I work as a {value}",
        "FINANCIAL": f"I have {value}",
        "FAMILY": f"my family includes {value}",
        "PHONE_NUMBER": f"my contact is {value}",
        "GOVERNMENT_ID": f"I have {value}",
    }
    return phrases[attribute_type]


def build_text(
    raw_attributes: list[tuple[str, str, float]],
    uniqueness: bool,
    profile: str,
    rng: random.Random,
) -> str:
    if not raw_attributes:
        return rng.choice(["Hi", "Okay", "I like cricket.", *LOW_FILLERS])

    if profile == "HIGH" and len(raw_attributes) <= 2:
        values = {attribute_type: value for attribute_type, value, _ in raw_attributes}
        if "AGE" in values and "LOCATION" in values:
            if uniqueness:
                return f"Only {values['AGE']} in {values['LOCATION']}."
            return f"I am {values['AGE']} in {values['LOCATION']}."

    details = "; ".join(
        attribute_phrase(attribute_type, value)
        for attribute_type, value, _ in raw_attributes
    )
    if uniqueness:
        details = f"I am the only person matching these details: {details}"
    text = rng.choice(TEXT_PATTERNS).format(details=details)

    if profile == "LOW" and rng.random() < 0.45:
        text += " " + rng.choice(LOW_FILLERS)
    elif profile == "HIGH" and rng.random() < 0.35:
        text += " " + rng.choice(HIGH_FILLERS)
    return text


def build_candidate(conversation_number: int, rng: random.Random) -> dict[str, Any]:
    profile = rng.choices(["LOW", "MEDIUM", "HIGH"], weights=[0.38, 0.34, 0.28])[0]
    raw_attributes = generate_raw_attributes(profile, rng)
    uniqueness = profile == "HIGH" and rng.random() < 0.55
    attributes = make_attributes(raw_attributes, rng)
    indicators = build_indicators(raw_attributes, uniqueness)
    text = build_text(raw_attributes, uniqueness, profile, rng)

    return {
        "conversation_id": f"ml_conv_{conversation_number:04d}",
        "conversation_text": text,
        "attributes": attributes,
        "contextual_indicators": indicators,
    }


def dataset_row(conversation: dict[str, Any]) -> dict[str, Any]:
    features = build_features(conversation)
    result = score_risk(features)
    return {
        "conversation_id": conversation["conversation_id"],
        "conversation_text": conversation["conversation_text"],
        **features,
        "risk_score": round(result["risk_score"], 6),
        "risk_level": result["risk_level"],
        "attributes_json": json.dumps(conversation["attributes"], ensure_ascii=False),
        "contextual_indicators_json": json.dumps(
            conversation["contextual_indicators"], ensure_ascii=False
        ),
    }


def feature_key(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(
        row[field]
        for field in (
            "attribute_count_normalized",
            "average_specificity",
            "maximum_specificity",
            "high_specificity_ratio",
            "sensitive_category_count_normalized",
            "uniqueness_risk",
        )
    )


def generate_dataset() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    selected: dict[str, list[dict[str, Any]]] = {label: [] for label in TARGET_COUNTS}
    selected_feature_keys: set[tuple[float, ...]] = set()
    attempts = 0

    while any(len(selected[label]) < TARGET_COUNTS[label] for label in TARGET_COUNTS):
        attempts += 1
        candidate = build_candidate(attempts, rng)
        row = dataset_row(candidate)
        label = row["risk_level"]
        key = feature_key(row)
        if len(selected[label]) < TARGET_COUNTS[label] and key not in selected_feature_keys:
            selected[label].append(row)
            selected_feature_keys.add(key)

        if attempts > 1000000:
            raise RuntimeError("Could not generate the requested risk distribution")

    rows = [row for label in TARGET_COUNTS for row in selected[label]]
    rng.shuffle(rows)
    for index, row in enumerate(rows, start=1):
        row["conversation_id"] = f"ml_conv_{index:04d}"
    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "conversation_id",
        "conversation_text",
        "attribute_count_normalized",
        "average_specificity",
        "maximum_specificity",
        "high_specificity_ratio",
        "sensitive_category_count_normalized",
        "uniqueness_risk",
        "risk_score",
        "risk_level",
        "attributes_json",
        "contextual_indicators_json",
    ]
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]]) -> None:
    scores = [row["risk_score"] for row in rows]
    counts = {label: sum(row["risk_level"] == label for row in rows) for label in TARGET_COUNTS}
    feature_keys = [feature_key(row) for row in rows]
    unique_feature_combinations = len(set(feature_keys))
    duplicate_feature_combinations = sum(
        count > 1 for count in {key: feature_keys.count(key) for key in set(feature_keys)}.values()
    )
    print(f"Total examples: {len(rows)}")
    print()
    for label in ("LOW", "MEDIUM", "HIGH"):
        print(f"{label}:    {counts[label]}")
    print()
    print(f"Minimum score: {min(scores):.2f}")
    print(f"Maximum score: {max(scores):.2f}")
    print(f"Average score: {sum(scores) / len(scores):.2f}")
    print(f"Unique six-feature combinations: {unique_feature_combinations}")
    print(f"Duplicate six-feature combinations: {duplicate_feature_combinations}")
    print()
    print("Representative examples:")
    for description, row in [
        ("Very low-risk", min(rows, key=lambda item: item["risk_score"])),
        ("Low/medium boundary", min(
            (row for row in rows if row["risk_level"] == "MEDIUM"),
            key=lambda item: item["risk_score"],
        )),
        ("Medium-risk", min(
            (row for row in rows if row["risk_level"] == "MEDIUM"),
            key=lambda item: abs(item["risk_score"] - 5.0),
        )),
        ("High-risk", min(
            (row for row in rows if row["risk_level"] == "HIGH"),
            key=lambda item: item["risk_score"],
        )),
        ("Extremely high-risk", max(rows, key=lambda item: item["risk_score"])),
    ]:
        print(f"{description}: {row['risk_score']:.2f} {row['risk_level']}")
        print(f"  {row['conversation_text']}")


if __name__ == "__main__":
    generated_rows = generate_dataset()
    write_csv(generated_rows)
    print_summary(generated_rows)
