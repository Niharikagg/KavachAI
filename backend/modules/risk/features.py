import json
from typing import Any


HIGH_SPECIFICITY_THRESHOLD = 0.7
ATTRIBUTE_COUNT_CAP = 5
SENSITIVE_CATEGORIES = {"HEALTH", "FINANCIAL", "FAMILY", "CRIME", "LAW"}
UNIQUENESS_INDICATOR_WEIGHTS = {
    "uniqueness_word": 1.0,
    "village_level_location": 0.6,
    "exact_age": 0.3,
    "exact_date": 0.15,
    "sensitive_health_attribute": 0.2,
}


def calculate_uniqueness_risk(contextual_indicators: list[dict[str, Any]]) -> float:
    uniqueness_risk = sum(
        UNIQUENESS_INDICATOR_WEIGHTS.get(
            indicator.get("indicator"), 0.0
        )
        for indicator in contextual_indicators
    )
    return min(uniqueness_risk, 1.0)


def build_features(conversation: dict[str, Any]) -> dict[str, float]:
    attributes = conversation.get("attributes", [])
    specificity_values = [attribute["specificity"] for attribute in attributes]
    attribute_count = len(attributes)
    high_specificity_count = sum(
        specificity >= HIGH_SPECIFICITY_THRESHOLD
        for specificity in specificity_values
    )
    sensitive_category_count = sum(
        attribute["type"] in SENSITIVE_CATEGORIES
        for attribute in attributes
    )

    return {
        "attribute_count_normalized": min(
            attribute_count / ATTRIBUTE_COUNT_CAP, 1.0
        ),
        "average_specificity": (
            sum(specificity_values) / attribute_count
            if attribute_count
            else 0.0
        ),
        "maximum_specificity": max(specificity_values, default=0.0),
        "high_specificity_ratio": (
            high_specificity_count / attribute_count
            if attribute_count
            else 0.0
        ),
        "sensitive_category_count_normalized": min(
            sensitive_category_count / len(SENSITIVE_CATEGORIES), 1.0
        ),
        "uniqueness_risk": calculate_uniqueness_risk(
            conversation.get("contextual_indicators", [])
        ),
    }


def print_features(conversation: dict[str, Any]) -> None:
    conversation_id = conversation["conversation_id"]
    print(f"Conversation: {conversation_id}")
    for attribute in conversation.get("attributes", []):
        print(
            f"  Message {attribute.get('source_message_id')}: "
            f"{attribute['type']} (specificity={attribute['specificity']})"
        )

    print("  Features:")
    for feature_name, feature_value in build_features(conversation).items():
        print(f"    {feature_name}: {feature_value:.2f}")
    print()


if __name__ == "__main__":
    with open("data/synthetic/module1_outputs.json", "r") as file:
        for conversation in json.load(file):
            print_features(conversation)