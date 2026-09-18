"""
test_privacy_module.py
----------------------
Unit tests for Module 3 — Privacy Transformation.
Uses only Python's built-in unittest — no pytest or other packages needed.

Run with:
    python tests/test_privacy_module.py
        or
    python -m unittest tests.test_privacy_module -v
"""

from __future__ import annotations
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from backend.modules.privacy.generalizer import (
    generalize_age,
    generalize_location,
    generalize_facility,
    generalize_date,
    generalize_health,
    generalize_uniqueness_word,
    get_transformation,
)
from backend.modules.privacy.utility import (
    information_loss,
    information_retained,
    utility_score,
)
from backend.modules.privacy.optimizer import (
    DEFAULT_RISK_THRESHOLD,
    _normalize_contextual_indicators,
    optimize,
)
from backend.modules.risk.features import build_features
from backend.modules.risk.scorer import score_risk

SAMPLE_DIR = ROOT / "data" / "sample"


def _load_mock(name: str) -> dict:
    path = SAMPLE_DIR / f"mock_{name}_risk.json"
    return json.loads(path.read_text())


def _m2_expected_risk(data: dict) -> float:
    """Return the optimizer's authoritative, normalized M2 score for a fixture."""
    return score_risk(build_features({
        "attributes": data["attributes"],
        "contextual_indicators": _normalize_contextual_indicators(
            data.get("contextual_indicators", [])
        ),
    }))["risk_score"] / 10.0


# ===========================================================================
# 1. Generalizer tests
# ===========================================================================

class TestGeneralizeAge(unittest.TestCase):
    def test_young_adult(self):
        result = generalize_age("23")
        self.assertEqual(result["new_value"], "a young adult")
        self.assertGreater(result["info_loss_cost"], 0)

    def test_senior(self):
        result = generalize_age("72")
        self.assertEqual(result["new_value"], "a senior")

    def test_child(self):
        result = generalize_age("8")
        self.assertEqual(result["new_value"], "a child")

    def test_non_numeric_unchanged(self):
        result = generalize_age("young adult")
        self.assertEqual(result["info_loss_cost"], 0.0)


class TestGeneralizeLocation(unittest.TestCase):
    def test_village_returns_area_label(self):
        result = generalize_location("Village X", granularity="village")
        label = result["new_value"].lower()
        self.assertTrue("rural" in label or "area" in label)
        self.assertGreater(result["info_loss_cost"], 0)

    def test_district_cheaper_than_village(self):
        village  = generalize_location("Village X", granularity="village")["info_loss_cost"]
        district = generalize_location("District Y", granularity="district")["info_loss_cost"]
        self.assertGreater(village, district)


class TestGeneralizeFacility(unittest.TestCase):
    def test_phc(self):
        result = generalize_facility("PHC Y")
        self.assertIn("healthcare", result["new_value"].lower())

    def test_hospital(self):
        result = generalize_facility("City Hospital")
        self.assertIn("healthcare", result["new_value"].lower())

    def test_unknown_facility(self):
        result = generalize_facility("Unknown Place Z")
        self.assertEqual(result["new_value"], "a facility")


class TestGeneralizeDate(unittest.TestCase):
    def test_yesterday(self):
        self.assertEqual(generalize_date("yesterday")["new_value"], "recently")

    def test_today(self):
        self.assertEqual(generalize_date("today")["new_value"], "recently")

    def test_month_name(self):
        self.assertEqual(generalize_date("September")["new_value"], "recently")


class TestGeneralizeHealth(unittest.TestCase):
    def test_neurological_disorder_becomes_neurological_condition(self):
        result = generalize_health("rare neurological disorder")
        self.assertEqual(result["new_value"], "neurological condition")

    def test_autoimmune_condition_remains_category_specific(self):
        result = generalize_health("rare autoimmune condition")
        self.assertEqual(result["new_value"], "autoimmune condition")

    def test_unknown_condition_becomes_broad_health_condition(self):
        result = generalize_health("rare unidentified medical disorder")
        self.assertEqual(result["new_value"], "health condition")


class TestGeneralizeUniquenessWord(unittest.TestCase):
    def test_only_removed(self):
        result = generalize_uniqueness_word("only")
        self.assertEqual(result["new_value"], "")
        self.assertLess(result["info_loss_cost"], 0.1)


class TestGetTransformation(unittest.TestCase):
    def test_age_attribute(self):
        attr = {"type": "AGE", "value": "23", "specificity": 1.0}
        t = get_transformation(attr)
        self.assertIsNotNone(t)
        self.assertEqual(t["attribute"], "AGE")
        self.assertEqual(t["original"], "23")

    def test_health_low_specificity_kept(self):
        """Health with specificity < 0.9 should NOT be transformed."""
        attr = {"type": "HEALTH", "value": "diabetes", "specificity": 0.5}
        t = get_transformation(attr)
        self.assertIsNone(t)

    def test_uniqueness_word_removed(self):
        attr = {"type": "UNIQUENESS_WORD", "value": "only", "specificity": 1.0}
        t = get_transformation(attr)
        self.assertIsNotNone(t)
        self.assertEqual(t["new_value"], "")


# ===========================================================================
# 2. Utility tests
# ===========================================================================

class TestInformationLoss(unittest.TestCase):
    def test_no_transformations(self):
        self.assertEqual(information_loss([]), 0.0)

    def test_cheap_transformation(self):
        loss = information_loss([{"attribute": "DATE", "info_loss_cost": 0.15}])
        self.assertGreater(loss, 0.0)
        self.assertLessEqual(loss, 0.2)

    def test_health_loss_greater_than_date_loss(self):
        health = information_loss([{"attribute": "HEALTH", "info_loss_cost": 0.6}])
        date   = information_loss([{"attribute": "DATE",   "info_loss_cost": 0.15}])
        self.assertGreater(health, date)

    def test_retained_is_complement(self):
        transforms = [{"attribute": "AGE", "info_loss_cost": 0.25}]
        loss     = information_loss(transforms)
        retained = information_retained(transforms)
        self.assertAlmostEqual(loss + retained, 1.0, places=6)


class TestUtilityScore(unittest.TestCase):
    def test_structure(self):
        attrs = [
            {"type": "AGE",    "value": "23",       "specificity": 1.0},
            {"type": "HEALTH", "value": "diabetes", "specificity": 0.5},
        ]
        transforms = [
            {"attribute": "AGE", "original": "23", "new_value": "a young adult", "info_loss_cost": 0.25}
        ]
        result = utility_score(attrs, transforms)
        self.assertEqual(result["attributes_total"],   2)
        self.assertEqual(result["attributes_changed"], 1)
        self.assertEqual(result["attributes_kept"],    1)
        self.assertGreaterEqual(result["information_loss"],     0.0)
        self.assertLessEqual(   result["information_loss"],     1.0)


# ===========================================================================
# 3. Optimizer end-to-end tests
# ===========================================================================

class TestOptimizerLowRisk(unittest.TestCase):
    def setUp(self):
        self.data   = _load_mock("low")
        self.result = optimize(self.data)

    def test_risk_below_threshold(self):
        self.assertLessEqual(self.result["final_risk_score"], DEFAULT_RISK_THRESHOLD)

    def test_no_transformations_applied(self):
        self.assertEqual(len(self.result["transformations"]), 0)

    def test_text_unchanged(self):
        self.assertEqual(self.result["sanitized_text"], self.data["original_text"])

    def test_zero_information_loss(self):
        self.assertEqual(self.result["utility"]["information_loss"], 0.0)


class TestOptimizerMediumRisk(unittest.TestCase):
    def setUp(self):
        self.data   = _load_mock("medium")
        self.result = optimize(self.data)

    def test_risk_is_reduced(self):
        self.assertLess(self.result["final_risk_score"], self.result["initial_risk_score"])
        self.assertGreater(self.result["risk_reduced"], 0)

    def test_transformations_applied(self):
        self.assertGreater(len(self.result["transformations"]), 0)

    def test_sanitized_text_differs(self):
        self.assertNotEqual(self.result["sanitized_text"], self.data["original_text"])

    def test_some_utility_retained(self):
        self.assertGreater(self.result["utility"]["information_retained"], 0.4)

    def test_age_generalization_removes_years_old_suffix(self):
        data = {
            "conversation_id": "conv_age_001",
            "original_text": "I am 23 years old and live in Village X.",
            "attributes": [
                {
                    "type": "AGE",
                    "value": "23",
                    "confidence": 0.97,
                    "specificity": 1.0,
                }
            ],
            "risk_score": 0.91,
            "risk_level": "HIGH",
        }

        result = optimize(data, scorer_fn=lambda attrs: 0.8 if attrs[0]["specificity"] > 0.5 else 0.0)

        self.assertEqual(
            result["sanitized_text"],
            "I am a young adult and live in Village X.",
        )


class TestOptimizerHighRisk(unittest.TestCase):
    def setUp(self):
        self.data   = _load_mock("high")
        self.result = optimize(self.data)

    def test_risk_is_reduced(self):
        self.assertLess(self.result["final_risk_score"], self.result["initial_risk_score"])
        self.assertGreater(self.result["risk_reduced"], 0)

    def test_uniqueness_word_removed(self):
        self.assertNotIn("only", self.result["sanitized_text"].lower())

    def test_age_generalized(self):
        self.assertNotIn("23", self.result["sanitized_text"])

    def test_location_generalized(self):
        self.assertNotIn("Village X", self.result["sanitized_text"])

    def test_facility_generalized(self):
        self.assertNotIn("PHC Y", self.result["sanitized_text"])

    def test_date_generalized(self):
        self.assertNotIn("yesterday", self.result["sanitized_text"].lower())

    def test_result_has_required_keys(self):
        required = {
            "conversation_id", "original_text", "sanitized_text",
            "initial_risk_score", "final_risk_score", "risk_reduced",
            "transformations", "utility", "threshold_used",
        }
        self.assertTrue(required.issubset(self.result.keys()))

    def test_health_attribute_kept(self):
        """Diabetes (low specificity) must not be changed."""
        kept_types = {
            d["attribute"] for d in self.result["utility"]["attribute_detail"]
            if not d["changed"]
        }
        self.assertIn("HEALTH", kept_types)

    def test_stricter_threshold_same_or_lower_risk(self):
        result_normal = optimize(self.data, threshold=0.40)
        result_strict = optimize(self.data, threshold=0.20)
        self.assertLessEqual(
            result_strict["final_risk_score"],
            result_normal["final_risk_score"],
        )

    def test_risk_reduced_non_negative(self):
        self.assertGreaterEqual(self.result["risk_reduced"], 0)


# ===========================================================================
# 4. External scorer integration
# ===========================================================================

class TestExternalScorer(unittest.TestCase):
    def test_custom_scorer_fn_is_used(self):
        """
        If Module 2 passes scorer_fn that always returns 0.0 (safe),
        no transformations should be applied even on the high-risk case.
        """
        data        = _load_mock("high")
        always_safe = lambda attrs: 0.0
        result      = optimize(data, scorer_fn=always_safe)
        self.assertEqual(len(result["transformations"]), 0)
        self.assertEqual(result["final_risk_score"], 0.0)


class TestModule2Integration(unittest.TestCase):
    def test_module2_output_flows_directly_and_preserves_metadata(self):
        data = _load_mock("high")

        result = optimize(data)

        self.assertEqual(result["conversation_id"], data["conversation_id"])
        self.assertEqual(result["original_text"], data["original_text"])
        self.assertAlmostEqual(
            result["initial_risk_score"],
            _m2_expected_risk(data),
            places=4,
        )
        self.assertEqual(result["initial_risk_level"], data["risk_level"])
        self.assertEqual(
            result["contextual_indicators"],
            data["contextual_indicators"],
        )
        self.assertEqual(result["risk_level"], data["risk_level"])

    def test_m3_uses_real_m2_scorer_not_placeholder(self):
        data = _load_mock("medium")
        expected = _m2_expected_risk(data)

        result = optimize(data)

        self.assertAlmostEqual(result["initial_risk_score"], expected, places=4)
        self.assertLess(result["final_risk_score"], result["initial_risk_score"])
        self.assertGreater(result["risk_reduced"], 0.0)

    def test_transformed_attribute_is_rescored_through_m2(self):
        data = _load_mock("high")
        result = optimize(data)

        self.assertGreater(result["initial_risk_score"], result["final_risk_score"])
        self.assertGreater(result["risk_reduced"], 0.0)
        self.assertGreater(len(result["transformations"]), 0)

    def test_threshold_mapping_is_correct(self):
        self.assertAlmostEqual(DEFAULT_RISK_THRESHOLD * 10.0, 4.0, places=6)

    def test_low_risk_input_remains_unchanged(self):
        data = _load_mock("low")
        result = optimize(data)

        self.assertEqual(result["sanitized_text"], data["original_text"])
        self.assertEqual(len(result["transformations"]), 0)
        self.assertAlmostEqual(result["initial_risk_score"], _m2_expected_risk(data), places=4)

    def test_m2_derived_risk_level_overrides_stale_input_metadata(self):
        data = _load_mock("high")
        data["risk_level"] = "LOW"

        result = optimize(data)

        self.assertEqual(result["initial_risk_level"], "HIGH")
        self.assertEqual(result["risk_level"], "HIGH")
        self.assertEqual(result["final_risk_level"], "MEDIUM")

    def test_health_generalization_does_not_duplicate_source_article(self):
        data = {
            "conversation_id": "conv_health_article_001",
            "original_text": "I sought treatment for a rare autoimmune condition.",
            "attributes": [{
                "type": "HEALTH",
                "value": "rare autoimmune condition",
                "confidence": 0.91,
                "specificity": 0.9,
            }],
        }

        result = optimize(data)

        self.assertEqual(
            result["sanitized_text"],
            "I sought treatment for an autoimmune condition.",
        )

    def test_health_generalization_preserves_article_and_grammar(self):
        data = {
            "conversation_id": "conv_health_grammar_001",
            "original_text": "I sought treatment for a rare neurological disorder.",
            "attributes": [{
                "type": "HEALTH",
                "value": "rare neurological disorder",
                "confidence": 0.91,
                "specificity": 0.9,
            }],
        }

        result = optimize(data)

        self.assertEqual(
            result["sanitized_text"],
            "I sought treatment for a neurological condition.",
        )
        self.assertNotIn("a a ", result["sanitized_text"].lower())
        self.assertNotIn("an an ", result["sanitized_text"].lower())

    def test_health_transform_is_rescored_through_m2(self):
        data = {
            "conversation_id": "conv_health_m2_001",
            "original_text": "I sought treatment for a rare neurological disorder.",
            "attributes": [{
                "type": "HEALTH",
                "value": "rare neurological disorder",
                "confidence": 0.91,
                "specificity": 0.9,
            }],
            "contextual_indicators": [],
        }
        expected_attributes = [{
            **data["attributes"][0],
            "value": "neurological condition",
            "specificity": 0.4,
        }]
        expected = score_risk(build_features({
            "attributes": expected_attributes,
            "contextual_indicators": [],
        }))["risk_score"] / 10.0

        result = optimize(data)

        self.assertAlmostEqual(result["final_risk_score"], expected, places=4)

    def test_transforms_can_reduce_normalized_m2_risk_below_threshold(self):
        data = {
            "conversation_id": "conv_threshold_001",
            "original_text": "I am 23 years old.",
            "attributes": [{
                "type": "AGE", "value": "23", "confidence": 0.97,
                "specificity": 1.0,
            }],
            "contextual_indicators": ["exact_age"],
        }
        result = optimize(data)

        self.assertLessEqual(result["final_risk_score"], DEFAULT_RISK_THRESHOLD)
        self.assertGreater(result["risk_reduced"], 0.0)
        self.assertGreater(len(result["transformations"]), 0)

    def test_contextual_indicator_strings_are_normalized_for_m2(self):
        data = _load_mock("medium")
        result = optimize(data)

        self.assertAlmostEqual(result["initial_risk_score"], _m2_expected_risk(data), places=4)

    def test_contextual_indicators_passed_to_m2_have_required_keys(self):
        data = _load_mock("medium")
        data["contextual_indicators"] = [
            "exact_age",
            {"indicator": "village_level_location"},
        ]
        captured_indicators = []
        actual_build_features = build_features

        def capture_m2_input(conversation):
            captured_indicators.extend(conversation["contextual_indicators"])
            return actual_build_features(conversation)

        with patch(
            "backend.modules.privacy.optimizer.build_features",
            side_effect=capture_m2_input,
        ):
            optimize(data)

        self.assertTrue(captured_indicators)
        for indicator in captured_indicators:
            self.assertIn("indicator", indicator)
            self.assertIn("attribute_type", indicator)

    def test_existing_contextual_indicator_dicts_are_accepted(self):
        data = _load_mock("medium")
        data["contextual_indicators"] = [
            {"indicator": "exact_age", "attribute_type": "AGE"},
            {"indicator": "village_level_location"},
        ]
        result = optimize(data)

        self.assertAlmostEqual(result["initial_risk_score"], _m2_expected_risk(data), places=4)

    def test_optional_module2_metadata_can_be_omitted(self):
        data = _load_mock("low")
        data.pop("risk_level")

        result = optimize(data)

        self.assertEqual(result["initial_risk_level"], "LOW")
        self.assertEqual(result["contextual_indicators"], [])


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
