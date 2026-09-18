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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.modules.privacy.generalizer import (
    generalize_age,
    generalize_location,
    generalize_facility,
    generalize_date,
    generalize_uniqueness_word,
    get_transformation,
)
from backend.modules.privacy.utility import (
    information_loss,
    information_retained,
    utility_score,
)
from backend.modules.privacy.optimizer import optimize, DEFAULT_RISK_THRESHOLD

SAMPLE_DIR = ROOT / "data" / "sample"


def _load_mock(name: str) -> dict:
    path = SAMPLE_DIR / f"mock_{name}_risk.json"
    return json.loads(path.read_text())


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

    def test_risk_below_threshold(self):
        self.assertLessEqual(self.result["final_risk_score"], DEFAULT_RISK_THRESHOLD)

    def test_transformations_applied(self):
        self.assertGreater(len(self.result["transformations"]), 0)

    def test_sanitized_text_differs(self):
        self.assertNotEqual(self.result["sanitized_text"], self.data["original_text"])

    def test_some_utility_retained(self):
        self.assertGreater(self.result["utility"]["information_retained"], 0.4)


class TestOptimizerHighRisk(unittest.TestCase):
    def setUp(self):
        self.data   = _load_mock("high")
        self.result = optimize(self.data)

    def test_risk_below_threshold(self):
        self.assertLessEqual(self.result["final_risk_score"], DEFAULT_RISK_THRESHOLD)

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

        result = optimize(data, scorer_fn=lambda attrs: 0.0)

        self.assertEqual(result["conversation_id"], data["conversation_id"])
        self.assertEqual(result["original_text"], data["original_text"])
        self.assertEqual(result["initial_risk_score"], data["risk_score"])
        self.assertEqual(result["initial_risk_level"], data["risk_level"])
        self.assertEqual(
            result["contextual_indicators"],
            data["contextual_indicators"],
        )
        self.assertEqual(result["risk_level"], data["risk_level"])

    def test_optional_module2_metadata_can_be_omitted(self):
        data = _load_mock("low")
        data.pop("risk_level")

        result = optimize(data, scorer_fn=lambda attrs: 0.0)

        self.assertEqual(result["initial_risk_level"], "LOW")
        self.assertEqual(result["contextual_indicators"], [])


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
