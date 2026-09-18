"""
test_m1_m2_m3_integration.py — End-to-end pipeline integration tests.

These tests exercise the COMPLETE flow:
    Citizen text → M1 extraction → M1→M2 adapter → M2 scoring
    → M2→M3 adapter → M3 optimisation → final protected output

No mocking.  Every module is called through its authoritative implementation.
"""

from __future__ import annotations

import pytest

from backend.modules.pipeline import process_conversation
from backend.modules.risk.m2_m3_adapter import adapt_m2_to_m3
from backend.modules.risk.m1_m2_adapter import adapt_m1_to_m2, analyze_with_m2
from backend.modules.privacy.optimizer import optimize


# ---------------------------------------------------------------------------
# Shared fixture: high-risk conversation used across multiple tests
# ---------------------------------------------------------------------------
HIGH_RISK_MESSAGES = [
    {
        "role": "user",
        "content": (
            "My name is Ravi Kumar and I am 68 years old. "
            "I live in Kotwali village near Kanpur district in Uttar Pradesh. "
            "I was diagnosed with Parkinson's disease last year and I visit "
            "SGPGI hospital every month. I work as a retired schoolteacher."
        ),
    }
]
HIGH_RISK_CONVERSATION_ID = "test_pipeline_001"


# ---------------------------------------------------------------------------
# test_pipeline_001: Full pipeline returns expected top-level keys
# ---------------------------------------------------------------------------
class TestPipelineContract:
    def test_pipeline_returns_all_top_level_keys(self):
        result = process_conversation(
            HIGH_RISK_CONVERSATION_ID,
            HIGH_RISK_MESSAGES,
        )
        assert "conversation_id" in result
        assert "m1" in result
        assert "m2" in result
        assert "m3" in result

    def test_conversation_id_echoed(self):
        result = process_conversation(
            HIGH_RISK_CONVERSATION_ID,
            HIGH_RISK_MESSAGES,
        )
        assert result["conversation_id"] == HIGH_RISK_CONVERSATION_ID

    def test_m2_risk_score_within_valid_range(self):
        result = process_conversation(
            HIGH_RISK_CONVERSATION_ID,
            HIGH_RISK_MESSAGES,
        )
        m2 = result["m2"]
        assert "risk_score" in m2
        assert "risk_level" in m2
        assert 0.0 <= m2["risk_score"] <= 10.0
        assert m2["risk_level"] in {"LOW", "MEDIUM", "HIGH"}

    def test_m3_output_has_required_keys(self):
        result = process_conversation(
            HIGH_RISK_CONVERSATION_ID,
            HIGH_RISK_MESSAGES,
        )
        m3 = result["m3"]
        for key in ("sanitized_text", "transformations",
                    "initial_risk_score", "final_risk_score",
                    "risk_reduced", "utility", "threshold_used"):
            assert key in m3, f"Missing key in M3 output: {key}"


# ---------------------------------------------------------------------------
# test_pipeline_002: M2 initial risk is REUSED by M3 (bypass active)
# ---------------------------------------------------------------------------
class TestM2ToM3Bypass:
    """Verify that M3 uses the pre-computed M2 risk instead of re-calling M2."""

    def test_m3_initial_risk_matches_m2_score(self):
        """
        M3 initial_risk_score (normalised 0-1 scale) must equal M2 risk_score/10.
        Any discrepancy means M3 is re-computing rather than reusing the
        authoritative M2 result.
        """
        result = process_conversation(
            "test_pipeline_002",
            HIGH_RISK_MESSAGES,
        )
        m2_risk      = result["m2"]["risk_score"]          # 0-10 scale
        m2_normalized = m2_risk / 10.0                     # → 0-1
        # M3 stores initial_risk_score in the same 0-1 normalised scale
        m3_initial   = result["m3"]["initial_risk_score"]  # 0-1 scale

        assert abs(m3_initial - m2_normalized) < 0.01, (
            f"M3 initial risk ({m3_initial:.4f}) does not match "
            f"M2 normalised risk ({m2_normalized:.4f}) [M2 raw={m2_risk:.4f}].  "
            f"Bypass may not be active."
        )

    def test_m3_reduces_risk_or_holds_when_already_low(self):
        result = process_conversation(
            "test_pipeline_002",
            HIGH_RISK_MESSAGES,
        )
        m3 = result["m3"]
        assert m3["final_risk_score"] <= m3["initial_risk_score"], (
            "M3 must not increase risk above M2 initial score."
        )


# ---------------------------------------------------------------------------
# test_pipeline_003: M2→M3 adapter contract
# ---------------------------------------------------------------------------
class TestM2M3AdapterContract:
    """Unit tests for adapt_m2_to_m3() in isolation."""

    CONTEXT = {
        "conversation_id": "adapter_test",
        "original_text":   "I am 45 and live in Delhi.",
        "attributes": [
            {"name": "AGE", "value": "45", "specificity": 0.6},
            {"name": "LOCATION", "value": "Delhi", "specificity": 0.5},
        ],
        "contextual_indicators": ["public_space"],
    }

    M2_RESULT = {
        "risk_score":  5.5,
        "risk_level":  "MEDIUM",
        "features":    {"attribute_count_normalized": 0.4},
    }

    def test_required_m3_keys_present(self):
        out = adapt_m2_to_m3(self.CONTEXT, self.M2_RESULT)
        for key in ("conversation_id", "original_text", "attributes",
                    "contextual_indicators", "risk_score", "risk_level",
                    "initial_risk_score", "initial_risk_level"):
            assert key in out, f"Missing key: {key}"

    def test_initial_risk_score_equals_m2_risk_score(self):
        out = adapt_m2_to_m3(self.CONTEXT, self.M2_RESULT)
        assert out["initial_risk_score"] == self.M2_RESULT["risk_score"]
        assert out["initial_risk_level"] == self.M2_RESULT["risk_level"]

    def test_attributes_passed_verbatim(self):
        out = adapt_m2_to_m3(self.CONTEXT, self.M2_RESULT)
        assert out["attributes"] == self.CONTEXT["attributes"]

    def test_bad_input_raises_type_error(self):
        with pytest.raises(TypeError):
            adapt_m2_to_m3("not a dict", self.M2_RESULT)
        with pytest.raises(TypeError):
            adapt_m2_to_m3(self.CONTEXT, "not a dict")


# ---------------------------------------------------------------------------
# test_pipeline_004: Optimizer bypass unit test (no M1 needed)
# ---------------------------------------------------------------------------
class TestOptimizerBypass:
    """Directly test that optimizer uses initial_risk_score when supplied."""

    BASE_ATTRS = [
        {"type": "AGE",        "value": "68",           "specificity": 0.8},
        {"type": "LOCATION",   "value": "Kotwali",       "specificity": 0.9},
        {"type": "HEALTH",     "value": "Parkinson's",   "specificity": 0.85},
        {"type": "OCCUPATION", "value": "schoolteacher", "specificity": 0.5},
    ]

    def _make_input(self, with_bypass: bool, risk_score: float = 8.0):
        base = {
            "conversation_id":       "bypass_test",
            "original_text":         "I am 68, live in Kotwali...",
            "attributes":            self.BASE_ATTRS,
            "contextual_indicators": [],
        }
        if with_bypass:
            base["initial_risk_score"] = risk_score
            base["initial_risk_level"] = "HIGH"
        return base

    def test_bypass_initial_risk_matches_supplied_score(self):
        """When initial_risk_score supplied, M3 initial output must reflect it.

        M3 normalises risk to 0-1 internally and _build_result() stores that
        normalised value.  8.0 (0-10 scale) → 0.8 (0-1 scale).
        """
        inp    = self._make_input(with_bypass=True, risk_score=8.0)
        result = optimize(inp, threshold=0.40)
        # initial_risk_score in M3 output is on 0-1 (normalised) scale
        expected = 8.0 / 10.0   # 0.8
        assert abs(result["initial_risk_score"] - expected) < 0.01, (
            f"Expected initial_risk_score≈{expected} but got {result['initial_risk_score']}"
        )

    def test_without_bypass_m3_computes_own_initial_risk(self):
        """Without initial_risk_score, M3 must calculate risk independently."""
        inp    = self._make_input(with_bypass=False)
        result = optimize(inp, threshold=0.40)
        # Just verify it runs and returns a non-negative score
        assert result["initial_risk_score"] >= 0.0


# ---------------------------------------------------------------------------
# test_pipeline_005: Low-risk conversation passes through with no transforms
# ---------------------------------------------------------------------------
class TestLowRiskPassthrough:
    LOW_RISK_MESSAGES = [
        {"role": "user", "content": "I would like information about public parks."}
    ]

    def test_low_risk_few_or_no_transformations(self):
        result = process_conversation(
            "test_pipeline_005",
            self.LOW_RISK_MESSAGES,
        )
        m3 = result["m3"]
        # Low-risk conversations should either pass through unchanged or
        # require very few (≤2) transformations.
        assert len(m3.get("transformations", [])) <= 2, (
            f"Expected ≤2 transformations for low-risk input, "
            f"got {len(m3['transformations'])}: {m3['transformations']}"
        )
        assert m3["final_risk_score"] <= m3["initial_risk_score"]
