"""
schemas.py
----------
Plain dataclass-based request/response models — no external dependencies.
These define the contract between Module 1+2 (input) and Module 3 (output).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Input models (produced by Module 1 + 2)
# ---------------------------------------------------------------------------

@dataclass
class AttributeIn:
    type:        str
    value:       str
    confidence:  float          = 1.0
    specificity: float          = 1.0
    granularity: Optional[str]  = None   # For LOCATION: village|block|district|state|region


@dataclass
class AnalyzeRequest:
    conversation_id:       str
    original_text:         str
    attributes:            list[dict]   # list of AttributeIn-shaped dicts
    risk_score:            float
    risk_level:            str          # LOW | MEDIUM | HIGH
    contextual_indicators: list[str]    = field(default_factory=list)
    threshold:             Optional[float] = None  # override default 0.40


# ---------------------------------------------------------------------------
# Output models (produced by Module 3)
# ---------------------------------------------------------------------------

@dataclass
class TransformationOut:
    attribute:      str
    original:       str
    new_value:      str
    info_loss_cost: float


@dataclass
class AttributeDetailOut:
    attribute:      str
    original_value: str
    new_value:      str
    changed:        bool
    info_loss_cost: float
    weight:         float


@dataclass
class UtilityOut:
    information_loss:     float
    information_retained: float
    attributes_total:     int
    attributes_changed:   int
    attributes_kept:      int
    attribute_detail:     list[dict]


@dataclass
class AnalyzeResponse:
    conversation_id:    str
    original_text:      str
    sanitized_text:     str
    initial_risk_score: float
    final_risk_score:   float
    initial_risk_level: str
    final_risk_level:   str
    risk_reduced:       float
    threshold_used:     float
    transformations:    list[dict]
    utility:            dict
