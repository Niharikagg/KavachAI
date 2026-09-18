"""
pipeline.py — End-to-end KavachAI privacy pipeline orchestrator.

Flow
----
Citizen conversation
    → M1  analyze_conversation()        (NLP extraction)
    → M1→M2 adapter  adapt_m1_to_m2()  (data normalisation)
    → M2  build_features()              (feature engineering)
    → M2  score_risk()                  (risk scoring)
    → M2→M3 adapter  adapt_m2_to_m3()  (merge context + authoritative risk)
    → M3  optimize()                    (privacy generalisation)
    → final protected output

Design constraints
------------------
* M1, M2, and M3 modules are NOT modified here.
* This file owns only the wiring between modules.
* Each module is imported from its authoritative location.
"""

from __future__ import annotations

from typing import Any

from backend.modules.nlp.pipeline import analyze_conversation
from backend.modules.risk.m1_m2_adapter import adapt_m1_to_m2
from backend.modules.risk.features import build_features
from backend.modules.risk.scorer import score_risk
from backend.modules.risk.m2_m3_adapter import adapt_m2_to_m3
from backend.modules.privacy.optimizer import optimize


def process_conversation(
    conversation_id: str,
    messages: list[dict[str, str]],
    *,
    threshold: float = 0.40,
    scorer_fn=None,
) -> dict[str, Any]:
    """
    Run a citizen conversation through the full KavachAI privacy pipeline.

    Parameters
    ----------
    conversation_id : str
        Unique identifier for the conversation (e.g. UUID or sequential ID).
    messages : list[dict]
        List of message dicts, each with at least ``"role"`` and ``"content"``
        keys (e.g. ``[{"role": "user", "content": "..."}]``).
    threshold : float, optional
        M3 target normalised risk threshold (default 0.40, i.e. 40 % of max).
    scorer_fn : callable, optional
        Custom re-scoring function for M3 (default ``None`` → uses M2 scorer).

    Returns
    -------
    dict
        Full pipeline result containing:

        * ``conversation_id``  – echoed from input
        * ``m1``               – raw M1 extraction output
        * ``m2``               – M2 risk scoring output
                                 ``{features, risk_score, risk_level}``
        * ``m3``               – M3 optimisation output
                                 ``{sanitized_text, transformations,
                                   initial_risk_score, final_risk_score,
                                   risk_reduced, utility, threshold_used}``
    """
    # ------------------------------------------------------------------
    # Stage 1 — M1: NLP extraction
    # ------------------------------------------------------------------
    # M1 pipeline expects {"id": int, "text": str}.  Callers may supply
    # OpenAI-style {"role": str, "content": str}; normalize transparently.
    def _normalize_msg(idx: int, msg: dict) -> dict:
        if "text" in msg:
            return msg                       # already in M1 format
        return {"id": msg.get("id", idx), "text": msg.get("content", "")}

    m1_messages = [_normalize_msg(i, m) for i, m in enumerate(messages)]
    m1_output = analyze_conversation(conversation_id, m1_messages)

    # ------------------------------------------------------------------
    # Stage 2 — M1 → M2: Adapt and score risk
    # ------------------------------------------------------------------
    m2_input  = adapt_m1_to_m2(m1_output)
    features  = build_features(m2_input)
    m2_result = score_risk(features)
    m2_result["features"] = features          # carry features forward

    # ------------------------------------------------------------------
    # Stage 3 — M2 → M3: Merge context + authoritative risk, optimise
    # ------------------------------------------------------------------
    m3_input  = adapt_m2_to_m3(m2_input, m2_result)
    m3_result = optimize(m3_input, threshold=threshold, scorer_fn=scorer_fn)

    return {
        "conversation_id": conversation_id,
        "m1":              m1_output,
        "m2":              m2_result,
        "m3":              m3_result,
    }
