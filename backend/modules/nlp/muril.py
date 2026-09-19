"""MuRIL-backed language-aware processing context for Module 1.

MuRIL is a multilingual encoder, not a language-identification or NER model.
This module uses it once per conversation to tokenize and encode messages.
Unicode script observations supplement its processing metadata; they are not
language-classification results. The context is intentionally not exposed as
an M1 attribute because it is metadata about processing, not detected citizen
data.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer


MURIL_MODEL_NAME = "google/muril-base-cased"

_SCRIPT_RANGES = {
    "hi": r"\u0900-\u097f",
    "bn": r"\u0980-\u09ff",
    "gu": r"\u0a80-\u0aff",
    "pa": r"\u0a00-\u0a7f",
    "ta": r"\u0b80-\u0bff",
    "te": r"\u0c00-\u0c7f",
    "kn": r"\u0c80-\u0cff",
    "ml": r"\u0d00-\u0d7f",
    "or": r"\u0b00-\u0b7f",
    "ur": r"\u0600-\u06ff",
}
_SCRIPT_PATTERNS = {
    language: re.compile(f"[{unicode_range}]")
    for language, unicode_range in _SCRIPT_RANGES.items()
}


_HINDI_ROMAN_MARKERS = {
    "meri", "mera", "mere", "mujhe", "mujhko", "hai", "hain", "hoon", "hun",
    "mein", "liye", "jaana", "jana", "gaya", "gayi", "gaye", "rehta", "rehti",
    "rehte", "karna", "karta", "karti", "karte", "kiya", "nahi", "nahin",
    "bahut", "kaise", "kahan", "kyun", "kyu", "accha", "achha", "theek",
    "aapke", "aapka", "unka", "inhe", "unhe",
}
_KANNADA_ROMAN_MARKERS = {
    "nanna", "naanu", "nanage", "vayassu", "varsha", "varshada", "hatra", "halli",
    "halliyalli", "sanna", "aparoopada", "mahile", "aagi", "kelasa", "maaduttiddene",
    "nalli", "padediddene", "ide", "mattu",
}


def _script_context(text: str) -> dict[str, Any]:
    scripts = {language for language, pattern in _SCRIPT_PATTERNS.items() if pattern.search(text)}
    if re.search(r"[A-Za-z]", text):
        scripts.add("en")
    tokens = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    if tokens & _HINDI_ROMAN_MARKERS:
        scripts.add("hi")
    if tokens & _KANNADA_ROMAN_MARKERS:
        scripts.add("kn")

    return {
        "observed_scripts": sorted(scripts),
        "is_code_mixed": len(scripts) > 1,
    }


class MuRILProcessor:
    """Load MuRIL once and summarize multilingual processing context."""

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MURIL_MODEL_NAME)
        self.model = AutoModel.from_pretrained(MURIL_MODEL_NAME)
        self.model.eval()

    def process(self, texts: list[str]) -> list[dict[str, Any]]:
        contexts = [_script_context(text) for text in texts]
        if not texts:
            return contexts

        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        with torch.no_grad():
            output = self.model(**encoded)

        token_counts = encoded["attention_mask"].sum(dim=1).tolist()
        hidden_size = int(output.last_hidden_state.shape[-1])
        for context, token_count in zip(contexts, token_counts):
            context.update({
                "model": MURIL_MODEL_NAME,
                "token_count": int(token_count),
                "hidden_size": hidden_size,
            })
        return contexts


_processor: MuRILProcessor | None = None


def get_processing_contexts(texts: list[str]) -> list[dict[str, Any]]:
    """Return MuRIL-backed context for each input message."""
    if not texts:
        return []
    global _processor
    if _processor is None:
        _processor = MuRILProcessor()
    return _processor.process(texts)
