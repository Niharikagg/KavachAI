"""
embeddings.py — Text embedding generation.

Converts text into a dense vector representation using a lightweight
sentence-transformer model. Used downstream (likely by the privacy module)
to measure semantic similarity between original and sanitized text —
ensuring anonymization doesn't destroy the meaning of the message.
"""

from sentence_transformers import SentenceTransformer

# Load once at module level — loading per-call would be very slow.
# 'all-MiniLM-L6-v2' is small (~80MB), fast, and good enough for
# similarity comparison tasks like this.
_model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embedding(text: str) -> list[float]:
    """
    Generate a semantic embedding vector for a piece of text.

    Args:
        text: Input text (e.g. original or sanitized message).

    Returns:
        A list of floats representing the text's embedding (384-dim
        for this model). Empty list if input is empty/invalid.
    """
    if not text or not text.strip():
        return []

    vector = _model.encode(text, convert_to_numpy=True)
    return vector.tolist()


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Utility: compute cosine similarity between two embeddings.
    Useful for comparing original vs sanitized text meaning-preservation.

    Returns a value between -1 and 1 (1 = identical meaning).
    """
    import numpy as np
    a, b = np.array(vec_a), np.array(vec_b)
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Quick manual test
if __name__ == "__main__":
    text1 = "I am a 23-year-old woman living in Village X with diabetes."
    text2 = "A young woman in a small village has a health condition."
    text3 = "The weather today is sunny and warm."

    emb1 = create_embedding(text1)
    emb2 = create_embedding(text2)
    emb3 = create_embedding(text3)

    print(f"Embedding length: {len(emb1)}")
    print(f"Similarity (related sentences): {cosine_similarity(emb1, emb2):.3f}")
    print(f"Similarity (unrelated sentences): {cosine_similarity(emb1, emb3):.3f}")