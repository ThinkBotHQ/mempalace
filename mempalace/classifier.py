"""Auto-classify drawers into wings/rooms using embedding cosine similarity.

Uses existing Gemini embeddings — no additional API calls needed.
When a drawer is added without a clear wing/room, this module computes
cosine similarity between the drawer's embedding and cached wing name
embeddings to suggest the best match.
"""

from typing import Optional

import numpy as np

_wing_embeddings_cache: dict[str, list[float]] = {}


def set_wing_embeddings(embeddings: dict[str, list[float]]) -> None:
    """Cache wing name embeddings. Called once when taxonomy changes."""
    global _wing_embeddings_cache
    _wing_embeddings_cache = embeddings


def classify_to_wing(
    text_embedding: list[float],
    min_confidence: float = 0.3,
) -> tuple[Optional[str], float]:
    """Classify an embedding to the most similar wing.

    Returns (wing_name, confidence) or (None, 0.0) if below threshold.
    """
    if not _wing_embeddings_cache:
        return None, 0.0

    text_vec = np.array(text_embedding)
    text_norm = text_vec / np.linalg.norm(text_vec)

    best_wing = None
    best_score = -1.0

    for wing_name, wing_vec in _wing_embeddings_cache.items():
        w = np.array(wing_vec)
        w_norm = w / np.linalg.norm(w)
        score = float(np.dot(text_norm, w_norm))
        if score > best_score:
            best_score = score
            best_wing = wing_name

    if best_score < min_confidence:
        return None, best_score
    return best_wing, best_score
