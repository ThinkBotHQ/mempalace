"""Tests for mempalace.classifier — embedding-similarity wing classification."""

import math

from mempalace import classifier


def _unit(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


def setup_function(_fn):
    """Reset module-level cache before each test."""
    classifier.set_wing_embeddings({})


# ── set_wing_embeddings ──────────────────────────────────────────────


def test_set_wing_embeddings_replaces_cache():
    classifier.set_wing_embeddings({"alpha": [1.0, 0.0]})
    classifier.set_wing_embeddings({"beta": [0.0, 1.0]})
    wing, _ = classifier.classify_to_wing([0.0, 1.0])
    assert wing == "beta"


# ── classify_to_wing ─────────────────────────────────────────────────


def test_classify_to_wing_empty_cache_returns_none():
    wing, score = classifier.classify_to_wing([1.0, 0.0, 0.0])
    assert wing is None
    assert score == 0.0


def test_classify_to_wing_picks_most_similar():
    classifier.set_wing_embeddings(
        {
            "people": _unit([1.0, 0.0, 0.0]),
            "projects": _unit([0.0, 1.0, 0.0]),
            "topics": _unit([0.0, 0.0, 1.0]),
        }
    )
    # Vector that aligns with "projects"
    wing, score = classifier.classify_to_wing([0.1, 0.95, 0.05])
    assert wing == "projects"
    assert score > 0.9


def test_classify_to_wing_below_threshold_returns_none():
    # All wings near-orthogonal to the query so cosine similarity is low.
    classifier.set_wing_embeddings(
        {
            "alpha": _unit([1.0, 0.0, 0.0, 0.0]),
            "beta": _unit([0.0, 1.0, 0.0, 0.0]),
        }
    )
    wing, score = classifier.classify_to_wing(
        [0.05, 0.05, 1.0, 1.0],
        min_confidence=0.5,
    )
    assert wing is None
    # Score is still reported even when below threshold.
    assert 0.0 <= score < 0.5


def test_classify_to_wing_default_threshold():
    classifier.set_wing_embeddings({"only": _unit([1.0, 0.0])})
    # Vector clearly aligned — well above default 0.3.
    wing, score = classifier.classify_to_wing([1.0, 0.0])
    assert wing == "only"
    assert score > 0.99


def test_classify_to_wing_handles_unnormalized_inputs():
    """Cosine similarity must normalize internally — caller may pass raw vectors."""
    classifier.set_wing_embeddings({"big": [10.0, 0.0, 0.0]})
    wing, score = classifier.classify_to_wing([0.001, 0.0, 0.0])
    assert wing == "big"
    assert score == 1.0 or math.isclose(score, 1.0, rel_tol=1e-6)
