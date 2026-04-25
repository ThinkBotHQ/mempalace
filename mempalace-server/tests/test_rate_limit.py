"""Rate limiter smoke tests."""

from __future__ import annotations

import time

from mempalace_server.rate_limit import RateLimiter


def test_allows_under_limit():
    limiter = RateLimiter()
    for _ in range(5):
        assert limiter.check("alice", limit=5) is True


def test_blocks_over_limit():
    limiter = RateLimiter()
    for _ in range(5):
        assert limiter.check("bob", limit=5) is True
    assert limiter.check("bob", limit=5) is False


def test_separate_keys_independent():
    limiter = RateLimiter()
    for _ in range(3):
        assert limiter.check("a", limit=3) is True
    assert limiter.check("a", limit=3) is False
    # Different key — fresh bucket.
    assert limiter.check("b", limit=3) is True


def test_window_resets():
    limiter = RateLimiter()
    limiter.WINDOW_SECONDS = 0.05  # type: ignore[misc]
    assert limiter.check("c", limit=1) is True
    assert limiter.check("c", limit=1) is False
    time.sleep(0.06)
    assert limiter.check("c", limit=1) is True


def test_zero_limit_unlimited():
    limiter = RateLimiter()
    for _ in range(1000):
        assert limiter.check("d", limit=0) is True
