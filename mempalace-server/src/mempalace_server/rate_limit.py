"""In-memory token-bucket rate limiter keyed by API key name.

Window is a fixed 60-second slot. Default 120 requests/min.
Single-process only — for multi-worker deploys use a shared backend.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Per-key fixed-window counter. Resets every 60 seconds."""

    WINDOW_SECONDS = 60

    def __init__(self) -> None:
        self._buckets: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

    def check(self, key_name: str, limit: int = 120) -> bool:
        """Return True if the request is allowed; False if rate-limited."""
        if limit <= 0:
            return True  # 0 / negative = unlimited
        now = time.monotonic()
        with self._lock:
            count, window_start = self._buckets.get(key_name, (0, now))
            if now - window_start >= self.WINDOW_SECONDS:
                self._buckets[key_name] = (1, now)
                return True
            if count >= limit:
                return False
            self._buckets[key_name] = (count + 1, window_start)
            return True

    def reset(self, key_name: str | None = None) -> None:
        with self._lock:
            if key_name is None:
                self._buckets.clear()
            else:
                self._buckets.pop(key_name, None)
