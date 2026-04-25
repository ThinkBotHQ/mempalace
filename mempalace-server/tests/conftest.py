"""Pytest configuration — make ``src/`` importable without installing."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Default DSN for tests that touch the auth/deps modules. Tests that talk to
# the DB are skipped unless the env points at a real Postgres.
os.environ.setdefault(
    "MEMPALACE_PGVECTOR_DSN",
    "postgresql://invalid:invalid@127.0.0.1:5432/nonexistent",
)
