"""
temporal.py -- Extract occurred_at timestamps from file paths and content.

Provides heuristics that parse dates from filenames, JSONL first-line
timestamps, and free-text content.  Used by miner.py and convo_miner.py
to stamp each drawer with the date the original content *occurred*, not
just when it was filed.
"""

import json
import re
from datetime import date, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path-based date extraction
# ---------------------------------------------------------------------------

# ISO date with separators: 2026-04-24.md, transcript_2026-04-24T01:30:00.jsonl
_ISO_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

# Compact date: 20260424_session.jsonl, session_20260424.jsonl
_COMPACT_DATE_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")


def extract_date_from_path(filepath: str) -> str | None:
    """Extract a date from a filepath.  Returns ISO date string or None.

    Patterns recognised (checked against the filename only, not parent dirs):
    - 2026-04-24.md, 2026-04-24.jsonl           (ISO with dashes)
    - transcript_2026-04-24T01:30:00.jsonl       (ISO datetime prefix)
    - 20260424_session.jsonl, session_20260424   (compact 8-digit)
    """
    name = Path(filepath).name

    # Try ISO-separated first (more specific)
    m = _ISO_DATE_RE.search(name)
    if m:
        candidate = m.group(1)
        try:
            datetime.strptime(candidate, "%Y-%m-%d")
            return candidate
        except ValueError:
            pass

    # Try compact 8-digit date
    m = _COMPACT_DATE_RE.search(name)
    if m:
        candidate = m.group(1)
        try:
            dt = datetime.strptime(candidate, "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    return None


# ---------------------------------------------------------------------------
# JSONL first-line timestamp extraction
# ---------------------------------------------------------------------------

# Keys commonly used for timestamps in JSONL session files
_TIMESTAMP_KEYS = ("timestamp", "created_at", "updated_at", "ts", "time", "date")


def extract_date_from_jsonl_first_line(filepath: str) -> str | None:
    """Read the first line of a JSONL file and extract the timestamp field.

    Checks common timestamp keys (timestamp, created_at, updated_at, ts,
    time, date).  The value can be an ISO datetime string or a numeric
    Unix epoch (seconds or milliseconds).

    Returns an ISO date string (YYYY-MM-DD) or None.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline().strip()
    except OSError:
        return None

    if not first_line:
        return None

    try:
        entry = json.loads(first_line)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(entry, dict):
        return None

    for key in _TIMESTAMP_KEYS:
        val = entry.get(key)
        if val is None:
            continue
        parsed = _parse_timestamp_value(val)
        if parsed is not None:
            return parsed

    return None


def _parse_timestamp_value(val) -> str | None:
    """Parse a timestamp value (string or numeric) to an ISO date string."""
    if isinstance(val, (int, float)):
        try:
            # Heuristic: values > 1e12 are likely milliseconds
            if val > 1e12:
                val = val / 1000.0
            dt = datetime.fromtimestamp(val)
            return dt.strftime("%Y-%m-%d")
        except (OSError, OverflowError, ValueError):
            return None

    if not isinstance(val, str):
        return None

    val = val.strip()
    if not val:
        return None

    # Try ISO datetime: 2026-04-18T09:28:44.081Z or 2026-04-18T09:28:44
    m = _ISO_DATE_RE.search(val)
    if m:
        candidate = m.group(1)
        try:
            datetime.strptime(candidate, "%Y-%m-%d")
            return candidate
        except ValueError:
            pass

    return None


# ---------------------------------------------------------------------------
# Content-based date extraction
# ---------------------------------------------------------------------------

# Matches ISO dates in free text: "on 2026-04-24 we discussed..."
_CONTENT_DATE_RE = re.compile(r"\b(\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01]))\b")


def extract_date_from_content(text: str) -> str | None:
    """Extract the earliest valid date mentioned in text content.

    Scans for ISO-formatted dates (YYYY-MM-DD).  Returns the earliest
    one found, or None if no valid dates appear.
    """
    if not text:
        return None

    # Scan a reasonable window -- first 5000 chars
    window = text[:5000]
    matches = _CONTENT_DATE_RE.findall(window)
    if not matches:
        return None

    valid_dates: list[str] = []
    for candidate in matches:
        try:
            datetime.strptime(candidate, "%Y-%m-%d")
            valid_dates.append(candidate)
        except ValueError:
            continue

    if not valid_dates:
        return None

    # Return the earliest date
    valid_dates.sort()
    return valid_dates[0]


# ---------------------------------------------------------------------------
# Unified resolver
# ---------------------------------------------------------------------------


def resolve_occurred_at(
    filepath: str = "",
    content: str = "",
    filed_at: str = "",
) -> str:
    """Resolve the best occurred_at date.

    Priority order:
        1. Date embedded in the filepath (most explicit signal)
        2. Timestamp from the first line of a JSONL file
        3. Earliest date mentioned in content text
        4. filed_at (the ingest timestamp, as fallback)
        5. Today's date (last resort)

    Always returns a YYYY-MM-DD string.
    """
    # 1. Filepath
    if filepath:
        result = extract_date_from_path(filepath)
        if result:
            return result

    # 2. JSONL first-line timestamp (only for .jsonl files)
    if filepath and Path(filepath).suffix.lower() == ".jsonl":
        result = extract_date_from_jsonl_first_line(filepath)
        if result:
            return result

    # 3. Content
    if content:
        result = extract_date_from_content(content)
        if result:
            return result

    # 4. filed_at
    if filed_at:
        m = _ISO_DATE_RE.search(filed_at)
        if m:
            return m.group(1)

    # 5. Today
    return date.today().isoformat()
