"""
test_temporal.py -- Tests for temporal date extraction and resolution.
"""

import json
from datetime import date, datetime

from mempalace.temporal import (
    extract_date_from_content,
    extract_date_from_jsonl_first_line,
    extract_date_from_path,
    resolve_occurred_at,
)


# =========================================================================
# extract_date_from_path
# =========================================================================


class TestExtractDateFromPath:
    """Date extraction from file paths."""

    def test_iso_date_md(self):
        assert extract_date_from_path("/data/2026-04-24.md") == "2026-04-24"

    def test_iso_date_jsonl(self):
        assert extract_date_from_path("/convos/2026-04-24.jsonl") == "2026-04-24"

    def test_iso_datetime_prefix(self):
        assert (
            extract_date_from_path("/convos/transcript_2026-04-24T01:30:00.jsonl") == "2026-04-24"
        )

    def test_compact_date_prefix(self):
        assert extract_date_from_path("/convos/20260424_session.jsonl") == "2026-04-24"

    def test_compact_date_suffix(self):
        assert extract_date_from_path("/convos/session_20260424.jsonl") == "2026-04-24"

    def test_no_date(self):
        assert extract_date_from_path("/convos/readme.md") is None

    def test_invalid_date_digits(self):
        # 13th month is invalid
        assert extract_date_from_path("/convos/20261324_session.jsonl") is None

    def test_iso_preferred_over_compact(self):
        # When both forms appear, ISO with dashes should win
        assert extract_date_from_path("/convos/2025-01-15_20260424.jsonl") == "2025-01-15"

    def test_parent_dirs_ignored(self):
        # Only the filename is checked, not parent directory names
        assert extract_date_from_path("/data/2026-01-01/notes.md") is None

    def test_empty_path(self):
        assert extract_date_from_path("") is None


# =========================================================================
# extract_date_from_jsonl_first_line
# =========================================================================


class TestExtractDateFromJsonlFirstLine:
    """JSONL first-line timestamp extraction."""

    def test_iso_timestamp_field(self, tmp_path):
        f = tmp_path / "session.jsonl"
        f.write_text(json.dumps({"type": "human", "timestamp": "2026-04-18T09:28:44.081Z"}) + "\n")
        assert extract_date_from_jsonl_first_line(str(f)) == "2026-04-18"

    def test_created_at_field(self, tmp_path):
        f = tmp_path / "session.jsonl"
        f.write_text(json.dumps({"created_at": "2025-12-01T00:00:00"}) + "\n")
        assert extract_date_from_jsonl_first_line(str(f)) == "2025-12-01"

    def test_unix_epoch_seconds(self, tmp_path):
        f = tmp_path / "session.jsonl"
        # 2026-04-24 00:00:00 UTC = 1777017600
        ts = datetime(2026, 4, 24).timestamp()
        f.write_text(json.dumps({"timestamp": ts}) + "\n")
        assert extract_date_from_jsonl_first_line(str(f)) == "2026-04-24"

    def test_unix_epoch_milliseconds(self, tmp_path):
        f = tmp_path / "session.jsonl"
        ts = datetime(2026, 4, 24).timestamp() * 1000
        f.write_text(json.dumps({"timestamp": ts}) + "\n")
        assert extract_date_from_jsonl_first_line(str(f)) == "2026-04-24"

    def test_no_timestamp_keys(self, tmp_path):
        f = tmp_path / "session.jsonl"
        f.write_text(json.dumps({"type": "human", "message": "hello"}) + "\n")
        assert extract_date_from_jsonl_first_line(str(f)) is None

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        assert extract_date_from_jsonl_first_line(str(f)) is None

    def test_invalid_json(self, tmp_path):
        f = tmp_path / "bad.jsonl"
        f.write_text("not json at all\n")
        assert extract_date_from_jsonl_first_line(str(f)) is None

    def test_nonexistent_file(self):
        assert extract_date_from_jsonl_first_line("/nonexistent/file.jsonl") is None

    def test_array_first_line(self, tmp_path):
        """First line is a JSON array -- not a dict, should return None."""
        f = tmp_path / "arr.jsonl"
        f.write_text(json.dumps([1, 2, 3]) + "\n")
        assert extract_date_from_jsonl_first_line(str(f)) is None


# =========================================================================
# extract_date_from_content
# =========================================================================


class TestExtractDateFromContent:
    """Date extraction from free-text content."""

    def test_single_date(self):
        assert extract_date_from_content("On 2026-04-24 we discussed the plan.") == "2026-04-24"

    def test_multiple_dates_returns_earliest(self):
        text = "We met on 2026-06-15 and again on 2026-03-10."
        assert extract_date_from_content(text) == "2026-03-10"

    def test_no_dates(self):
        assert extract_date_from_content("No dates here at all.") is None

    def test_invalid_date_month_13(self):
        assert extract_date_from_content("Date: 2026-13-01 is wrong") is None

    def test_invalid_date_day_32(self):
        assert extract_date_from_content("Date: 2026-01-32 is wrong") is None

    def test_empty_content(self):
        assert extract_date_from_content("") is None

    def test_none_content(self):
        assert extract_date_from_content(None) is None

    def test_date_at_start(self):
        assert extract_date_from_content("2025-01-01 is the start.") == "2025-01-01"

    def test_scans_only_first_5000_chars(self):
        """Dates beyond the 5000-char window should be ignored."""
        text = "x" * 5001 + " 2026-12-25 Christmas"
        assert extract_date_from_content(text) is None

    def test_date_within_window(self):
        # "2026-12-25" is 10 chars, plus a space = 11.  5000 - 11 = 4989.
        text = "x" * 4989 + " 2026-12-25"
        assert len(text) == 5000
        assert extract_date_from_content(text) == "2026-12-25"


# =========================================================================
# resolve_occurred_at — priority order
# =========================================================================


class TestResolveOccurredAt:
    """Priority: filepath > JSONL timestamp > content > filed_at > today."""

    def test_filepath_wins_over_all(self, tmp_path):
        f = tmp_path / "2025-06-01.jsonl"
        f.write_text(json.dumps({"timestamp": "2024-01-01T00:00:00"}) + "\n")
        result = resolve_occurred_at(
            filepath=str(f),
            content="On 2023-05-05 we talked.",
            filed_at="2026-04-24T12:00:00",
        )
        assert result == "2025-06-01"

    def test_jsonl_timestamp_when_no_path_date(self, tmp_path):
        f = tmp_path / "session.jsonl"
        f.write_text(json.dumps({"timestamp": "2024-03-15T10:00:00"}) + "\n")
        result = resolve_occurred_at(
            filepath=str(f),
            content="No date references here.",
            filed_at="2026-04-24T12:00:00",
        )
        assert result == "2024-03-15"

    def test_content_date_not_used_in_resolution(self, tmp_path):
        """Content-based date extraction is excluded from resolve_occurred_at
        because it returns the earliest date in the text, which is often a
        historical reference rather than the actual event date. The resolver
        should fall through to filed_at instead."""
        f = tmp_path / "notes.txt"
        f.write_text("Just some notes.\n")
        result = resolve_occurred_at(
            filepath=str(f),
            content="We discussed this on 2025-11-20.",
            filed_at="2026-04-24T12:00:00",
        )
        assert result == "2026-04-24"

    def test_filed_at_fallback(self):
        result = resolve_occurred_at(
            filepath="",
            content="No dates here.",
            filed_at="2026-04-24T12:00:00",
        )
        assert result == "2026-04-24"

    def test_today_last_resort(self):
        result = resolve_occurred_at(filepath="", content="", filed_at="")
        assert result == date.today().isoformat()

    def test_jsonl_skipped_for_non_jsonl_files(self, tmp_path):
        """JSONL first-line extraction only triggers for .jsonl files."""
        f = tmp_path / "data.txt"
        f.write_text(json.dumps({"timestamp": "2020-01-01T00:00:00"}) + "\n")
        result = resolve_occurred_at(
            filepath=str(f),
            content="",
            filed_at="2026-04-24T12:00:00",
        )
        # Should NOT extract 2020-01-01 from the .txt file's first line;
        # falls through to filed_at
        assert result == "2026-04-24"

    def test_filed_at_iso_datetime_parsed(self):
        result = resolve_occurred_at(
            filepath="",
            content="",
            filed_at="2026-04-24T15:30:00.123456",
        )
        assert result == "2026-04-24"

    def test_returns_string_always(self):
        result = resolve_occurred_at()
        assert isinstance(result, str)
        # Verify it is a valid ISO date
        datetime.strptime(result, "%Y-%m-%d")
