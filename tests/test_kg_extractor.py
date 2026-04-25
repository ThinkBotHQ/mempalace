"""Tests for mempalace.kg_extractor — LLM-driven KG triple extraction."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from mempalace import kg_extractor


# ── extract_triples ──────────────────────────────────────────────────────


def _fake_completion(content: str) -> MagicMock:
    """Build a MagicMock shaped like an OpenAI ChatCompletion."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def test_extract_triples_parses_json(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    payload = json.dumps(
        [
            {
                "subject": "Max",
                "predicate": "loves",
                "object": "chess",
                "valid_from": "2025-10-01",
            },
            {"subject": "Alice", "predicate": "works_on", "object": "mempalace"},
        ]
    )

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_completion(payload)

    with patch.object(kg_extractor, "OpenAI", return_value=fake_client) as mock_ctor:
        triples = kg_extractor.extract_triples("Max loves chess. Alice works on mempalace.")

    mock_ctor.assert_called_once_with(api_key="sk-fake")
    call_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-5.4-mini"
    assert call_kwargs["temperature"] == 0.0
    assert "Max loves chess" in call_kwargs["messages"][0]["content"]

    assert triples == [
        {
            "subject": "Max",
            "predicate": "loves",
            "object": "chess",
            "valid_from": "2025-10-01",
        },
        {"subject": "Alice", "predicate": "works_on", "object": "mempalace"},
    ]


def test_extract_triples_strips_prose_around_json(monkeypatch):
    """LLM sometimes wraps JSON in prose / fences — we tolerate that."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    wrapped = (
        "Here you go!\n"
        "```json\n"
        '[{"subject": "Max", "predicate": "does", "object": "swimming"}]\n'
        "```\n"
        "Hope that helps."
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_completion(wrapped)

    with patch.object(kg_extractor, "OpenAI", return_value=fake_client):
        triples = kg_extractor.extract_triples("Max does swimming.")

    assert triples == [{"subject": "Max", "predicate": "does", "object": "swimming"}]


def test_extract_triples_handles_malformed(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_completion(
        "this is not valid json at all"
    )

    with patch.object(kg_extractor, "OpenAI", return_value=fake_client):
        triples = kg_extractor.extract_triples("some text")

    assert triples == []


def test_extract_triples_skips_invalid_entries(monkeypatch):
    """Items missing required fields or with wrong types are dropped silently."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    payload = json.dumps(
        [
            {"subject": "Max", "predicate": "loves", "object": "chess"},
            {"subject": "Max", "predicate": "loves"},  # missing object
            {"subject": "", "predicate": "x", "object": "y"},  # empty subject
            "not even a dict",
            {"subject": 5, "predicate": "is", "object": "five"},  # bad type
            {"subject": "Alice", "predicate": "uses", "object": "vim"},
        ]
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_completion(payload)

    with patch.object(kg_extractor, "OpenAI", return_value=fake_client):
        triples = kg_extractor.extract_triples("text")

    assert triples == [
        {"subject": "Max", "predicate": "loves", "object": "chess"},
        {"subject": "Alice", "predicate": "uses", "object": "vim"},
    ]


def test_extract_triples_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # If it tried to talk to OpenAI we'd notice — but it shouldn't.
    with patch.object(kg_extractor, "OpenAI", side_effect=AssertionError("must not call")):
        result = kg_extractor.extract_triples("Max loves chess.")

    assert result == []


def test_extract_triples_empty_text_short_circuits(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    with patch.object(kg_extractor, "OpenAI", side_effect=AssertionError("must not call")):
        assert kg_extractor.extract_triples("") == []
        assert kg_extractor.extract_triples("   \n  ") == []


def test_extract_triples_swallows_api_errors(monkeypatch, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("network down")

    with patch.object(kg_extractor, "OpenAI", return_value=fake_client):
        with caplog.at_level("WARNING"):
            triples = kg_extractor.extract_triples("anything")

    assert triples == []
    assert any("KG extraction API call failed" in r.message for r in caplog.records)


def test_extract_triples_non_array_root(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_completion('{"oops": "object"}')
    with patch.object(kg_extractor, "OpenAI", return_value=fake_client):
        assert kg_extractor.extract_triples("text") == []


# ── backfill_drawers ─────────────────────────────────────────────────────


class _FakeCollection:
    """Minimal in-memory BaseCollection-like for backfill tests."""

    def __init__(self, drawers: list[tuple[str, str, dict]]):
        self.drawers = drawers
        self.calls: list[dict] = []

    def get(self, *, limit=None, offset=None, include=None, **kwargs):
        self.calls.append({"limit": limit, "offset": offset, "include": include})
        page = self.drawers[offset : (offset or 0) + (limit or len(self.drawers))]
        ids = [d[0] for d in page]
        docs = [d[1] for d in page]
        metas = [d[2] for d in page]
        return SimpleNamespace(ids=ids, documents=docs, metadatas=metas)


class _FakeKG:
    def __init__(self):
        self.added: list[dict] = []

    def add_triple(self, subject, predicate, obj, **kwargs):
        self.added.append(
            {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                **kwargs,
            }
        )
        return f"t_{len(self.added)}"


def test_backfill_skips_already_extracted(tmp_path, monkeypatch):
    tracker = tmp_path / "tracker.json"
    tracker.write_text(json.dumps({"drawer_ids": ["d1", "d2"]}))

    collection = _FakeCollection(
        [
            ("d1", "Max loves chess.", {}),
            ("d2", "Alice uses vim.", {}),
            ("d3", "Bob lives in Paris.", {}),
        ]
    )
    kg = _FakeKG()

    extract_mock = MagicMock(
        return_value=[{"subject": "Bob", "predicate": "lives_in", "object": "Paris"}]
    )
    monkeypatch.setattr(kg_extractor, "extract_triples", extract_mock)

    stats = kg_extractor.backfill_drawers(
        collection,
        kg,
        batch_size=10,
        tracker_path=str(tracker),
    )

    # extract_triples should only have been called for d3.
    assert extract_mock.call_count == 1
    assert extract_mock.call_args.args[0] == "Bob lives in Paris."
    assert stats["drawers_processed"] == 1
    assert stats["drawers_skipped"] == 2
    assert stats["triples_added"] == 1

    # Tracker should now contain all three ids.
    saved = json.loads(tracker.read_text())
    assert set(saved["drawer_ids"]) == {"d1", "d2", "d3"}


def test_backfill_adds_triples_to_kg(tmp_path, monkeypatch):
    tracker = tmp_path / "tracker.json"

    collection = _FakeCollection(
        [
            ("d1", "Max loves chess.", {"adapter_name": "claude_code"}),
            ("d2", "Alice uses vim.", {"adapter": "codex"}),
        ]
    )
    kg = _FakeKG()

    def fake_extract(text, model="gpt-5.4-mini"):
        if "Max" in text:
            return [
                {
                    "subject": "Max",
                    "predicate": "loves",
                    "object": "chess",
                    "valid_from": "2025-10-01",
                }
            ]
        return [{"subject": "Alice", "predicate": "uses", "object": "vim"}]

    monkeypatch.setattr(kg_extractor, "extract_triples", fake_extract)

    stats = kg_extractor.backfill_drawers(
        collection,
        kg,
        batch_size=1,  # exercise pagination
        tracker_path=str(tracker),
    )

    assert stats["drawers_processed"] == 2
    assert stats["triples_extracted"] == 2
    assert stats["triples_added"] == 2
    assert stats["errors"] == 0

    assert kg.added == [
        {
            "subject": "Max",
            "predicate": "loves",
            "object": "chess",
            "valid_from": "2025-10-01",
            "source_drawer_id": "d1",
            "adapter_name": "claude_code",
        },
        {
            "subject": "Alice",
            "predicate": "uses",
            "object": "vim",
            "valid_from": None,
            "source_drawer_id": "d2",
            "adapter_name": "codex",
        },
    ]


def test_backfill_respects_max_drawers(tmp_path, monkeypatch):
    collection = _FakeCollection(
        [(f"d{i}", f"text {i}", {}) for i in range(10)],
    )
    kg = _FakeKG()

    monkeypatch.setattr(
        kg_extractor,
        "extract_triples",
        MagicMock(return_value=[{"subject": "s", "predicate": "p", "object": "o"}]),
    )

    stats = kg_extractor.backfill_drawers(
        collection,
        kg,
        batch_size=4,
        max_drawers=3,
        tracker_path=str(tmp_path / "t.json"),
    )

    assert stats["drawers_processed"] == 3
    assert stats["triples_added"] == 3


def test_backfill_skips_empty_documents(tmp_path, monkeypatch):
    collection = _FakeCollection(
        [
            ("d1", "", {}),
            ("d2", "   ", {}),
            ("d3", "real text here", {}),
        ]
    )
    kg = _FakeKG()
    extract_mock = MagicMock(return_value=[{"subject": "x", "predicate": "y", "object": "z"}])
    monkeypatch.setattr(kg_extractor, "extract_triples", extract_mock)

    stats = kg_extractor.backfill_drawers(
        collection,
        kg,
        batch_size=10,
        tracker_path=str(tmp_path / "t.json"),
    )

    assert extract_mock.call_count == 1
    assert stats["drawers_processed"] == 1
    assert stats["drawers_skipped"] == 2


def test_backfill_handles_kg_errors(tmp_path, monkeypatch):
    collection = _FakeCollection([("d1", "text", {})])

    class _ExplodingKG:
        def add_triple(self, *a, **kw):
            raise RuntimeError("kg down")

    monkeypatch.setattr(
        kg_extractor,
        "extract_triples",
        MagicMock(return_value=[{"subject": "s", "predicate": "p", "object": "o"}]),
    )

    stats = kg_extractor.backfill_drawers(
        collection,
        _ExplodingKG(),
        batch_size=10,
        tracker_path=str(tmp_path / "t.json"),
    )

    assert stats["triples_extracted"] == 1
    assert stats["triples_added"] == 0
    assert stats["errors"] == 1


def test_backfill_no_tracker_path_works(monkeypatch):
    collection = _FakeCollection([("d1", "text", {})])
    kg = _FakeKG()
    monkeypatch.setattr(
        kg_extractor,
        "extract_triples",
        MagicMock(return_value=[{"subject": "s", "predicate": "p", "object": "o"}]),
    )

    stats = kg_extractor.backfill_drawers(collection, kg, batch_size=10, tracker_path=None)
    assert stats["drawers_processed"] == 1
    assert stats["triples_added"] == 1


# ── tracker helpers ──────────────────────────────────────────────────────


def test_tracker_round_trip(tmp_path):
    path = tmp_path / "tracker.json"
    kg_extractor._save_tracker(str(path), {"a", "b", "c"})
    assert kg_extractor._load_tracker(str(path)) == {"a", "b", "c"}


def test_tracker_load_handles_legacy_list(tmp_path):
    path = tmp_path / "tracker.json"
    path.write_text(json.dumps(["x", "y"]))
    assert kg_extractor._load_tracker(str(path)) == {"x", "y"}


def test_tracker_load_missing_file(tmp_path):
    assert kg_extractor._load_tracker(str(tmp_path / "nope.json")) == set()
    assert kg_extractor._load_tracker(None) == set()


def test_tracker_load_corrupt_file(tmp_path, caplog):
    path = tmp_path / "tracker.json"
    path.write_text("{not json")
    with caplog.at_level("WARNING"):
        assert kg_extractor._load_tracker(str(path)) == set()
    assert any("Could not read" in r.message for r in caplog.records)
