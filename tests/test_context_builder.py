"""Tests for mempalace.context_builder — context block pipeline."""

from mempalace.context_builder import (
    _collect_kg_facts,
    _extract_entities_from_query,
    _format_context,
    _format_fact,
    _get_recent_drawers,
    _search_relevant,
    build_context,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockKG:
    """Minimal mock matching KnowledgeGraph / PgKnowledgeGraph interface."""

    def __init__(self, triples=None, entity_names=None):
        self._triples = triples or {}  # {entity_name: [triple_dicts]}
        self._entity_names = entity_names or []
        self._lock = _FakeLock()
        self._connection = _FakeConn(entity_names or [])

    def query_entity(self, name, as_of=None, direction="outgoing"):
        return self._triples.get(name, [])

    def _conn(self):
        return self._connection


class _FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _FakeRow:
    """Mimics sqlite3.Row with key access."""

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]


class _FakeConn:
    def __init__(self, entity_names):
        self._entity_names = entity_names

    def execute(self, sql, params=None):
        return _FakeCursor(self._entity_names)


class _FakeCursor:
    def __init__(self, entity_names):
        self._entity_names = entity_names

    def fetchall(self):
        return [_FakeRow({"name": n}) for n in self._entity_names]


class MockCollection:
    """Minimal mock matching BaseCollection.get interface."""

    def __init__(self, documents=None, metadatas=None):
        self._docs = documents or []
        self._metas = metadatas or []

    def get(self, *, include=None, where=None, limit=None, **kwargs):
        docs = self._docs
        metas = self._metas

        # Simulate wing filtering
        if where and "wing" in where:
            target_wing = where["wing"]
            filtered_docs = []
            filtered_metas = []
            for d, m in zip(docs, metas):
                if m.get("wing") == target_wing:
                    filtered_docs.append(d)
                    filtered_metas.append(m)
            docs = filtered_docs
            metas = filtered_metas

        if limit:
            docs = docs[:limit]
            metas = metas[:limit]

        return _MockGetResult(docs, metas)


class _MockGetResult:
    def __init__(self, documents, metadatas):
        self.documents = documents
        self.metadatas = metadatas
        self.ids = [f"id_{i}" for i in range(len(documents))]


def _make_search_fn(results=None):
    """Return a search_fn callable that returns canned results."""

    def search_fn(query, *, wing=None, n_results=5):
        return {"query": query, "results": results or []}

    return search_fn


# ---------------------------------------------------------------------------
# Test: entity extraction
# ---------------------------------------------------------------------------


class TestExtractEntitiesFromQuery:
    def test_known_entity_detected(self):
        entities = _extract_entities_from_query(
            "How is Alice doing?",
            known_entity_names=["Alice", "Bob"],
        )
        assert "Alice" in entities

    def test_unknown_entity_not_in_known(self):
        entities = _extract_entities_from_query(
            "Tell me about Charlie",
            known_entity_names=["Alice"],
        )
        assert "Charlie" in entities

    def test_stopwords_excluded(self):
        entities = _extract_entities_from_query(
            "What does Alice know about the project?",
            known_entity_names=[],
        )
        # "What" and "Alice" -- only Alice should remain (What is stopword).
        assert "Alice" in entities
        for word in ("What", "The"):
            assert word not in entities

    def test_sentence_initial_entity(self):
        entities = _extract_entities_from_query(
            "Alice went to the park",
            known_entity_names=[],
        )
        assert "Alice" in entities

    def test_empty_query(self):
        entities = _extract_entities_from_query("", known_entity_names=["Alice"])
        assert entities == []

    def test_case_insensitive_known_match(self):
        entities = _extract_entities_from_query(
            "tell me about alice",
            known_entity_names=["Alice"],
        )
        assert "Alice" in entities

    def test_multiple_entities(self):
        entities = _extract_entities_from_query(
            "Alice and Bob went to see Charlie",
            known_entity_names=["Alice", "Bob"],
        )
        assert "Alice" in entities
        assert "Bob" in entities
        assert "Charlie" in entities

    def test_no_duplicates(self):
        entities = _extract_entities_from_query(
            "Alice saw Alice again",
            known_entity_names=["Alice"],
        )
        assert entities.count("Alice") == 1


# ---------------------------------------------------------------------------
# Test: KG fact collection
# ---------------------------------------------------------------------------


class TestCollectKGFacts:
    def test_collects_current_facts(self):
        kg = MockKG(
            triples={
                "Alice": [
                    {
                        "subject": "Alice",
                        "predicate": "married_to",
                        "object": "Bob",
                        "valid_from": "2015-04-01",
                        "valid_to": None,
                        "current": True,
                    },
                ]
            }
        )
        facts = _collect_kg_facts(kg, ["Alice"])
        assert len(facts) == 1
        assert facts[0]["subject"] == "Alice"
        assert facts[0]["predicate"] == "married_to"
        assert facts[0]["object"] == "Bob"

    def test_skips_expired_facts(self):
        kg = MockKG(
            triples={
                "Alice": [
                    {
                        "subject": "Alice",
                        "predicate": "lives_in",
                        "object": "Portland",
                        "valid_from": "2020-01-01",
                        "valid_to": "2023-06-01",
                        "current": False,
                    },
                ]
            }
        )
        facts = _collect_kg_facts(kg, ["Alice"])
        assert len(facts) == 0

    def test_includes_expired_when_as_of_set(self):
        kg = MockKG(
            triples={
                "Alice": [
                    {
                        "subject": "Alice",
                        "predicate": "lives_in",
                        "object": "Portland",
                        "valid_from": "2020-01-01",
                        "valid_to": "2023-06-01",
                        "current": False,
                    },
                ]
            }
        )
        # When as_of is set, KG already filters by date -- we trust the KG
        # and include what it returns.
        facts = _collect_kg_facts(kg, ["Alice"], as_of="2022-01-01")
        assert len(facts) == 1

    def test_deduplicates_facts(self):
        kg = MockKG(
            triples={
                "Alice": [
                    {
                        "subject": "Alice",
                        "predicate": "married_to",
                        "object": "Bob",
                        "valid_from": "2015-04-01",
                        "valid_to": None,
                        "current": True,
                    },
                ],
                "Bob": [
                    {
                        "subject": "Alice",
                        "predicate": "married_to",
                        "object": "Bob",
                        "valid_from": "2015-04-01",
                        "valid_to": None,
                        "current": True,
                    },
                ],
            }
        )
        facts = _collect_kg_facts(kg, ["Alice", "Bob"])
        assert len(facts) == 1

    def test_respects_max_triples(self):
        triples = [
            {
                "subject": "Alice",
                "predicate": f"rel_{i}",
                "object": f"thing_{i}",
                "valid_from": None,
                "valid_to": None,
                "current": True,
            }
            for i in range(50)
        ]
        kg = MockKG(triples={"Alice": triples})
        facts = _collect_kg_facts(kg, ["Alice"], max_triples=10)
        assert len(facts) == 10

    def test_empty_entities_list(self):
        kg = MockKG()
        facts = _collect_kg_facts(kg, [])
        assert facts == []

    def test_kg_query_failure_handled(self):
        class FailingKG:
            def query_entity(self, name, as_of=None, direction="outgoing"):
                raise RuntimeError("DB is down")

        facts = _collect_kg_facts(FailingKG(), ["Alice"])
        assert facts == []


# ---------------------------------------------------------------------------
# Test: recent drawers
# ---------------------------------------------------------------------------


class TestGetRecentDrawers:
    def test_returns_sorted_by_date(self):
        col = MockCollection(
            documents=["old text", "new text", "mid text"],
            metadatas=[
                {"wing": "personal", "room": "day1", "filed_at": "2026-01-01T10:00:00"},
                {"wing": "personal", "room": "day3", "filed_at": "2026-01-03T10:00:00"},
                {"wing": "personal", "room": "day2", "filed_at": "2026-01-02T10:00:00"},
            ],
        )
        recent = _get_recent_drawers(col, max_recent=3)
        assert len(recent) == 3
        assert recent[0]["date"] == "2026-01-03T10:00:00"
        assert recent[1]["date"] == "2026-01-02T10:00:00"
        assert recent[2]["date"] == "2026-01-01T10:00:00"

    def test_respects_max_recent(self):
        col = MockCollection(
            documents=["a", "b", "c"],
            metadatas=[
                {"filed_at": "2026-01-03"},
                {"filed_at": "2026-01-02"},
                {"filed_at": "2026-01-01"},
            ],
        )
        recent = _get_recent_drawers(col, max_recent=2)
        assert len(recent) == 2

    def test_wing_filtering(self):
        col = MockCollection(
            documents=["proj text", "personal text"],
            metadatas=[
                {"wing": "project", "room": "r1", "filed_at": "2026-01-03"},
                {"wing": "personal", "room": "r2", "filed_at": "2026-01-02"},
            ],
        )
        recent = _get_recent_drawers(col, max_recent=5, wing="personal")
        assert len(recent) == 1
        assert recent[0]["wing"] == "personal"

    def test_empty_collection(self):
        col = MockCollection(documents=[], metadatas=[])
        recent = _get_recent_drawers(col, max_recent=5)
        assert recent == []

    def test_truncates_long_text(self):
        long_text = "x" * 500
        col = MockCollection(
            documents=[long_text],
            metadatas=[{"filed_at": "2026-01-01"}],
        )
        recent = _get_recent_drawers(col, max_recent=5)
        assert len(recent) == 1
        assert recent[0]["text"].endswith("...")
        assert len(recent[0]["text"]) == 303  # 300 chars + "..."


# ---------------------------------------------------------------------------
# Test: search results
# ---------------------------------------------------------------------------


class TestSearchRelevant:
    def test_collects_search_hits(self):
        search_fn = _make_search_fn(
            results=[
                {
                    "text": "pgvector backend is now live",
                    "wing": "mempalace",
                    "room": "decisions",
                    "similarity": 0.82,
                },
            ]
        )
        hits = _search_relevant(search_fn, "pgvector migration")
        assert len(hits) == 1
        assert hits[0]["similarity"] == 0.82

    def test_handles_search_error(self):
        def failing_search(query, *, wing=None, n_results=5):
            raise RuntimeError("search failed")

        hits = _search_relevant(failing_search, "anything")
        assert hits == []

    def test_handles_error_response(self):
        def error_search(query, *, wing=None, n_results=5):
            return {"error": "No palace found"}

        hits = _search_relevant(error_search, "anything")
        assert hits == []

    def test_respects_max_drawers(self):
        results = [
            {"text": f"result {i}", "wing": "w", "room": "r", "similarity": 0.5} for i in range(20)
        ]
        search_fn = _make_search_fn(results=results)
        hits = _search_relevant(search_fn, "query", max_drawers=3)
        assert len(hits) == 3


# ---------------------------------------------------------------------------
# Test: formatting
# ---------------------------------------------------------------------------


class TestFormatting:
    def test_format_fact(self):
        fact = {
            "subject": "Alice",
            "predicate": "married_to",
            "object": "Bob",
            "valid_from": "2015-04-01",
        }
        line = _format_fact(fact)
        assert "Alice" in line
        assert "married to" in line  # underscore replaced with space
        assert "Bob" in line
        assert "(since 2015-04-01)" in line

    def test_format_fact_no_date(self):
        fact = {"subject": "Max", "predicate": "loves", "object": "chess"}
        line = _format_fact(fact)
        assert "Max loves chess" in line
        assert "since" not in line

    def test_formatted_output_structure(self):
        known_facts = [
            {
                "subject": "Alice",
                "predicate": "married_to",
                "object": "Bob",
                "valid_from": "2015-04-01",
            }
        ]
        recent = [
            {
                "wing": "mempalace",
                "room": "backend",
                "text": "pgvector migration completed",
                "date": "2026-04-20",
            },
        ]
        relevant = [
            {
                "wing": "mempalace",
                "room": "decisions",
                "text": "pgvector backend is now live",
                "similarity": 0.82,
            }
        ]
        formatted = _format_context(known_facts, recent, relevant, ["Alice"])
        assert "## Known facts" in formatted
        assert "## Recent activity" in formatted
        assert "## Relevant memories" in formatted
        assert "Alice married to Bob" in formatted
        assert "0.82 similarity" in formatted

    def test_formatted_empty_sections(self):
        formatted = _format_context([], [], [], [])
        assert formatted == ""

    def test_formatted_partial_sections(self):
        facts = [{"subject": "Max", "predicate": "loves", "object": "chess"}]
        formatted = _format_context(facts, [], [], ["Max"])
        assert "## Known facts" in formatted
        assert "## Recent activity" not in formatted
        assert "## Relevant memories" not in formatted


# ---------------------------------------------------------------------------
# Test: build_context (integration)
# ---------------------------------------------------------------------------


class TestBuildContext:
    def _make_kg(self):
        return MockKG(
            triples={
                "Alice": [
                    {
                        "subject": "Alice",
                        "predicate": "married_to",
                        "object": "Bob",
                        "valid_from": "2015-04-01",
                        "valid_to": None,
                        "current": True,
                    },
                    {
                        "subject": "Max",
                        "predicate": "child_of",
                        "object": "Alice",
                        "valid_from": "2015-04-01",
                        "valid_to": None,
                        "current": True,
                        "direction": "incoming",
                    },
                ],
            },
            entity_names=["Alice", "Bob", "Max"],
        )

    def _make_collection(self):
        return MockCollection(
            documents=["recent doc 1", "recent doc 2"],
            metadatas=[
                {"wing": "personal", "room": "diary", "filed_at": "2026-04-20T10:00:00"},
                {"wing": "work", "room": "standup", "filed_at": "2026-04-19T09:00:00"},
            ],
        )

    def _make_search_fn(self):
        return _make_search_fn(
            results=[
                {
                    "text": "Alice and Bob celebrated their anniversary",
                    "wing": "personal",
                    "room": "events",
                    "similarity": 0.88,
                }
            ]
        )

    def test_full_pipeline(self):
        ctx = build_context(
            query="How is Alice doing?",
            kg=self._make_kg(),
            search_fn=self._make_search_fn(),
            collection=self._make_collection(),
        )
        # Structure checks
        assert "known_facts" in ctx
        assert "recent_activity" in ctx
        assert "relevant_drawers" in ctx
        assert "entities_detected" in ctx
        assert "formatted" in ctx

        # Entity detection
        assert "Alice" in ctx["entities_detected"]

        # KG facts
        assert len(ctx["known_facts"]) >= 1
        assert any(f["predicate"] == "married_to" for f in ctx["known_facts"])

        # Formatted string is non-empty
        assert len(ctx["formatted"]) > 0
        assert "## Known facts" in ctx["formatted"]

    def test_empty_kg_returns_gracefully(self):
        ctx = build_context(
            query="What happened today?",
            kg=MockKG(),
            search_fn=_make_search_fn(results=[]),
            collection=MockCollection(),
        )
        assert ctx["known_facts"] == []
        assert ctx["entities_detected"] == []
        assert ctx["recent_activity"] == []
        assert ctx["relevant_drawers"] == []
        assert ctx["formatted"] == ""

    def test_wing_filtering(self):
        col = MockCollection(
            documents=["work doc", "personal doc"],
            metadatas=[
                {"wing": "work", "room": "r1", "filed_at": "2026-04-20"},
                {"wing": "personal", "room": "r2", "filed_at": "2026-04-19"},
            ],
        )
        ctx = build_context(
            query="status update",
            kg=MockKG(),
            search_fn=_make_search_fn(),
            collection=col,
            wing="work",
        )
        for drawer in ctx["recent_activity"]:
            assert drawer["wing"] == "work"

    def test_as_of_parameter_passed(self):
        """Ensure as_of is forwarded to KG query."""
        calls = []

        class TrackingKG:
            _lock = _FakeLock()

            def _conn(self):
                return _FakeConn([])

            def query_entity(self, name, as_of=None, direction="outgoing"):
                calls.append({"name": name, "as_of": as_of})
                return []

        build_context(
            query="Alice in January",
            kg=TrackingKG(),
            search_fn=_make_search_fn(),
            collection=MockCollection(),
            as_of="2026-01-15",
        )
        # Alice should be detected and queried with as_of
        alice_calls = [c for c in calls if c["name"] == "Alice"]
        assert len(alice_calls) >= 1
        assert alice_calls[0]["as_of"] == "2026-01-15"

    def test_return_type_is_dict(self):
        ctx = build_context(
            query="hello",
            kg=MockKG(),
            search_fn=_make_search_fn(),
            collection=MockCollection(),
        )
        assert isinstance(ctx, dict)
        expected_keys = {
            "known_facts",
            "recent_activity",
            "relevant_drawers",
            "entities_detected",
            "formatted",
        }
        assert set(ctx.keys()) == expected_keys

    def test_max_triples_respected(self):
        triples = [
            {
                "subject": "Alice",
                "predicate": f"rel_{i}",
                "object": f"obj_{i}",
                "valid_from": None,
                "valid_to": None,
                "current": True,
            }
            for i in range(30)
        ]
        kg = MockKG(triples={"Alice": triples}, entity_names=["Alice"])
        ctx = build_context(
            query="Tell me about Alice",
            kg=kg,
            search_fn=_make_search_fn(),
            collection=MockCollection(),
            max_triples=5,
        )
        assert len(ctx["known_facts"]) == 5
