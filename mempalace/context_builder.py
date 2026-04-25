"""Build a context block from KG facts + recent drawers + search results.

Pipeline approach: entity detection -> KG traversal -> search -> rerank -> pack.
No LLM needed -- all retrieval + ranking. Runs in <100ms.

Usage:
    from mempalace.context_builder import build_context

    ctx = build_context(
        query="How is Alice doing?",
        kg=kg_instance,
        search_fn=search_memories,
        collection=drawers_collection,
    )
    # ctx["formatted"] is ready for system-prompt injection
    # ctx["known_facts"], ctx["relevant_drawers"], etc. for structured access
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

logger = logging.getLogger("mempalace.context_builder")

# ---------------------------------------------------------------------------
# Entity extraction from query text
# ---------------------------------------------------------------------------

# Capitalized-word regex: captures words starting with uppercase that are at
# least 2 chars long and consist of word characters. Skips sentence-initial
# words by requiring a non-start-of-string, non-period predecessor.
_CAPITALIZED_WORD_RE = re.compile(r"(?<!\.\s)(?<!^)\b([A-Z][a-z]{1,})\b")

# Sentence-initial words we never treat as entities.
_QUERY_STOPWORDS = frozenset(
    {
        "the",
        "what",
        "when",
        "where",
        "who",
        "how",
        "why",
        "does",
        "did",
        "has",
        "had",
        "have",
        "was",
        "were",
        "will",
        "would",
        "could",
        "should",
        "can",
        "may",
        "might",
        "shall",
        "is",
        "are",
        "been",
        "being",
        "about",
        "tell",
        "show",
        "find",
        "search",
        "look",
        "get",
        "give",
        "list",
        "any",
        "all",
        "some",
        "most",
        "last",
        "first",
        "recent",
        "new",
        "old",
        "other",
        "know",
        "known",
        "remember",
        "recall",
        "much",
        "many",
        "this",
        "that",
        "these",
        "those",
        "with",
        "from",
        "into",
        "for",
        "and",
        "but",
        "not",
        "also",
        "just",
        "only",
        "very",
        "more",
        "less",
        "then",
        "than",
        "here",
        "there",
    }
)


def _extract_entities_from_query(
    query: str,
    known_entity_names: list[str],
) -> list[str]:
    """Extract entity names from a query string.

    Uses two strategies:
    1. Match any known entity name (from KG or registry) via word-boundary
       regex -- this is the high-confidence path.
    2. Find capitalized words that are not common query stopwords -- this
       catches entities not yet in the KG.

    Returns a deduplicated list of entity name strings.
    """
    found: dict[str, None] = {}  # ordered set via dict

    # Strategy 1: known entity names (case-insensitive word boundary match)
    for name in known_entity_names:
        if not name:
            continue
        pattern = rf"\b{re.escape(name)}\b"
        if re.search(pattern, query, re.IGNORECASE):
            found[name] = None

    # Strategy 2: capitalized words not at sentence start
    for match in _CAPITALIZED_WORD_RE.finditer(query):
        word = match.group(1)
        if word.lower() not in _QUERY_STOPWORDS and word not in found:
            found[word] = None

    # Strategy 3: also check the very first word if it is capitalized and
    # not a stopword -- queries like "Alice's birthday?" start with the entity.
    first_word_match = re.match(r"^([A-Z][a-z]{1,})\b", query)
    if first_word_match:
        word = first_word_match.group(1)
        if word.lower() not in _QUERY_STOPWORDS and word not in found:
            found[word] = None

    return list(found)


# ---------------------------------------------------------------------------
# KG fact collection
# ---------------------------------------------------------------------------


def _collect_kg_facts(
    kg: Any,
    entities: list[str],
    *,
    max_triples: int = 20,
    as_of: str | None = None,
) -> list[dict]:
    """Query the KG for each entity and collect current triples.

    Works with both ``KnowledgeGraph`` (SQLite) and ``PgKnowledgeGraph``
    (PostgreSQL) -- both expose ``query_entity(name, as_of=, direction=)``.

    Returns a list of fact dicts, capped at ``max_triples``.
    """
    facts: list[dict] = []
    seen_keys: set[tuple] = set()

    for entity_name in entities:
        try:
            triples = kg.query_entity(entity_name, as_of=as_of, direction="both")
        except Exception as exc:
            logger.debug("KG query failed for %r: %s", entity_name, exc)
            continue

        for triple in triples:
            # Deduplicate by (subject, predicate, object)
            key = (
                triple.get("subject", ""),
                triple.get("predicate", ""),
                triple.get("object", ""),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)

            # Only include current (non-expired) facts unless as_of filtering
            # already handled that.
            if not as_of and not triple.get("current", True):
                continue

            facts.append(
                {
                    "subject": triple.get("subject", ""),
                    "predicate": triple.get("predicate", ""),
                    "object": triple.get("object", ""),
                    "valid_from": triple.get("valid_from"),
                    "valid_to": triple.get("valid_to"),
                }
            )

            if len(facts) >= max_triples:
                return facts

    return facts


# ---------------------------------------------------------------------------
# Recent drawer retrieval
# ---------------------------------------------------------------------------


def _get_recent_drawers(
    collection: Any,
    *,
    max_recent: int = 5,
    wing: str | None = None,
) -> list[dict]:
    """Retrieve the most recent drawers from the collection.

    Calls ``collection.get(...)`` and sorts by ``filed_at`` descending.
    Returns a list of drawer dicts.
    """
    try:
        where: dict | None = None
        if wing:
            where = {"wing": wing}

        kwargs: dict[str, Any] = {
            "include": ["documents", "metadatas"],
        }
        if where is not None:
            kwargs["where"] = where
        # Fetch more than needed so we can sort by date and take top N.
        kwargs["limit"] = max_recent * 3

        results = collection.get(**kwargs)
    except Exception as exc:
        logger.debug("Failed to get recent drawers: %s", exc)
        return []

    # Normalize access -- supports both GetResult (attribute) and dict shapes.
    docs = getattr(results, "documents", None)
    if docs is None and isinstance(results, dict):
        docs = results.get("documents", [])
    metas = getattr(results, "metadatas", None)
    if metas is None and isinstance(results, dict):
        metas = results.get("metadatas", [])

    if not docs:
        return []

    # Pair docs + metadata, sort by filed_at descending.
    paired: list[tuple[str, dict]] = []
    for doc, meta in zip(docs, metas or [{}] * len(docs)):
        meta = meta or {}
        paired.append((doc, meta))

    paired.sort(
        key=lambda p: p[1].get("filed_at", "") or "",
        reverse=True,
    )

    recent: list[dict] = []
    for doc, meta in paired[:max_recent]:
        recent.append(
            {
                "wing": meta.get("wing", "unknown"),
                "room": meta.get("room", "unknown"),
                "text": (doc[:300] + "...") if len(doc) > 300 else doc,
                "date": meta.get("filed_at", "unknown"),
            }
        )
    return recent


# ---------------------------------------------------------------------------
# Search-result collection
# ---------------------------------------------------------------------------


def _search_relevant(
    search_fn: Callable[..., dict],
    query: str,
    *,
    max_drawers: int = 5,
    wing: str | None = None,
) -> list[dict]:
    """Call the search function and normalize results into drawer dicts.

    ``search_fn`` is expected to match the signature of
    ``searcher.search_memories``, returning a dict with a ``results`` key
    containing a list of hit dicts.
    """
    try:
        raw = search_fn(query, wing=wing, n_results=max_drawers * 3)
    except Exception as exc:
        logger.debug("Search failed: %s", exc)
        return []

    # Handle both dict-with-results and dict-with-error shapes.
    if isinstance(raw, dict) and "error" in raw:
        logger.debug("Search returned error: %s", raw["error"])
        return []

    hits: list[dict] = []
    if isinstance(raw, dict):
        raw_results = raw.get("results", [])
    else:
        raw_results = []

    for hit in raw_results[:max_drawers]:
        text = hit.get("text", "")
        hits.append(
            {
                "wing": hit.get("wing", "unknown"),
                "room": hit.get("room", "unknown"),
                "text": (text[:300] + "...") if len(text) > 300 else text,
                "similarity": hit.get("similarity", 0.0),
            }
        )
    return hits


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _format_fact(fact: dict) -> str:
    """Format a single KG fact as a human-readable line."""
    subj = fact.get("subject", "?")
    pred = fact.get("predicate", "?").replace("_", " ")
    obj = fact.get("object", "?")
    since = fact.get("valid_from")
    suffix = f" (since {since})" if since else ""
    return f"{subj} {pred} {obj}{suffix}"


def _format_context(
    known_facts: list[dict],
    recent_activity: list[dict],
    relevant_drawers: list[dict],
    entities_detected: list[str],
) -> str:
    """Produce a human-readable formatted string for system-prompt injection."""
    sections: list[str] = []

    # Known facts
    if known_facts:
        lines = ["## Known facts"]
        for fact in known_facts:
            lines.append(f"- {_format_fact(fact)}")
        sections.append("\n".join(lines))

    # Recent activity
    if recent_activity:
        lines = [f"## Recent activity (last {len(recent_activity)})"]
        for drawer in recent_activity:
            wing = drawer.get("wing", "?")
            room = drawer.get("room", "?")
            text = drawer.get("text", "").split("\n")[0][:120]
            lines.append(f"- [{wing}/{room}] {text}")
        sections.append("\n".join(lines))

    # Relevant memories
    if relevant_drawers:
        lines = ["## Relevant memories"]
        for drawer in relevant_drawers:
            wing = drawer.get("wing", "?")
            room = drawer.get("room", "?")
            text = drawer.get("text", "").split("\n")[0][:120]
            sim = drawer.get("similarity", 0.0)
            lines.append(f'- [{wing}/{room}] "{text}" ({sim:.2f} similarity)')
        sections.append("\n".join(lines))

    if not sections:
        return ""

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Known entity list helper
# ---------------------------------------------------------------------------


def _get_known_entity_names(kg: Any) -> list[str]:
    """Extract a list of known entity names from the KG.

    Works by reading the entities table. Falls back to an empty list on error.
    """
    try:
        # Both KnowledgeGraph and PgKnowledgeGraph store entities in a table
        # with a ``name`` column. We use a lightweight query.
        if hasattr(kg, "_conn") and hasattr(kg, "_lock"):
            with kg._lock:
                conn = kg._conn()
                rows = conn.execute("SELECT name FROM entities").fetchall()
                # rows are sqlite3.Row or dict depending on backend
                names = []
                for row in rows:
                    if isinstance(row, dict):
                        names.append(row.get("name", ""))
                    else:
                        # sqlite3.Row supports key access
                        try:
                            names.append(row["name"])
                        except (KeyError, IndexError):
                            pass
                return [n for n in names if n]
    except Exception as exc:
        logger.debug("Could not list KG entities: %s", exc)
    return []


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_context(
    query: str,
    kg: Any,
    search_fn: Callable[..., dict],
    collection: Any,
    *,
    max_triples: int = 20,
    max_drawers: int = 5,
    max_recent: int = 5,
    wing: str | None = None,
    as_of: str | None = None,
) -> dict:
    """Build a context block for injection into a conversation.

    Pipeline steps:
      1. Extract entities from the query (regex + known KG entities).
      2. Query KG for each detected entity -- collect current triples.
      3. Search for relevant drawers via ``search_fn``.
      4. Retrieve recent drawers from the collection.
      5. Format everything into both structured and human-readable output.

    Parameters
    ----------
    query:
        The user's natural-language query or message.
    kg:
        A ``KnowledgeGraph`` or ``PgKnowledgeGraph`` instance.
    search_fn:
        A callable matching ``search_memories(query, *, wing=, n_results=)``
        that returns ``{"results": [...]}``.
    collection:
        A ``BaseCollection`` instance for drawer retrieval.
    max_triples:
        Maximum number of KG facts to include.
    max_drawers:
        Maximum number of search-result drawers to include.
    max_recent:
        Maximum number of recent drawers to include.
    wing:
        Optional wing filter applied to search and recent drawers.
    as_of:
        Optional date string for temporal KG filtering.

    Returns
    -------
    dict with keys:
        ``known_facts`` -- list of fact dicts from the KG
        ``recent_activity`` -- list of recent drawer dicts
        ``relevant_drawers`` -- list of search-hit drawer dicts
        ``entities_detected`` -- list of entity name strings
        ``formatted`` -- human-readable string for system-prompt injection
    """
    # Step 1: entity extraction
    known_names = _get_known_entity_names(kg)
    entities = _extract_entities_from_query(query, known_names)

    # Step 2: KG traversal
    known_facts = _collect_kg_facts(
        kg,
        entities,
        max_triples=max_triples,
        as_of=as_of,
    )

    # Step 3: search for relevant drawers
    relevant_drawers = _search_relevant(
        search_fn,
        query,
        max_drawers=max_drawers,
        wing=wing,
    )

    # Step 4: recent drawers
    recent_activity = _get_recent_drawers(
        collection,
        max_recent=max_recent,
        wing=wing,
    )

    # Step 5: format
    formatted = _format_context(
        known_facts,
        recent_activity,
        relevant_drawers,
        entities,
    )

    return {
        "known_facts": known_facts,
        "recent_activity": recent_activity,
        "relevant_drawers": relevant_drawers,
        "entities_detected": entities,
        "formatted": formatted,
    }
