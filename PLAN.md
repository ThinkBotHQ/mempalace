# Plan: MemPalace pgvector Backend

**Status:** Ready for review  
**Created:** 2026-04-23  
**Updated:** 2026-04-24  
**Estimated effort:** ~9 dev-days  
**Migration cost:** ~$4 USD for 100K docs ($2.00 via batch API)  

---

## Executive Summary

Replace ChromaDB (crash-prone Rust bindings) with PostgreSQL + pgvector as
MemPalace's storage backend. Use Gemini embedding-2 at 768 dimensions.
Ship as `mempalace-pgvector` pip package with entry-point registration.
Team memory support via PostgreSQL roles + RLS (future).

---

## Key Decisions

### 1. Embedding Model: `gemini-embedding-2` at 768 dims

| Factor | gemini-embedding-001 | gemini-embedding-2 |
|--------|----------------------|---------------------|
| Status | Stable (June 2025) | **GA (April 22, 2026)** |
| Dims | 3,072 (truncatable) | 3,072 (truncatable) |
| Max tokens | 2,048 | **8,192** |
| task_type param | Yes (parameter) | No (prompt instruction instead) |
| Auto-normalize | No (manual L2 needed) | **Yes** |
| Batch API | Yes ($0.075/1M) | **Yes ($0.10/1M)** |
| Standard price | $0.15/1M | $0.20/1M |

**Choice: `gemini-embedding-2` at 768 dims** because:
- GA as of April 2026, actively developed — embedding-001 is aging
- 8,192 token context future-proofs for larger chunks
- Automatic renormalization for truncated dimensions (no manual L2 needed)
- Batch API supported on Developer API at 50% cost ($0.10/1M)
- Prompt-based task instructions replace `task_type` with equivalent functionality
- 768 dims stays within pgvector's `vector` type index limit (2,000 max)
- 4x less storage than 3,072 (3KB vs 12KB per row)

**task_type replacement:** embedding-2 ignores the `task_type` parameter.
Use prompt instructions in the content string instead:
- Documents: `"task: retrieval document | {text}"`
- Queries: `"task: search query | query: {text}"`

**SDK:** `google-genai` package. `from google import genai`.

### 2. Embedding Ownership: Backend-owned via DI

ChromaDB generates embeddings internally. All MemPalace callers pass
`documents=` without `embeddings=`. Zero caller churn path:

```python
class Embedder(Protocol):
    name: str
    dimension: int
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, texts: list[str]) -> list[list[float]]: ...
```

`PgvectorBackend` accepts `embedder: Embedder | None` and lazily creates
a `GeminiEmbedder` from env. Collection calls `self._embedder.embed()`
inside `add/upsert` when embeddings= is None.

### 3. Python Client: `psycopg[binary]` + `pgvector`

Use psycopg for all vector operations. Direct PostgreSQL connection,
no ORM or REST layer.

### 4. Index: HNSW (not IVFFlat)

HNSW is 15.5x faster at 0.998 recall, handles inserts without rebuild.
Parameters: `m=24, ef_construction=128` for 50K-200K vectors at 768 dims.

### 5. Connection: Direct connection for local dev, pooler optional for production

Direct PostgreSQL connection (port 5434 local). `SET LOCAL` for HNSW tuning
works in direct and session-mode pooler connections. Transaction poolers
work too since `match_memories` uses `SET LOCAL` inside the function body.

### 6. Local Dev: Dedicated Docker container

Use a separate pgvector container for local development (not the existing
THINKS or auto_trade containers).

```bash
docker run -d --name mempalace-pgvector \
  -e POSTGRES_PASSWORD=mempalace \
  -e POSTGRES_DB=mempalace \
  -p 5434:5432 \
  pgvector/pgvector:pg17
```

DSN: `postgresql://postgres:mempalace@localhost:5434/mempalace`

---

## Schema Design

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Collection registry (palace + collection identity)
CREATE TABLE mp_collections (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    palace_id       text NOT NULL,
    namespace       text,
    collection_name text NOT NULL,
    embedder_name   text NOT NULL,
    embedder_dim    int  NOT NULL,
    hnsw_space      text NOT NULL DEFAULT 'cosine',
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (palace_id, collection_name)
);

-- Main content table
CREATE TABLE mp_documents (
    collection_id   uuid NOT NULL REFERENCES mp_collections(id) ON DELETE CASCADE,
    item_id         text NOT NULL,
    document        text NOT NULL,
    metadata        jsonb NOT NULL DEFAULT '{}',
    embedding       vector(768) NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (collection_id, item_id)
);

-- Indexes
CREATE INDEX idx_mp_docs_embedding_hnsw
    ON mp_documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 24, ef_construction = 128);

CREATE INDEX idx_mp_docs_metadata_gin
    ON mp_documents USING gin (metadata jsonb_path_ops);

CREATE INDEX idx_mp_docs_collection ON mp_documents (collection_id);

-- Expression indexes for hot metadata paths
CREATE INDEX idx_mp_docs_wing ON mp_documents ((metadata->>'wing'));
CREATE INDEX idx_mp_docs_room ON mp_documents ((metadata->>'room'));
CREATE INDEX idx_mp_docs_source ON mp_documents ((metadata->>'source_file'));
```

### Vector Search Function

Uses `SET LOCAL hnsw.iterative_scan = relaxed_order` — critical for
filtered vector search. Without it, ANN index may return fewer rows than
requested because metadata filtering happens post-index-scan. Iterative
scan continues scanning until enough results match the filter.

```sql
CREATE FUNCTION match_memories(
    query_embedding vector(768),
    match_count     int DEFAULT 10,
    match_threshold float DEFAULT 0.7,
    filter_metadata jsonb DEFAULT '{}'::jsonb
) RETURNS TABLE (
    item_id    text, document text, metadata jsonb,
    similarity float, collection_id uuid
) LANGUAGE plpgsql STABLE AS $$
BEGIN
    SET LOCAL hnsw.iterative_scan = relaxed_order;
    SET LOCAL hnsw.ef_search = 100;
    RETURN QUERY
    SELECT d.item_id, d.document, d.metadata,
           (1 - (d.embedding <=> query_embedding))::float AS similarity,
           d.collection_id
    FROM mp_documents d
    WHERE d.embedding <=> query_embedding < (1 - match_threshold)
      AND (filter_metadata = '{}'::jsonb OR d.metadata @> filter_metadata)
    ORDER BY d.embedding <=> query_embedding ASC
    LIMIT least(match_count, 200);
END; $$;
```

Note: `<=>` returns cosine **distance** (0 = identical). Similarity =
`1 - distance`.

---

## Package Structure

```
mempalace-pgvector/
  pyproject.toml
  src/mempalace_pgvector/
    __init__.py           # exports PgvectorBackend
    backend.py            # PgvectorBackend(BaseBackend)
    collection.py         # PgvectorCollection(BaseCollection)
    embedder.py           # Embedder protocol + GeminiEmbedder
    where_compiler.py     # where-dict -> parameterized SQL
    schema.sql            # migration DDL (shipped in wheel)
    migrate.py            # ChromaDB -> pgvector migration script
    py.typed
  tests/
```

### Dependencies (pyproject.toml)
```
mempalace>=0.4,<1.0
psycopg[binary]>=3.2.0,<4.0
psycopg-pool>=3.2.0,<4.0
pgvector>=0.3.6,<1.0
google-genai>=0.8.0,<2.0
tenacity>=8.2.0,<10.0
```

Activate: `uv pip install mempalace-pgvector && export MEMPALACE_BACKEND=pgvector`

---

## Migration Strategy

1. **Never touch Chroma data** — read-only export, rollback = env var flip
2. **Re-embed everything** with Gemini (384-dim MiniLM -> 768-dim Gemini, incompatible spaces)
3. **Cursor-file resumability** — crash at doc 50K resumes at 50K
4. **Batch 100 docs/request** to Gemini, exponential backoff on 429
5. **Cost:** ~$4 for 100K docs (20M tokens @ $0.20/1M, or $2.00 via batch API)
6. **Time:** 15-45 minutes for 100K docs

```bash
uv run python -m mempalace_pgvector.migrate \
    --chroma-path ~/.mempalace/palace \
    --collection mempalace_drawers \
    --pgvector-dsn "$PGVECTOR_DSN" \
    --batch-size 100
```

---

## Env Vars

| Variable | Required | Default |
|----------|----------|---------|
| MEMPALACE_BACKEND | Yes | "chroma" |
| MEMPALACE_PGVECTOR_DSN | Yes* | DATABASE_URL fallback |
| GEMINI_API_KEY | Yes | GOOGLE_API_KEY fallback |
| MEMPALACE_EMBEDDER_MODEL | No | "gemini-embedding-2" |
| MEMPALACE_EMBEDDER_DIM | No | 768 |
| MEMPALACE_PALACE_ID | No | derived from palace_path |
| MEMPALACE_PGVECTOR_POOL_MAX | No | 4 |

---

## Agent Routing Table

| Phase | Worker | Effort | Parallel | Validator |
|-------|--------|--------|----------|-----------|
| 1. Embedder module | Codex gpt-5.5 | medium | 1 | Sonnet typecheck |
| 2. where_compiler.py | Codex gpt-5.5 | high | 1 | Sonnet + test suite |
| 3. PgvectorCollection | Codex gpt-5.4 | high | 1 | Sonnet conformance |
| 4. PgvectorBackend | Codex gpt-5.4 | high | 1 | Sonnet |
| 5. Schema SQL + migration | Codex gpt-5.4 | medium | 1 | Manual pgvector test |
| 6. MCP server integration | Opus 4.7 | high | 1 | Sonnet + browser |
| 7. Migration script | Codex gpt-5.5 | medium | 1 | Manual dry-run |
| 8. Packaging + tests | Codex gpt-5.5 | medium | 1 | Sonnet |

Phases 1-4 are sequential (each depends on prior). Phase 5 can parallel
with 3-4. Phases 6-8 after 1-5 complete.

---

## Risk Register

1. **Embedder identity drift** — stored model name mismatch must fail loudly (EmbedderIdentityMismatchError)
2. **Gemini rate limits on bulk mine** — batch 64-100 texts/request, tenacity backoff
3. **jsonb numeric cast failures** — `(metadata->>'k')::numeric` errors on non-numeric; catch and raise UnsupportedFilterError
4. **Connection pool exhaustion** — pool_max=4, monitor with health() check
5. **Palace-ID stability across machines** — explicit MEMPALACE_PALACE_ID slug, not filesystem path
6. **pgvector 2000-dim index limit** — locked to 768 dims; document that changing embedder requires new collection
7. **SET LOCAL in transaction pooler** — works inside plpgsql functions, verified
8. **No offline mode** — pgvector requires a running PostgreSQL instance; document this as a known limitation
9. **Query embedding cost** — every search = 1 Gemini API call; add LRU cache (256 entries) for hot re-queries
10. **Query performance** — expression indexes on hot metadata paths (wing, room, source_file) are critical for filtered vector search
11. **Similarity threshold drift** — cosine similarity scores shift between embedding models; `match_threshold=0.7` in `match_memories()` needs recalibration after migration
12. **Multi-text batching** — embedding-2 may produce aggregated embeddings when passing multiple items in `contents`; test carefully or use one text per `embed_content` call (Batch API file-based approach is safe)

---

## Pre-Existing Issues to Fix

### Save hook runaway mining (PARTIALLY FIXED)

`mempal_save_hook.sh` fires `mempalace mine "$MINE_DIR" &` in background
every 15 human messages. Fire-and-forget with no dedup — old mines
accumulate, eating 100%+ CPU for hours. Uses system Python 3.9 which
hits the crashy ChromaDB Rust bindings.

**Fixed (2026-04-24):**
- Added `_mine_disabled()` guard to all three mining paths in `hooks_cli.py`
- Respects `MEMPALACE_NO_MINE` env var and `~/.mempalace/hook_state/no_mine` flag file
- Fixed `_maybe_auto_ingest` and `_mine_sync` to use `_mempalace_python()` instead of `sys.executable`

**Remaining:**
- Add `fcntl.flock`-based lockfile to prevent concurrent mines
- Unify shell hooks to thin wrappers calling `mempalace hook run`
- Add timeout/watchdog to spawned mine processes

### ChromaDB patches applied (temporary, in site-packages)

Four patches applied directly to installed mempalace package:
1. Split `get_or_create_collection` → `get`/`create` try/except
2. Wired `quarantine_stale_hnsw()` into client init (was dead code)
3. Same for mcp_server.py
4. `CHROMA_SERVER_THREAD_POOL_SIZE=15`

These will be lost on `uv tool upgrade mempalace`. File upstream PRs or
accept that pgvector backend replaces the need for them.

---

## Scope Boundaries (Non-Goals)

- Real-time collaboration via PostgreSQL LISTEN/NOTIFY (future)
- Multi-embedder support within one collection (requires table-per-dimension)
- Web UI for team management (CLI/API only for v0.1)
- Stored procedures beyond match_memories (all logic in Python backend)
- Async/await (sync psycopg for v0.1, async optional later)

---

## Investigation Attribution

| Finding | Source Agent | Unique? |
|---------|-------------|---------|
| Backend-owned embedder via DI (zero caller churn) | Opus 4.7 (arch) | Yes |
| quarantine_stale_hnsw is dead code | Explore agent | Yes |
| get_or_create_collection crash path (mempalace#1089) | Web researcher | Yes |
| pgvector vector type 2000-dim index limit | Best-practices (pgvector) | Yes |
| halfvec casting for >2000 dims | Best-practices (pgvector) | Yes |
| SET LOCAL hnsw.iterative_scan for filtered search | Best-practices (pgvector) | Yes |
| gemini-embedding-2 ignores task_type — use prompt instructions | Best-practices (Gemini) | Yes |
| Batch API at 50% cost (embedding-2: $0.10/1M on Developer API) | Best-practices (Gemini) | Yes |
| embedding-2 auto-normalizes truncated dims (no manual L2) | Best-practices (Gemini) | Yes |
| embedding-2 GA April 22, 2026 — 8192 token context | Best-practices (Gemini) | Yes |
| Similarity threshold drift between embedding models | Best-practices (Gemini) | Yes |
| Cursor-file resumable migration | Opus 4.7 (packaging) | Yes |
| Migration cost ~$4 for 100K docs ($2 batch) | Opus 4.7 (packaging) | Yes |
| Codex gpt-5.5 stdin hang on heredoc background | Conductor (observed) | Yes |
| hnsw.iterative_scan for filtered vector search | Best-practices (pgvector) | Yes |
| pgvector 0.8.2 (current version in Docker pg17 image) | Best-practices (pgvector) | Yes |
| prepare_threshold=0 needed for transaction pooler | Best-practices (pgvector) | Yes |
| Save hook fire-and-forget mining causes CPU burn | Conductor (system investigation) | Yes |
| Gemini embedding-001 batch limit: 250 texts, 20K tokens/req | Best-practices (Gemini) | Yes |
| text-embedding-004 deprecated Jan 14 2026 | Best-practices (Gemini) | Yes |

---

## Gemini Embeddings Quick Reference

```python
from google import genai
from google.genai import types

client = genai.Client()  # uses GOOGLE_API_KEY env var

# Embed documents (for indexing) — task instruction in content string
result = client.models.embed_content(
    model="gemini-embedding-2",
    contents=[
        "task: retrieval document | doc1 text",
        "task: retrieval document | doc2 text",
    ],
    config=types.EmbedContentConfig(
        output_dimensionality=768,
    ),
)
vectors = [e.values for e in result.embeddings]

# Embed query (for search) — task instruction in content string
query_result = client.models.embed_content(
    model="gemini-embedding-2",
    contents="task: search query | query: search terms here",
    config=types.EmbedContentConfig(
        output_dimensionality=768,
    ),
)
query_vec = query_result.embeddings[0].values

# Batch API (for migration, 50% cost via Developer API)
import json

with open("embed_requests.jsonl", "w") as f:
    for text in all_texts:
        f.write(json.dumps({
            "model": "gemini-embedding-2",
            "contents": f"task: retrieval document | {text}",
            "config": {"output_dimensionality": 768},
        }) + "\n")

uploaded = client.files.upload(file="embed_requests.jsonl")
batch_job = client.batches.create_embeddings(
    model="gemini-embedding-2",
    src={"file_name": uploaded.name},
)
```

Notes:
- embedding-2 ignores `task_type` param — use prompt instructions instead
- Auto-normalizes truncated dimensions (no manual L2 normalization needed)
- 8,192 token context per input (4x embedding-001)
- Batch API: Developer API only (not Vertex AI), $0.10/1M tokens (50% off)
- Standard pricing: $0.20/1M tokens, ~1500 RPM (paid tier)
