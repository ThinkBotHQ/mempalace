-- MemPalace pgvector schema
-- Run against a PostgreSQL 15+ database with pgvector extension

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Collection registry (palace + collection identity)
CREATE TABLE IF NOT EXISTS mp_collections (
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
CREATE TABLE IF NOT EXISTS mp_documents (
    collection_id   uuid NOT NULL REFERENCES mp_collections(id) ON DELETE CASCADE,
    item_id         text NOT NULL,
    document        text NOT NULL,
    metadata        jsonb NOT NULL DEFAULT '{}',
    embedding       vector(768) NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (collection_id, item_id)
);

-- HNSW index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_mp_docs_embedding_hnsw
    ON mp_documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 24, ef_construction = 128);

-- GIN index for jsonb metadata filtering
CREATE INDEX IF NOT EXISTS idx_mp_docs_metadata_gin
    ON mp_documents USING gin (metadata jsonb_path_ops);

-- Expression indexes for hot metadata paths
CREATE INDEX IF NOT EXISTS idx_mp_docs_wing ON mp_documents ((metadata->>'wing'));
CREATE INDEX IF NOT EXISTS idx_mp_docs_room ON mp_documents ((metadata->>'room'));
CREATE INDEX IF NOT EXISTS idx_mp_docs_source ON mp_documents ((metadata->>'source_file'));

-- Collection lookup
CREATE INDEX IF NOT EXISTS idx_mp_docs_collection ON mp_documents (collection_id);

-- ── Knowledge Graph ───────────────────────────────────────────────────────
-- Temporal entity-relationship graph (people, projects, tools, concepts + typed edges)
-- Mirrors the SQLite schema in mempalace/knowledge_graph.py but uses jsonb
-- for properties and pg_trgm for fuzzy entity name search.

CREATE TABLE IF NOT EXISTS mp_kg_entities (
    id              text PRIMARY KEY,
    name            text NOT NULL,
    type            text DEFAULT 'unknown',
    properties      jsonb DEFAULT '{}',
    created_at      timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_kg_entities_name_trgm
    ON mp_kg_entities USING gin (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_kg_entities_type
    ON mp_kg_entities (type);

CREATE TABLE IF NOT EXISTS mp_kg_triples (
    id              text PRIMARY KEY,
    subject         text NOT NULL REFERENCES mp_kg_entities(id),
    predicate       text NOT NULL,
    object          text NOT NULL REFERENCES mp_kg_entities(id),
    valid_from      text,
    valid_to        text,
    confidence      real DEFAULT 1.0,
    source_closet   text,
    source_file     text,
    source_drawer_id text,
    adapter_name    text,
    metadata        jsonb DEFAULT '{}',
    extracted_at    timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_kg_triples_subject ON mp_kg_triples(subject);
CREATE INDEX IF NOT EXISTS idx_kg_triples_object ON mp_kg_triples(object);
CREATE INDEX IF NOT EXISTS idx_kg_triples_predicate ON mp_kg_triples(predicate);
CREATE INDEX IF NOT EXISTS idx_kg_triples_valid ON mp_kg_triples(valid_from, valid_to);
CREATE INDEX IF NOT EXISTS idx_kg_triples_source_drawer ON mp_kg_triples(source_drawer_id);

-- Vector search function with iterative scan for filtered queries
CREATE OR REPLACE FUNCTION match_memories(
    query_embedding vector(768),
    match_count     int DEFAULT 10,
    match_threshold float DEFAULT 0.7,
    filter_collection uuid DEFAULT NULL,
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
    WHERE (filter_collection IS NULL OR d.collection_id = filter_collection)
      AND d.embedding <=> query_embedding < (1 - match_threshold)
      AND (filter_metadata = '{}'::jsonb OR d.metadata @> filter_metadata)
    ORDER BY d.embedding <=> query_embedding ASC
    LIMIT least(match_count, 200);
END; $$;
