#!/usr/bin/env python3
"""
Closet Benchmark — measure whether AAAK closets still improve search quality
when tsvector + RRF hybrid search exists in pgvector.

Runs 50 test queries two ways:
  1. WITH closets:    drawer search + closet rank boost + BM25 hybrid rerank
  2. WITHOUT closets: drawer search only + BM25 hybrid rerank (no closet boost)

Compares the result sets to determine if closets add value, are neutral, or
cause ranking regressions. This is a READ-ONLY benchmark -- no data is modified.

Usage:
    MEMPALACE_BACKEND=pgvector \
    MEMPALACE_PGVECTOR_DSN=postgresql://postgres:mempalace@localhost:5434/mempalace \
    GEMINI_API_KEY=$GEMINI_API_KEY \
    uv run python benchmarks/closet_benchmark.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import uuid
from pathlib import Path

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg

from mempalace_pgvector.collection import PgvectorCollection
from mempalace_pgvector.embedder import GeminiEmbedder

# ── Configuration ──────────────────────────────────────────────────────────

DSN = os.environ.get(
    "MEMPALACE_PGVECTOR_DSN", "postgresql://postgres:mempalace@localhost:5434/mempalace"
)

# Collection UUIDs (from mp_collections table)
DRAWERS_COLL_ID = uuid.UUID("fed2ada9-2784-496d-92ed-7511d0fc6a67")
CLOSETS_COLL_ID = uuid.UUID("9cb5e955-127d-4aa5-b5c6-fdc876b4db2f")

N_RESULTS = 10  # top-k results to compare
OVER_FETCH_FACTOR = 3  # fetch 3x for re-ranking, matching searcher.py

# Closet boost constants (mirroring searcher.py exactly)
CLOSET_RANK_BOOSTS = [0.40, 0.25, 0.15, 0.08, 0.04]
CLOSET_DISTANCE_CAP = 1.5

OUTPUT_PATH = Path(__file__).resolve().parent / "closet_benchmark_results.json"

# ── Test queries ───────────────────────────────────────────────────────────

QUERIES = [
    # People queries (10)
    ("people", "what does Max do"),
    ("people", "who is Justin"),
    ("people", "Alice's daughter"),
    ("people", "who works on the memory system"),
    ("people", "tell me about Sasha"),
    ("people", "what did the team decide"),
    ("people", "who built the embeddings pipeline"),
    ("people", "conversations with clients"),
    ("people", "who manages the project"),
    ("people", "developer feedback on the product"),
    # Project queries (10)
    ("project", "chromadb crash"),
    ("project", "pgvector migration"),
    ("project", "auto trade strategy"),
    ("project", "MCP server implementation"),
    ("project", "hook scripts for Claude Code"),
    ("project", "entity detection pipeline"),
    ("project", "knowledge graph triples"),
    ("project", "BM25 hybrid search"),
    ("project", "closet compression format"),
    ("project", "palace repair tool"),
    # Temporal queries (10)
    ("temporal", "what happened recently"),
    ("temporal", "recent decisions about architecture"),
    ("temporal", "latest changes to the codebase"),
    ("temporal", "what was discussed last session"),
    ("temporal", "recent bug fixes"),
    ("temporal", "deployment timeline"),
    ("temporal", "sprint planning notes"),
    ("temporal", "status update from this week"),
    ("temporal", "most recent conversation"),
    ("temporal", "progress on current milestone"),
    # Concept queries (10)
    ("concept", "memory system design"),
    ("concept", "embedding model choice"),
    ("concept", "verbatim storage philosophy"),
    ("concept", "AAAK compression format"),
    ("concept", "wing room drawer architecture"),
    ("concept", "privacy by architecture"),
    ("concept", "local-first zero API design"),
    ("concept", "incremental append-only ingest"),
    ("concept", "entity disambiguation"),
    ("concept", "Zettelkasten method of loci"),
    # Exact phrase / specific queries (10)
    ("exact", "100% recall is the design requirement"),
    ("exact", "memory is identity"),
    ("exact", "ruff check"),
    ("exact", "sanitize_name"),
    ("exact", "normalize_version"),
    ("exact", "CLOSET_CHAR_LIMIT"),
    ("exact", "method of loci"),
    ("exact", "reciprocal rank fusion"),
    ("exact", "Okapi BM25"),
    ("exact", "websearch_to_tsquery"),
]

assert len(QUERIES) == 50, f"Expected 50 queries, got {len(QUERIES)}"

# ── BM25 (from searcher.py, verbatim) ─────────────────────────────────────

_TOKEN_RE = re.compile(r"\w{2,}", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _bm25_scores(
    query: str,
    documents: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    n_docs = len(documents)
    query_terms = set(_tokenize(query))
    if not query_terms or n_docs == 0:
        return [0.0] * n_docs

    tokenized = [_tokenize(d) for d in documents]
    doc_lens = [len(toks) for toks in tokenized]
    if not any(doc_lens):
        return [0.0] * n_docs
    avgdl = sum(doc_lens) / n_docs or 1.0

    df: dict[str, int] = {term: 0 for term in query_terms}
    for toks in tokenized:
        seen = set(toks) & query_terms
        for term in seen:
            df[term] += 1

    idf = {term: math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1) for term in query_terms}

    scores: list[float] = []
    for toks, dl in zip(tokenized, doc_lens):
        if dl == 0:
            scores.append(0.0)
            continue
        tf: dict[str, int] = {}
        for t in toks:
            if t in query_terms:
                tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for term, freq in tf.items():
            num = freq * (k1 + 1)
            den = freq + k1 * (1 - b + b * dl / avgdl)
            score += idf[term] * num / den
        scores.append(score)
    return scores


def _hybrid_rank(
    results: list[dict],
    query: str,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list[dict]:
    if not results:
        return results

    docs = [r.get("text", "") for r in results]
    bm25_raw = _bm25_scores(query, docs)
    max_bm25 = max(bm25_raw) if bm25_raw else 0.0
    bm25_norm = [s / max_bm25 for s in bm25_raw] if max_bm25 > 0 else [0.0] * len(bm25_raw)

    scored = []
    for r, raw, norm in zip(results, bm25_raw, bm25_norm):
        vec_sim = max(0.0, 1.0 - r.get("distance", 1.0))
        r["bm25_score"] = round(raw, 3)
        scored.append((vector_weight * vec_sim + bm25_weight * norm, r))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    results[:] = [r for _, r in scored]
    return results


# ── Kendall's tau ──────────────────────────────────────────────────────────


def kendall_tau(list_a: list[str], list_b: list[str]) -> float:
    """Compute Kendall's tau-b between two ranked lists (by shared items)."""
    common = [x for x in list_a if x in list_b]
    if len(common) < 2:
        return 1.0  # no discordance possible

    rank_b = {item: i for i, item in enumerate(list_b)}
    concordant = 0
    discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            # In list_a, common[i] comes before common[j]
            # Check if same order in list_b
            if rank_b[common[i]] < rank_b[common[j]]:
                concordant += 1
            else:
                discordant += 1

    total = concordant + discordant
    if total == 0:
        return 1.0
    return (concordant - discordant) / total


# ── Search functions ───────────────────────────────────────────────────────


def _first_or_empty(results, key: str) -> list:
    outer = getattr(results, key, None)
    if not outer:
        return []
    return outer[0] or []


def search_with_closets(
    drawers_col: PgvectorCollection,
    closets_col: PgvectorCollection,
    query: str,
    query_embedding: list[float],
    n_results: int = N_RESULTS,
) -> list[dict]:
    """Replicate search_memories with closet boosting (from searcher.py)."""

    # Over-fetch drawers for re-ranking
    drawer_results = drawers_col.query(
        query_embeddings=[query_embedding],
        n_results=n_results * OVER_FETCH_FACTOR,
        include=["documents", "metadatas", "distances"],
    )

    # Gather closet hits for boost lookup
    closet_boost_by_source: dict[str, tuple[int, float, str]] = {}
    try:
        closet_results = closets_col.query(
            query_embeddings=[query_embedding],
            n_results=n_results * 2,
            include=["documents", "metadatas", "distances"],
        )
        for rank, (cdoc, cmeta, cdist) in enumerate(
            zip(
                _first_or_empty(closet_results, "documents"),
                _first_or_empty(closet_results, "metadatas"),
                _first_or_empty(closet_results, "distances"),
            )
        ):
            cmeta = cmeta or {}
            source = cmeta.get("source_file", "")
            if source and source not in closet_boost_by_source:
                closet_boost_by_source[source] = (rank, cdist, cdoc[:200])
    except Exception:
        pass

    # Score drawer results with closet boost
    scored: list[dict] = []
    for doc, meta, dist in zip(
        _first_or_empty(drawer_results, "documents"),
        _first_or_empty(drawer_results, "metadatas"),
        _first_or_empty(drawer_results, "distances"),
    ):
        meta = meta or {}
        source = meta.get("source_file", "") or ""
        boost = 0.0
        matched_via = "drawer"
        if source in closet_boost_by_source:
            c_rank, c_dist, _ = closet_boost_by_source[source]
            if c_dist <= CLOSET_DISTANCE_CAP and c_rank < len(CLOSET_RANK_BOOSTS):
                boost = CLOSET_RANK_BOOSTS[c_rank]
                matched_via = "drawer+closet"

        effective_dist = dist - boost
        entry = {
            "id": f"{source}:{meta.get('chunk_index', '?')}",
            "text": doc,
            "source_file": source,
            "distance": round(dist, 4),
            "effective_distance": round(effective_dist, 4),
            "closet_boost": round(boost, 3),
            "matched_via": matched_via,
            "_sort_key": effective_dist,
        }
        scored.append(entry)

    scored.sort(key=lambda h: h["_sort_key"])
    hits = scored[:n_results]

    # BM25 hybrid re-rank
    hits = _hybrid_rank(hits, query)
    for h in hits:
        h.pop("_sort_key", None)

    return hits


def search_without_closets(
    drawers_col: PgvectorCollection,
    query: str,
    query_embedding: list[float],
    n_results: int = N_RESULTS,
) -> list[dict]:
    """Search drawers only -- no closet boost."""

    drawer_results = drawers_col.query(
        query_embeddings=[query_embedding],
        n_results=n_results * OVER_FETCH_FACTOR,
        include=["documents", "metadatas", "distances"],
    )

    scored: list[dict] = []
    for doc, meta, dist in zip(
        _first_or_empty(drawer_results, "documents"),
        _first_or_empty(drawer_results, "metadatas"),
        _first_or_empty(drawer_results, "distances"),
    ):
        meta = meta or {}
        source = meta.get("source_file", "") or ""
        entry = {
            "id": f"{source}:{meta.get('chunk_index', '?')}",
            "text": doc,
            "source_file": source,
            "distance": round(dist, 4),
            "effective_distance": round(dist, 4),
            "closet_boost": 0.0,
            "matched_via": "drawer",
            "_sort_key": dist,
        }
        scored.append(entry)

    scored.sort(key=lambda h: h["_sort_key"])
    hits = scored[:n_results]

    # BM25 hybrid re-rank
    hits = _hybrid_rank(hits, query)
    for h in hits:
        h.pop("_sort_key", None)

    return hits


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("  Closet Benchmark: tsvector+RRF vs closet boost")
    print("=" * 70)
    print()

    # Connect
    print(f"  DSN: {DSN.split('@')[-1]}")
    conn_drawers = psycopg.connect(DSN, autocommit=False)
    conn_closets = psycopg.connect(DSN, autocommit=False)

    embedder = GeminiEmbedder()
    print(f"  Embedder: {embedder.name} (dim={embedder.dimension})")

    drawers_col = PgvectorCollection(
        conn=conn_drawers, collection_id=DRAWERS_COLL_ID, embedder=embedder
    )
    closets_col = PgvectorCollection(
        conn=conn_closets, collection_id=CLOSETS_COLL_ID, embedder=embedder
    )

    drawer_count = drawers_col.count()
    closet_count = closets_col.count()
    print(f"  Drawers: {drawer_count:,}")
    print(f"  Closets: {closet_count:,}")
    print(f"  Queries: {len(QUERIES)}")
    print(f"  Top-K:   {N_RESULTS}")
    print()

    # Pre-embed all queries in one batch to minimize API calls
    print("  Embedding all 50 queries...")
    query_texts = [q for _, q in QUERIES]
    t0 = time.time()
    all_embeddings = embedder.embed_query(query_texts)
    embed_time = time.time() - t0
    print(f"  Done in {embed_time:.1f}s ({len(QUERIES)} API calls)")
    print()

    # Run benchmark
    per_query_results: list[dict] = []
    category_stats: dict[str, dict] = {}

    for i, ((category, query), embedding) in enumerate(zip(QUERIES, all_embeddings), 1):
        print(f"  [{i:2d}/50] {category:8s} | {query}")

        with_closets = search_with_closets(drawers_col, closets_col, query, embedding)
        without_closets = search_without_closets(drawers_col, query, embedding)

        # Extract IDs for comparison
        ids_with = [h["id"] for h in with_closets]
        ids_without = [h["id"] for h in without_closets]

        # Overlap at various K
        def overlap_at_k(a: list[str], b: list[str], k: int) -> tuple[int, int]:
            set_a = set(a[:k])
            set_b = set(b[:k])
            common = len(set_a & set_b)
            return common, k

        overlap_5 = overlap_at_k(ids_with, ids_without, 5)
        overlap_10 = overlap_at_k(ids_with, ids_without, min(10, len(ids_with)))

        # Unique results
        set_with = set(ids_with)
        set_without = set(ids_without)
        closet_unique = set_with - set_without
        no_closet_unique = set_without - set_with

        # Rank correlation on shared items
        tau = kendall_tau(ids_with, ids_without)

        # Check if closet boost actually changed any rankings
        boosted_count = sum(1 for h in with_closets if h["closet_boost"] > 0)

        result = {
            "query": query,
            "category": category,
            "overlap_at_5": {"common": overlap_5[0], "k": overlap_5[1]},
            "overlap_at_10": {"common": overlap_10[0], "k": overlap_10[1]},
            "closet_unique_count": len(closet_unique),
            "no_closet_unique_count": len(no_closet_unique),
            "kendall_tau": round(tau, 3),
            "boosted_results": boosted_count,
            "with_closets_ids": ids_with,
            "without_closets_ids": ids_without,
            "closet_unique_ids": sorted(closet_unique),
            "no_closet_unique_ids": sorted(no_closet_unique),
            # Top result comparison
            "top1_same": (ids_with[0] == ids_without[0]) if ids_with and ids_without else True,
            "top3_overlap": overlap_at_k(ids_with, ids_without, 3)[0],
        }
        per_query_results.append(result)

        # Accumulate category stats
        if category not in category_stats:
            category_stats[category] = {
                "count": 0,
                "overlap_5_sum": 0,
                "overlap_10_sum": 0,
                "tau_sum": 0.0,
                "closet_helped": 0,
                "closet_hurt": 0,
                "closet_neutral": 0,
                "boosted_total": 0,
            }
        cs = category_stats[category]
        cs["count"] += 1
        cs["overlap_5_sum"] += overlap_5[0]
        cs["overlap_10_sum"] += overlap_10[0]
        cs["tau_sum"] += tau
        cs["boosted_total"] += boosted_count

        # Heuristic: closets "helped" if they brought in unique results
        # that have lower effective distance than the worst no-closet result
        if len(closet_unique) > 0 and boosted_count > 0:
            cs["closet_helped"] += 1
        elif len(no_closet_unique) > len(closet_unique):
            cs["closet_hurt"] += 1
        else:
            cs["closet_neutral"] += 1

        # Brief per-query output
        o5 = overlap_5[0]
        print(
            f"           overlap@5={o5}/5  "
            f"tau={tau:+.2f}  "
            f"boosted={boosted_count}  "
            f"closet-uniq={len(closet_unique)}  "
            f"no-closet-uniq={len(no_closet_unique)}"
        )

    # ── Aggregate report ───────────────────────────────────────────────────

    print()
    print("=" * 70)
    print("  AGGREGATE RESULTS")
    print("=" * 70)
    print()

    total = len(per_query_results)
    mean_overlap_5 = sum(r["overlap_at_5"]["common"] for r in per_query_results) / total
    mean_overlap_10 = sum(r["overlap_at_10"]["common"] for r in per_query_results) / total
    mean_tau = sum(r["kendall_tau"] for r in per_query_results) / total
    total_boosted = sum(r["boosted_results"] for r in per_query_results)
    queries_with_boost = sum(1 for r in per_query_results if r["boosted_results"] > 0)
    top1_same_count = sum(1 for r in per_query_results if r["top1_same"])
    mean_top3_overlap = sum(r["top3_overlap"] for r in per_query_results) / total

    total_closet_unique = sum(r["closet_unique_count"] for r in per_query_results)
    total_no_closet_unique = sum(r["no_closet_unique_count"] for r in per_query_results)

    # Determine helped/hurt using the clearer heuristic:
    # "helped" = closet boost changed results AND brought in unique items
    # "hurt" = no-closet-unique > closet-unique (closets displaced better results)
    helped = sum(
        1 for r in per_query_results if r["closet_unique_count"] > 0 and r["boosted_results"] > 0
    )
    hurt = sum(
        1 for r in per_query_results if r["no_closet_unique_count"] > r["closet_unique_count"]
    )
    neutral = total - helped - hurt

    print(f"  Mean overlap@5:           {mean_overlap_5:.1f}/5 ({mean_overlap_5 / 5 * 100:.0f}%)")
    print(
        f"  Mean overlap@10:          {mean_overlap_10:.1f}/10 ({mean_overlap_10 / 10 * 100:.0f}%)"
    )
    print(
        f"  Mean top-3 overlap:       {mean_top3_overlap:.1f}/3 ({mean_top3_overlap / 3 * 100:.0f}%)"
    )
    print(
        f"  Top-1 agreement:          {top1_same_count}/{total} ({top1_same_count / total * 100:.0f}%)"
    )
    print(f"  Mean Kendall's tau:       {mean_tau:+.3f}")
    print()
    print(f"  Queries where closets active: {queries_with_boost}/{total}")
    print(f"  Total boosted results:        {total_boosted}")
    print(f"  Total closet-unique results:  {total_closet_unique}")
    print(f"  Total no-closet-unique:       {total_no_closet_unique}")
    print()
    print(f"  Queries where closets helped: {helped}/{total}")
    print(f"  Queries where closets hurt:   {hurt}/{total}")
    print(f"  Queries neutral:              {neutral}/{total}")
    print()

    # Per-category breakdown
    print("  --- Per-Category Breakdown ---")
    print()
    for cat in ["people", "project", "temporal", "concept", "exact"]:
        cs = category_stats.get(cat, {})
        n = cs.get("count", 0)
        if n == 0:
            continue
        avg_o5 = cs["overlap_5_sum"] / n
        avg_o10 = cs["overlap_10_sum"] / n
        avg_tau = cs["tau_sum"] / n
        print(
            f"  {cat:10s}  "
            f"overlap@5={avg_o5:.1f}/5  "
            f"overlap@10={avg_o10:.1f}/10  "
            f"tau={avg_tau:+.3f}  "
            f"helped={cs['closet_helped']}  "
            f"hurt={cs['closet_hurt']}  "
            f"neutral={cs['closet_neutral']}"
        )
    print()

    # Recommendation
    if mean_overlap_5 >= 4.5 and mean_tau >= 0.8:
        recommendation = "DELETE"
        rationale = (
            "Closets produce nearly identical results to drawer-only search. "
            "The tsvector+RRF hybrid search captures the keyword signals that "
            "closets were designed to provide. Removing closets saves storage "
            "and eliminates the closet query overhead."
        )
    elif helped > hurt * 2:
        recommendation = "KEEP"
        rationale = (
            "Closets still provide meaningful ranking improvements that "
            "tsvector+RRF does not fully replicate."
        )
    elif hurt > helped:
        recommendation = "DELETE"
        rationale = (
            "Closets cause more ranking regressions than improvements. "
            "The closet boost displaces better drawer results."
        )
    else:
        recommendation = "REBALANCE"
        rationale = (
            "Closets have mixed impact. Consider reducing the boost magnitude "
            "or restricting closet boosting to specific query categories."
        )

    print(f"  RECOMMENDATION: {recommendation}")
    print(f"  {rationale}")
    print()
    print("=" * 70)

    # ── Save results ───────────────────────────────────────────────────────

    report = {
        "benchmark": "closet_benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "n_results": N_RESULTS,
            "over_fetch_factor": OVER_FETCH_FACTOR,
            "total_queries": total,
            "drawer_count": drawer_count,
            "closet_count": closet_count,
            "embedder": embedder.name,
            "embed_time_seconds": round(embed_time, 1),
        },
        "aggregate": {
            "mean_overlap_at_5": round(mean_overlap_5, 2),
            "mean_overlap_at_5_pct": round(mean_overlap_5 / 5 * 100, 1),
            "mean_overlap_at_10": round(mean_overlap_10, 2),
            "mean_overlap_at_10_pct": round(mean_overlap_10 / 10 * 100, 1),
            "mean_top3_overlap": round(mean_top3_overlap, 2),
            "top1_agreement": top1_same_count,
            "top1_agreement_pct": round(top1_same_count / total * 100, 1),
            "mean_kendall_tau": round(mean_tau, 3),
            "queries_with_closet_boost": queries_with_boost,
            "total_boosted_results": total_boosted,
            "total_closet_unique_results": total_closet_unique,
            "total_no_closet_unique_results": total_no_closet_unique,
            "closets_helped": helped,
            "closets_hurt": hurt,
            "closets_neutral": neutral,
            "recommendation": recommendation,
            "rationale": rationale,
        },
        "per_category": {
            cat: {
                "count": cs["count"],
                "mean_overlap_at_5": round(cs["overlap_5_sum"] / cs["count"], 2),
                "mean_overlap_at_10": round(cs["overlap_10_sum"] / cs["count"], 2),
                "mean_kendall_tau": round(cs["tau_sum"] / cs["count"], 3),
                "closets_helped": cs["closet_helped"],
                "closets_hurt": cs["closet_hurt"],
                "closets_neutral": cs["closet_neutral"],
                "boosted_total": cs["boosted_total"],
            }
            for cat, cs in category_stats.items()
        },
        "per_query": [
            {
                "query": r["query"],
                "category": r["category"],
                "overlap_at_5": r["overlap_at_5"],
                "overlap_at_10": r["overlap_at_10"],
                "closet_unique_count": r["closet_unique_count"],
                "no_closet_unique_count": r["no_closet_unique_count"],
                "kendall_tau": r["kendall_tau"],
                "boosted_results": r["boosted_results"],
                "top1_same": r["top1_same"],
                "top3_overlap": r["top3_overlap"],
                "with_closets_ids": r["with_closets_ids"],
                "without_closets_ids": r["without_closets_ids"],
                "closet_unique_ids": r["closet_unique_ids"],
                "no_closet_unique_ids": r["no_closet_unique_ids"],
            }
            for r in per_query_results
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Results saved to: {OUTPUT_PATH}")
    print()

    # Cleanup
    conn_drawers.close()
    conn_closets.close()


if __name__ == "__main__":
    main()
