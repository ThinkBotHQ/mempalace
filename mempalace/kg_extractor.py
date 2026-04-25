"""Extract KG triples from drawer text using gpt-5.4-mini structured output.

This module turns verbatim drawer text into structured (subject, predicate,
object) triples and writes them into the temporal knowledge graph. The LLM
extraction is conservative — it only emits facts explicitly stated, never
inferred — which keeps the graph faithful to the drawer it was derived from.

Usage:
    from mempalace.kg_extractor import backfill_drawers
    stats = backfill_drawers(collection, kg, batch_size=50)

CLI:
    python -m mempalace.kg_extractor --backfill --batch-size 50
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


# The prompt deliberately uses an example block that contains JSON braces; we
# keep the f-string limited to a single ``{text}`` insertion to avoid escape
# noise. ``str.format`` is therefore fed only one placeholder.
EXTRACTION_PROMPT = """Extract factual relationships from this text as JSON.
Return an array of objects with: subject, predicate, object, valid_from (optional ISO date).
Only extract facts explicitly stated. Do not infer or hallucinate.
Use simple predicates like: child_of, works_on, lives_in, loves, does, has, uses, married_to, etc.

Text:
{text}

Return ONLY valid JSON array. Example:
[{{"subject": "Max", "predicate": "loves", "object": "chess", "valid_from": "2025-10-01"}}]
"""


def _extract_json_array(raw: str) -> str:
    """Return the first JSON array substring in ``raw``, or ``raw`` unchanged.

    LLMs sometimes wrap output in code fences or add prose. We tolerate that by
    locating the first ``[`` and last ``]`` and slicing between them.
    """
    if not raw:
        return raw
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        return raw
    return raw[start : end + 1]


def _validate_triple(item: Any) -> Optional[dict]:
    """Coerce one parsed JSON item into a clean triple dict, or None if invalid."""
    if not isinstance(item, dict):
        return None
    subject = item.get("subject")
    predicate = item.get("predicate")
    obj = item.get("object")
    if not isinstance(subject, str) or not subject.strip():
        return None
    if not isinstance(predicate, str) or not predicate.strip():
        return None
    if not isinstance(obj, str) or not obj.strip():
        return None
    triple: dict[str, Any] = {
        "subject": subject.strip(),
        "predicate": predicate.strip(),
        "object": obj.strip(),
    }
    valid_from = item.get("valid_from")
    if isinstance(valid_from, str) and valid_from.strip():
        triple["valid_from"] = valid_from.strip()
    return triple


def extract_triples(text: str, model: str = "gpt-5.4-mini") -> list[dict]:
    """Extract KG triples from ``text`` via the OpenAI chat completions API.

    Returns a list of validated triple dicts with at least ``subject``,
    ``predicate``, and ``object`` keys, plus optional ``valid_from``. Returns
    an empty list on any failure path (no API key, network error, malformed
    JSON, empty input, cloud features disabled) — callers should treat
    extraction as best-effort.

    Requires MEMPALACE_ALLOW_CLOUD_FEATURES=1 to be set, since this sends
    user content to an external API. Defaults to OFF for privacy.
    """
    if not os.environ.get("MEMPALACE_ALLOW_CLOUD_FEATURES"):
        return []

    if not text or not text.strip():
        return []

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.debug("OPENAI_API_KEY not set; skipping triple extraction")
        return []

    prompt = EXTRACTION_PROMPT.format(text=text)

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.0,
        )
    except Exception as e:  # network / auth / rate-limit
        logger.warning("KG extraction API call failed: %s", e)
        return []

    try:
        raw = response.choices[0].message.content or ""
    except (AttributeError, IndexError) as e:
        logger.warning("KG extraction response had unexpected shape: %s", e)
        return []

    payload = _extract_json_array(raw.strip())
    try:
        parsed = json.loads(payload)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("KG extraction returned non-JSON output: %s", e)
        return []

    if not isinstance(parsed, list):
        logger.warning("KG extraction returned non-array JSON: %r", type(parsed).__name__)
        return []

    triples: list[dict] = []
    for item in parsed:
        triple = _validate_triple(item)
        if triple is not None:
            triples.append(triple)
    return triples


# ── Tracker (already-extracted drawer ids) ────────────────────────────────


def _load_tracker(tracker_path: Optional[str]) -> set[str]:
    if not tracker_path:
        return set()
    path = Path(tracker_path)
    if not path.exists():
        return set()
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read KG extraction tracker %s: %s", tracker_path, e)
        return set()
    if isinstance(data, list):
        return {str(item) for item in data}
    if isinstance(data, dict) and isinstance(data.get("drawer_ids"), list):
        return {str(item) for item in data["drawer_ids"]}
    return set()


def _save_tracker(tracker_path: Optional[str], drawer_ids: set[str]) -> None:
    if not tracker_path:
        return
    path = Path(tracker_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump({"drawer_ids": sorted(drawer_ids)}, fh, indent=2)
    except OSError as e:
        logger.warning("Could not write KG extraction tracker %s: %s", tracker_path, e)


# ── Backfill ──────────────────────────────────────────────────────────────


def backfill_drawers(
    collection,
    kg,
    *,
    batch_size: int = 50,
    max_drawers: Optional[int] = None,
    tracker_path: Optional[str] = None,
    model: str = "gpt-5.4-mini",
) -> dict:
    """Walk every drawer in ``collection`` and extract triples into ``kg``.

    Args:
        collection: a ``BaseCollection`` (Chroma, pgvector, …) to read from.
        kg: a ``KnowledgeGraph`` or ``PgKnowledgeGraph`` to write triples to.
            Must expose ``add_triple(subject, predicate, obj, *, valid_from,
            source_drawer_id, adapter_name)``.
        batch_size: number of drawers to fetch + process per page.
        max_drawers: stop after considering this many drawers (None = all).
        tracker_path: path to a JSON file recording already-extracted ids so
            re-runs are incremental. Set to None to disable tracking.
        model: chat model identifier passed to ``extract_triples``.

    Returns:
        dict with keys ``drawers_processed``, ``drawers_skipped``,
        ``triples_extracted``, ``triples_added``, ``errors``.
    """
    extracted_ids = _load_tracker(tracker_path)

    stats = {
        "drawers_processed": 0,
        "drawers_skipped": 0,
        "triples_extracted": 0,
        "triples_added": 0,
        "errors": 0,
    }

    offset = 0
    considered = 0
    while True:
        if max_drawers is not None and considered >= max_drawers:
            break

        page_size = batch_size
        if max_drawers is not None:
            page_size = min(page_size, max_drawers - considered)
            if page_size <= 0:
                break

        try:
            page = collection.get(
                limit=page_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.error("Failed to read drawers at offset %d: %s", offset, e)
            stats["errors"] += 1
            break

        ids = list(page.ids or [])
        if not ids:
            break

        documents = list(page.documents or [])
        metadatas = list(page.metadatas or [])
        # Pad in case a backend short-changes one of the parallel lists.
        while len(documents) < len(ids):
            documents.append("")
        while len(metadatas) < len(ids):
            metadatas.append({})

        for drawer_id, doc, meta in zip(ids, documents, metadatas):
            considered += 1
            if drawer_id in extracted_ids:
                stats["drawers_skipped"] += 1
                continue
            if not doc or not str(doc).strip():
                # Mark empty drawers as processed so we never re-poke them.
                extracted_ids.add(drawer_id)
                stats["drawers_skipped"] += 1
                continue

            try:
                triples = extract_triples(doc, model=model)
            except Exception as e:  # defensive — extract_triples already swallows
                logger.warning("Triple extraction crashed on %s: %s", drawer_id, e)
                stats["errors"] += 1
                continue

            stats["drawers_processed"] += 1
            stats["triples_extracted"] += len(triples)

            adapter_name = None
            if isinstance(meta, dict):
                adapter_name = meta.get("adapter_name") or meta.get("adapter")

            for triple in triples:
                try:
                    kg.add_triple(
                        triple["subject"],
                        triple["predicate"],
                        triple["object"],
                        valid_from=triple.get("valid_from"),
                        source_drawer_id=drawer_id,
                        adapter_name=adapter_name,
                    )
                    stats["triples_added"] += 1
                except Exception as e:
                    logger.warning(
                        "kg.add_triple failed for drawer %s (%s -> %s -> %s): %s",
                        drawer_id,
                        triple.get("subject"),
                        triple.get("predicate"),
                        triple.get("object"),
                        e,
                    )
                    stats["errors"] += 1

            extracted_ids.add(drawer_id)

            if max_drawers is not None and considered >= max_drawers:
                break

        # Persist tracker after each page so a Ctrl-C mid-run is recoverable.
        _save_tracker(tracker_path, extracted_ids)

        if len(ids) < page_size:
            break
        offset += len(ids)

    _save_tracker(tracker_path, extracted_ids)
    return stats


# ── CLI ───────────────────────────────────────────────────────────────────


def _build_default_collection_and_kg():
    """Resolve the active palace's collection + KG using the standard config.

    Imported lazily so unit tests can run without dragging the full backend
    stack into module import.
    """
    from mempalace import config
    from mempalace.backends import get_backend
    from mempalace.backends.base import PalaceRef
    from mempalace.knowledge_graph import KnowledgeGraph

    palace_path = config.get_palace_path()
    backend = get_backend()
    palace_ref = PalaceRef(id=str(palace_path), local_path=str(palace_path))
    collection = backend.get_collection(
        palace=palace_ref,
        collection_name=config.get_collection_name(),
        create=False,
    )
    kg = KnowledgeGraph()
    return collection, kg


def main() -> int:
    """CLI: ``python -m mempalace.kg_extractor --backfill``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract KG triples from drawers using gpt-5.4-mini.",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Iterate every drawer and extract triples into the KG.",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--max-drawers",
        type=int,
        default=None,
        help="Stop after this many drawers (default: all).",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=os.path.expanduser("~/.mempalace/kg_extraction_tracker.json"),
        help="Path to JSON tracker file recording extracted drawer ids.",
    )
    parser.add_argument("--model", type=str, default="gpt-5.4-mini")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.backfill:
        parser.error("nothing to do — pass --backfill")

    collection, kg = _build_default_collection_and_kg()
    try:
        stats = backfill_drawers(
            collection,
            kg,
            batch_size=args.batch_size,
            max_drawers=args.max_drawers,
            tracker_path=args.tracker,
            model=args.model,
        )
    finally:
        close = getattr(kg, "close", None)
        if callable(close):
            close()
        coll_close = getattr(collection, "close", None)
        if callable(coll_close):
            coll_close()

    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
