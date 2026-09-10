"""Read models for the web dashboard: turn the persisted lineage ``Corpus`` and
the live ``Settings`` into JSON the frontend can render (corpus stats,
per-document ingest stage, redacted config).

This is the ONE place those views are computed (CLAUDE.md reusability): the web
API is a thin transport over these functions, so the CLI or eval harness can
reuse them without duplicating the derivation.

Post-cutover the corpus is lineage-only (no vectors, no chunk text), so the
old RAM-bound capacity model is gone: the local store is small by construction
and retrieval is the engine's. Capacity is disk-bound only."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

from ..config import Settings, load_settings
from ..ingest.index import Corpus

# Keep this much disk free — never let ingestion fill the device (honest about
# scale). Also a nominal per-document size for the very first estimate, before
# any real document has been ingested to measure against.
_DISK_RESERVE_BYTES = 1 * 1024**3        # 1 GiB headroom
_NOMINAL_BYTES_PER_DOC = 8_000           # ~a dozen lineage rows (no text, no vectors)


@dataclass
class DocView:
    doc_id: str
    source: str
    chunks: int
    pages: int          # distinct source pages covered
    page_min: int
    page_max: int
    stage: str          # queued | indexed
    chars: int          # total text characters (rough corpus weight)
    form_type: str = ""   # filing metadata (T22), empty when unknown
    company: str = ""
    ticker: str = ""
    fiscal_period: str = ""


def _doc_stage(chunks: int) -> str:
    """A document's terminal ingest stage from what landed in the lineage.

    Live in-flight stages (ocr/chunking/engine-post) come from the ingest job
    registry; ``indexed`` is the resting state once lineage is persisted (the
    chunks themselves live engine-side)."""
    return "indexed" if chunks else "queued"


def corpus_stats(settings: Settings | None = None, *, corpus: Corpus | None = None) -> dict:
    """Aggregate corpus + per-document stats for the dashboard."""
    s = settings or load_settings()
    c = corpus if corpus is not None else Corpus.load(s.index_dir)

    # Fold lineage rows into per-document accumulators in one pass.
    docs: dict[str, dict] = {}
    for ch in c.chunks:
        d = docs.setdefault(
            ch.doc_id,
            {"source": ch.source, "pages": set(), "chunks": 0, "chars": 0,
             "form_type": ch.form_type, "company": ch.company, "ticker": ch.ticker,
             "fiscal_period": ch.fiscal_period},
        )
        d["chunks"] += 1
        d["chars"] += ch.chars
        d["pages"].add(ch.page)

    views: list[DocView] = []
    for doc_id, d in sorted(docs.items()):
        pages = d["pages"]
        views.append(
            DocView(
                doc_id=doc_id,
                source=d["source"],
                chunks=d["chunks"],
                pages=len(pages),
                page_min=min(pages) if pages else 0,
                page_max=max(pages) if pages else 0,
                stage=_doc_stage(d["chunks"]),
                chars=d["chars"],
                form_type=d["form_type"],
                company=d["company"],
                ticker=d["ticker"],
                fiscal_period=d["fiscal_period"],
            )
        )

    return {
        "documents": [v.__dict__ for v in views],
        "totals": {
            "documents": len(views),
            "chunks": len(c.chunks),
            "chars": sum(v.chars for v in views),
        },
        "retrieval": {
            "mode": s.retrieval_mode,
            "collection": s.engine_collection,
            "embed_model": s.embed_model,
            "rerank_model": s.rerank_model,
        },
        "index_dir": s.index_dir,
    }


def _existing_dir(path: str) -> str:
    """Nearest existing ancestor of ``path`` — ``disk_usage`` needs a real path,
    and ``index_dir`` may not exist until the first ingest."""
    p = path
    while not os.path.exists(p):
        nxt = os.path.dirname(p)
        if nxt == p:
            return "."
        p = nxt
    return p or "."


def _device_specs(index_dir: str) -> dict:
    """Disk (where the index lives) + total RAM, dependency-free and best-effort."""
    try:
        usage = shutil.disk_usage(_existing_dir(index_dir))
        disk_total = usage.total
        disk_free = usage.free
    except OSError:
        disk_total = disk_free = 0
    try:
        ram_total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        ram_total = 0
    return {"disk_total_bytes": disk_total, "disk_free_bytes": disk_free,
            "ram_total_bytes": ram_total}


def capacity_stats(settings: Settings | None = None, *, corpus: Corpus | None = None) -> dict:
    """Estimate how many more documents fit given this device's disk.

    The local store is lineage-only — a few hundred bytes per chunk, no text,
    no vectors — so the binding constraint is disk. We measure the average
    on-disk cost of an already-ingested document and divide the usable free
    disk (minus a reserve); before anything is ingested it falls back to a
    nominal per-doc size so a fresh install still shows a sensible number.
    """
    s = settings or load_settings()
    stats = corpus_stats(s, corpus=corpus)
    totals = stats["totals"]
    n_docs = totals["documents"]
    n_chunks = totals["chunks"]

    corpus_path = os.path.join(s.index_dir, "corpus.jsonl")
    corpus_bytes = os.path.getsize(corpus_path) if os.path.exists(corpus_path) else 0

    measured = n_docs > 0 and corpus_bytes > 0
    per_doc = (corpus_bytes / n_docs) if measured else float(_NOMINAL_BYTES_PER_DOC)
    per_doc = max(per_doc, 1024.0)  # floor: guard against absurd estimates

    dev = _device_specs(s.index_dir)
    usable_disk = max(0, dev["disk_free_bytes"] - _DISK_RESERVE_BYTES)
    disk_bound = int(usable_disk // per_doc)

    return {
        "device": dev,
        "corpus_bytes": corpus_bytes,
        "documents_ingested": n_docs,
        "chunks_indexed": n_chunks,
        "bytes_per_document": round(per_doc),
        "bytes_per_document_measured": measured,
        "reserve_bytes": _DISK_RESERVE_BYTES,
        "disk_bound_documents_est": disk_bound,
        "binding_constraint": "disk",
        "remaining_documents_est": disk_bound,
        "basis": (
            "disk-bound. (free disk − 1 GiB reserve) ÷ avg on-disk bytes/doc; "
            "the local store is lineage-only — text and vectors live engine-side. "
            "Per-doc costs are "
            + ("measured from ingested documents." if measured else "a nominal estimate.")
        ),
    }


def config_view(settings: Settings | None = None) -> dict:
    """Runtime configuration for the Observability panel, secrets redacted."""
    s = settings or load_settings()
    return {
        "profile": s.profile.value,
        "engine": "mettle",
        "backend_url": s.backend_url,
        "chat_model": s.chat_model,
        "embed_model": s.embed_model,
        "rerank_model": s.rerank_model,
        "api_keys": {"engine": bool(s.engine_key)},
        "retrieval": {
            "chunk_words": s.chunk_words,
            "chunk_overlap": s.chunk_overlap,
            "retrieval_mode": s.retrieval_mode,
            "engine_collection": s.engine_collection,
            "retrieval_k": s.retrieval_k,
            "context_chunks": s.context_chunks,
            "max_hops": s.max_hops,
        },
        "supervision": {
            "engine_binary": s.engine_binary,
            "mettle_config": s.mettle_config,
        },
        "request_timeout_s": s.request_timeout_s,
        "index_dir": s.index_dir,
    }
