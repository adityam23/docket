"""Read models for the web dashboard: turn a persisted ``Corpus`` and the live
``Settings`` into JSON the frontend can render (corpus stats, per-document ingest
stage, embedding state, redacted config).

This is the ONE place those views are computed (CLAUDE.md reusability): the web
API is a thin transport over these functions, so the CLI or eval harness can
reuse them without duplicating the derivation.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

from ..config import Settings, load_settings
from ..ingest.embed import embeddings_available
from ..ingest.index import Corpus

# Keep this much disk free — never let ingestion fill the device (docs/decisions
# Q13: honest about scale). Also a nominal per-document size for the very first
# estimate, before any real document has been ingested to measure against.
_DISK_RESERVE_BYTES = 1 * 1024**3        # 1 GiB headroom
_NOMINAL_BYTES_PER_DOC = 80_000          # ~a dozen chunks incl. 768-dim vectors

# Retrieval is pure-Python and in-memory: ``Corpus.load`` reads every chunk +
# vector into process RAM and brute-force cosine-scans them, so RAM is a real
# second bound as the corpus grows (VRAM is NOT — embeddings are precomputed by
# the backend and stored; the model's VRAM is fixed regardless of corpus size).
# We estimate the in-RAM footprint and take the tighter of disk/RAM.
_RAM_USABLE_FRACTION = 0.5               # corpus may claim at most half of RAM
_RAM_BYTES_PER_FLOAT = 24                # a boxed float in a Python list (approx)
_RAM_BYTES_PER_CHUNK = 512               # Chunk object + str/dict overhead (approx)
_NOMINAL_RAM_BYTES_PER_DOC = 300_000     # first-estimate in-RAM cost of a doc


def _ram_footprint_bytes(chars: int, chunks: int, vectorized_chunks: int, dim: int) -> int:
    """Rough in-RAM cost of a loaded corpus: text + per-chunk object overhead +
    boxed float vectors. Deliberately generous on floats (Python lists box each
    element) so we do not overstate capacity."""
    return (
        chars
        + chunks * _RAM_BYTES_PER_CHUNK
        + vectorized_chunks * max(dim, 0) * _RAM_BYTES_PER_FLOAT
    )


@dataclass
class DocView:
    doc_id: str
    source: str
    chunks: int
    pages: int          # distinct source pages covered
    page_min: int
    page_max: int
    vectorized: int     # chunks that carry a dense vector
    coverage: float     # vectorized / chunks  (embedding coverage, 0..1)
    stage: str          # queued | indexed | embedding | embedded
    chars: int          # total text characters (rough corpus weight)


def _doc_stage(chunks: int, vectorized: int, dense_on: bool) -> str:
    """Derive a document's terminal ingest stage from what landed in the corpus.

    Live in-flight stages (ocr/chunking/embedding) come from the ingest job
    registry; this is the resting state once chunks are persisted.
    """
    if chunks == 0:
        return "queued"
    if not dense_on or vectorized == 0:
        return "indexed"          # sparse-only (BM25) — no dense vectors
    if vectorized < chunks:
        return "embedding"        # partial dense coverage
    return "embedded"             # full hybrid dense+sparse


def corpus_stats(settings: Settings | None = None, *, corpus: Corpus | None = None) -> dict:
    """Aggregate corpus + per-document stats for the dashboard."""
    s = settings or load_settings()
    c = corpus if corpus is not None else Corpus.load(s.index_dir)
    dense_on = embeddings_available(s)

    # Fold chunks into per-document accumulators in one pass.
    docs: dict[str, dict] = {}
    embed_dim = 0
    total_vectorized = 0
    for ch, vec in zip(c.chunks, c.vectors):
        d = docs.setdefault(
            ch.doc_id,
            {"source": ch.source, "pages": set(), "chunks": 0, "vectorized": 0, "chars": 0},
        )
        d["chunks"] += 1
        d["chars"] += len(ch.text)
        d["pages"].add(ch.page)
        if vec is not None:
            d["vectorized"] += 1
            total_vectorized += 1
            if not embed_dim:
                embed_dim = len(vec)

    views: list[DocView] = []
    for doc_id, d in sorted(docs.items()):
        pages = d["pages"]
        chunks = d["chunks"]
        vectorized = d["vectorized"]
        views.append(
            DocView(
                doc_id=doc_id,
                source=d["source"],
                chunks=chunks,
                pages=len(pages),
                page_min=min(pages) if pages else 0,
                page_max=max(pages) if pages else 0,
                vectorized=vectorized,
                coverage=round(vectorized / chunks, 4) if chunks else 0.0,
                stage=_doc_stage(chunks, vectorized, dense_on),
                chars=d["chars"],
            )
        )

    total_chunks = len(c.chunks)
    return {
        "documents": [v.__dict__ for v in views],
        "totals": {
            "documents": len(views),
            "chunks": total_chunks,
            "chars": sum(v.chars for v in views),
            "vectorized_chunks": total_vectorized,
            "embedding_coverage": round(total_vectorized / total_chunks, 4) if total_chunks else 0.0,
        },
        "embeddings": {
            "enabled": dense_on,
            "dim": embed_dim,
            "model": s.embed_model if dense_on else None,
            "endpoint": s.embed_url if dense_on else None,
            "retrieval_mode": "hybrid (dense + BM25)" if (dense_on and total_vectorized) else "sparse-only (BM25)",
        },
        "index_dir": s.index_dir,
    }


def _existing_dir(path: str) -> str:
    """Nearest existing ancestor of ``path`` — ``disk_usage`` needs a real path,
    and ``index_dir`` may not exist until the first ingest."""
    p = os.path.abspath(path)
    while p and not os.path.isdir(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return p or "."


def _device_specs(index_dir: str) -> dict:
    """Disk (where the index lives) + total RAM, dependency-free and best-effort.

    Both bound how much a *lite*, in-memory, single-file corpus can hold: the
    JSONL grows on disk, and retrieval loads chunks+vectors into RAM.
    """
    du = shutil.disk_usage(_existing_dir(index_dir))
    ram_total: int | None = None
    try:  # Linux/POSIX; absent on some platforms — degrade to None, not crash.
        ram_total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        ram_total = None
    return {
        "disk_free_bytes": du.free,
        "disk_total_bytes": du.total,
        "disk_used_bytes": du.used,
        "ram_total_bytes": ram_total,
    }


def capacity_stats(settings: Settings | None = None, *, corpus: Corpus | None = None) -> dict:
    """Estimate how many more documents can be ingested given this device.

    Two bounds, and we report the *tighter* of them:

    - **Disk-bound**: the persisted corpus (chunks + optional dense vectors) is a
      single JSONL under ``index_dir``. We measure the average on-disk cost of an
      already-ingested document and divide the usable free disk (minus a reserve).
    - **RAM-bound**: retrieval loads the whole corpus into process memory
      (:func:`_ram_footprint_bytes`), so beyond some size it stops fitting in RAM
      regardless of free disk. VRAM does *not* scale with the corpus.

    Before anything is ingested both fall back to nominal per-doc sizes, so a
    fresh install still shows a sensible number. Reuses ``corpus_stats`` for the
    document/chunk counts (CLAUDE.md: one derivation).
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

    # --- disk bound ---
    usable_disk = max(0, dev["disk_free_bytes"] - _DISK_RESERVE_BYTES)
    disk_bound = int(usable_disk // per_doc)

    # --- RAM bound (in-memory corpus) ---
    ram_footprint = _ram_footprint_bytes(
        totals["chars"], n_chunks, totals["vectorized_chunks"], stats["embeddings"]["dim"]
    )
    ram_per_doc = (ram_footprint / n_docs) if measured else float(_NOMINAL_RAM_BYTES_PER_DOC)
    ram_per_doc = max(ram_per_doc, 1024.0)
    ram_total = dev["ram_total_bytes"]
    ram_bound: int | None = None
    if ram_total:
        usable_ram = max(0, int(ram_total * _RAM_USABLE_FRACTION) - ram_footprint)
        ram_bound = int(usable_ram // ram_per_doc)

    # Report the binding constraint (the tighter of the two).
    if ram_bound is not None and ram_bound < disk_bound:
        remaining_docs, binding = ram_bound, "ram"
    else:
        remaining_docs, binding = disk_bound, "disk"

    return {
        "device": dev,
        "corpus_bytes": corpus_bytes,
        "documents_ingested": n_docs,
        "chunks_indexed": n_chunks,
        "bytes_per_document": round(per_doc),
        "bytes_per_document_measured": measured,
        "reserve_bytes": _DISK_RESERVE_BYTES,
        "ram_bytes_per_document": round(ram_per_doc),
        "ram_footprint_bytes": ram_footprint,
        "disk_bound_documents_est": disk_bound,
        "ram_bound_documents_est": ram_bound,
        "binding_constraint": binding,
        "remaining_documents_est": remaining_docs,
        "basis": (
            f"{binding}-bound (the tighter of disk and RAM). Disk: (free disk − "
            "1 GiB reserve) ÷ avg on-disk bytes/doc. RAM: (½ total RAM − current "
            "corpus) ÷ avg in-RAM bytes/doc, since retrieval holds the corpus in "
            "memory. Per-doc costs are "
            + ("measured from ingested documents." if measured else "nominal estimates.")
        ),
    }


def config_view(settings: Settings | None = None) -> dict:
    """Runtime configuration for the Observability panel, secrets redacted."""
    s = settings or load_settings()
    return {
        "profile": s.profile.value,
        "provider": s.provider.value,
        "backend_url": s.backend_url,
        "chat_model": s.chat_model,
        "embed": {
            "enabled": embeddings_available(s),
            "url": s.embed_url,
            "model": s.embed_model,
        },
        "api_keys": {
            "cerebras": bool(s.cerebras_api_key),
            "groq": bool(s.groq_api_key),
        },
        "retrieval": {
            "chunk_words": s.chunk_words,
            "chunk_overlap": s.chunk_overlap,
            "retrieval_k": s.retrieval_k,
            "context_chunks": s.context_chunks,
            "max_hops": s.max_hops,
        },
        "request_timeout_s": s.request_timeout_s,
        "index_dir": s.index_dir,
    }
