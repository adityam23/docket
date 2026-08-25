"""Bundled, opt-in sample corpus.

Nothing auto-loads (the index starts empty until the user acts).
``load_samples`` is invoked only on an explicit user action — the CLI ``dk
samples`` command or the dashboard "Load sample documents" button — and adds a
few *synthetic* documents so a new user can try retrieval/QA without exposing any
real data. Sample doc ids are prefixed ``sample-`` so they are easy to spot and
remove via the normal document controls.

Reuses the shared chunk → embed → Corpus.add tail (CLAUDE.md: one ingest path);
only the PDF/OCR front is skipped because the content ships as plain text.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Optional

from ..config import Settings, load_settings
from .chunk import Chunk, chunk_pages
from .embed import embed_texts, embeddings_available
from .index import Corpus
from .pipeline import StageEvent

SAMPLE_PREFIX = "sample-"


def _sample_docs() -> list[dict]:
    raw = resources.files("docket.ingest").joinpath("sample_docs.json").read_text()
    return json.loads(raw)


def sample_doc_ids() -> list[str]:
    return [d["doc_id"] for d in _sample_docs()]


def load_samples(
    *,
    settings: Settings | None = None,
    corpus: Corpus | None = None,
    on_event: Optional[StageEvent] = None,
) -> Corpus:
    """Add the bundled synthetic sample documents to the corpus and persist.

    Idempotent: ``Corpus.add`` skips chunks already present by id, so re-loading
    the samples is a no-op. Embeddings are computed only when a dedicated endpoint
    is configured (otherwise the samples are sparse-only, like any other doc).
    """
    s = settings or load_settings()
    c = corpus if corpus is not None else Corpus.load(s.index_dir)
    want_vectors = embeddings_available(s)

    def emit(doc_id: str, stage: str, **info: object) -> None:
        if on_event is not None:
            on_event(doc_id, stage, dict(info))

    for doc in _sample_docs():
        doc_id = doc["doc_id"]
        source = f"sample://{doc.get('title', doc_id)}"
        emit(doc_id, "chunk", pages=len(doc["pages"]))
        chunks: list[Chunk] = chunk_pages(
            doc["pages"],
            doc_id=doc_id,
            source=source,
            words=s.chunk_words,
            overlap=s.chunk_overlap,
        )
        if not chunks:
            emit(doc_id, "skipped", reason="empty sample")
            continue
        vectors = None
        if want_vectors:
            emit(doc_id, "embed", chunks=len(chunks))
            vectors = embed_texts([ch.text for ch in chunks], settings=s)
        emit(doc_id, "index", chunks=len(chunks))
        c.add(chunks, vectors)
        emit(doc_id, "done", chunks=len(chunks), vectorized=bool(vectors))

    c.save(s.index_dir)
    return c
