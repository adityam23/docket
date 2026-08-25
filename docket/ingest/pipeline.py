"""Lite ingest pipeline: PDF folder → pages → chunks → (optional) embeddings →
persisted Corpus. One place wires the stages together so the CLI and web share
exactly one ingest path (CLAUDE.md reusability).

Embeddings are optional: when ``DK_EMBED_URL`` is unset the corpus is built with
sparse-only retrieval in mind (no vectors) and retrieval degrades to BM25 — the
pipeline still runs end to end with zero extra services.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from ..config import Settings, load_settings
from .chunk import Chunk, chunk_pages
from .embed import embed_texts, embeddings_available
from .index import Corpus
from .ocr import pdf_to_pages

# Progress hook: on_event(doc_id, stage, info). Stages are the pipeline steps a
# document moves through — the web ingest-job runner subscribes to render live
# per-document progress; passing None (the default) keeps the CLI path silent.
StageEvent = Callable[[str, str, dict], None]


def _doc_id(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def ingest_paths(
    paths: list[str],
    *,
    settings: Settings | None = None,
    corpus: Corpus | None = None,
    on_event: Optional[StageEvent] = None,
    source_of: Optional[Callable[[str], str]] = None,
    skip_existing: bool = True,
) -> Corpus:
    """Ingest specific PDF files into (a new or given) corpus and return it.

    ``on_event`` (optional) is called as the single document walks the stages
    ``ocr → chunk → embed → index → done`` so callers can surface live progress
    without re-implementing the pipeline (CLAUDE.md: one orchestration path).

    ``source_of`` (optional) maps a read-from path to the citation ``source``
    stored on each chunk. It lets the upload path read from a throwaway temp file
    while recording the user's original filename — so temp paths never leak into
    the persisted index. Defaults to the path itself.

    ``skip_existing`` (default True): a document whose id is already in the corpus
    is skipped WITHOUT re-running OCR/embed (the expensive stages). ``Corpus.add``
    was already chunk-idempotent, but only after that work was needlessly redone.
    Identity is the filename stem (``_doc_id``); to intentionally refresh a
    document, remove it first (``remove_document``) then ingest, or pass
    ``skip_existing=False`` to force a re-run.
    """
    s = settings or load_settings()
    # NB: `corpus or Corpus()` would discard a passed-in *empty* corpus (len 0 is
    # falsy), silently dropping incremental ingests into a fresh index. Use an
    # explicit None check.
    c = corpus if corpus is not None else Corpus()
    want_vectors = embeddings_available(s)
    # Doc-level dedup: never re-ingest an already-indexed document (nor the same
    # doc_id twice within one batch). Seeded from the corpus, grows as we add.
    seen: set[str] = set(c.doc_ids())

    def emit(doc_id: str, stage: str, **info: object) -> None:
        if on_event is not None:
            on_event(doc_id, stage, dict(info))

    for path in paths:
        doc_id = _doc_id(path)
        if skip_existing and doc_id in seen:
            emit(doc_id, "skipped", reason="already indexed")
            continue
        emit(doc_id, "ocr", source=path)
        pages = pdf_to_pages(path)

        emit(doc_id, "chunk", pages=len(pages))
        chunks: list[Chunk] = chunk_pages(
            pages,
            doc_id=doc_id,
            source=source_of(path) if source_of else path,
            words=s.chunk_words,
            overlap=s.chunk_overlap,
        )
        if not chunks:
            emit(doc_id, "skipped", reason="no extractable text")
            continue

        vectors = None
        if want_vectors:
            emit(doc_id, "embed", chunks=len(chunks))
            vectors = embed_texts([ch.text for ch in chunks], settings=s)

        emit(doc_id, "index", chunks=len(chunks))
        c.add(chunks, vectors)
        seen.add(doc_id)
        emit(doc_id, "done", chunks=len(chunks), vectorized=bool(vectors))
    return c


def ingest_folder(folder: str, *, settings: Settings | None = None) -> Corpus:
    """Ingest every ``*.pdf`` under a folder into a persisted corpus.

    Persists to ``settings.index_dir`` and returns the corpus.
    """
    s = settings or load_settings()
    pdfs = sorted(
        os.path.join(root, f)
        for root, _, files in os.walk(folder)
        for f in files
        if f.lower().endswith(".pdf")
    )
    corpus = Corpus.load(s.index_dir)  # incremental: extend an existing index
    ingest_paths(pdfs, settings=s, corpus=corpus)
    corpus.save(s.index_dir)
    return corpus
