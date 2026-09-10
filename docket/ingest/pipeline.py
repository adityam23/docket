"""Ingest pipeline: PDF folder → pages → chunks → engine documents → persisted
lineage. One place wires the stages together so the CLI and web share exactly
one ingest path (CLAUDE.md reusability).

Post-cutover, docket owns ONLY the chunker and the citation lineage: chunks are
POSTed to the engine's collection (one engine document per docket chunk, id =
chunk_id, metadata = filing lineage) and the local store keeps lineage rows
only — text, embeddings and the index are the engine's (cutover contract §1–2,
2026-09-10). Engine unreachable → the pipeline fails loudly; there is no local
fallback."""

from __future__ import annotations

import os
from typing import Callable, Optional

from ..config import Settings, load_settings
from ..providers.router import get_provider
from .chunk import Chunk, chunk_pages
from .index import Corpus
from .metadata import extract_filing_metadata
from .ocr import pdf_to_pages

# Progress hook: on_event(doc_id, stage, info). Stages are the pipeline steps a
# document moves through — the web ingest-job runner subscribes to render live
# per-document progress; passing None (the default) keeps the CLI path silent.
StageEvent = Callable[[str, str, dict], None]


def _doc_id(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _post_chunks(chunks: list[Chunk], *, settings: Settings) -> None:
    """Ensure the collection exists, then POST each docket chunk to it. Every
    chunk is one engine document with the filing metadata attached, so the
    engine's hits can join back to citations on ``document_id``."""
    provider = get_provider(settings)
    provider.ensure_collection(settings.engine_collection)
    for ch in chunks:
        provider.post_document(
            settings.engine_collection,
            ch.chunk_id,
            ch.text,
            {
                "page": str(ch.page),
                "form_type": ch.form_type,
                "company": ch.company,
                "ticker": ch.ticker,
                "fiscal_period": ch.fiscal_period,
            },
        )


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
    ``ocr → chunk → post → done`` so callers can surface live progress without
    re-implementing the pipeline (CLAUDE.md: one orchestration path).

    ``source_of`` (optional) maps a read-from path to the citation ``source``
    stored on each chunk. It lets the upload path read from a throwaway temp file
    while recording the user's original filename — so temp paths never leak into
    the persisted index. Defaults to the path itself.

    ``skip_existing`` (default True): a document whose id is already in the
    corpus is skipped WITHOUT re-running OCR (the expensive stage) or re-posting
    its chunks to the engine. Identity is the filename stem (``_doc_id``); to
    intentionally refresh a document, remove it first (``remove_document``) then
    ingest, or pass ``skip_existing=False`` to force a re-run.
    """
    s = settings or load_settings()
    # NB: `corpus or Corpus()` would discard a passed-in *empty* corpus (len 0 is
    # falsy), silently dropping incremental ingests into a fresh index. Use an
    # explicit None check.
    c = corpus if corpus is not None else Corpus()
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
        src = source_of(path) if source_of else path
        meta = extract_filing_metadata(doc_id, src)
        chunks: list[Chunk] = chunk_pages(
            pages,
            doc_id=doc_id,
            source=src,
            words=s.chunk_words,
            overlap=s.chunk_overlap,
            **meta,
        )
        if not chunks:
            emit(doc_id, "skipped", reason="no extractable text")
            continue

        emit(doc_id, "post", chunks=len(chunks))
        _post_chunks(chunks, settings=s)
        c.add(chunks)
        seen.add(doc_id)
        emit(doc_id, "done", chunks=len(chunks))
    return c


def ingest_folder(folder: str, *, settings: Settings | None = None) -> Corpus:
    """Ingest every ``*.pdf`` under a folder into a persisted corpus.

    Persists the lineage to ``settings.index_dir`` and returns the corpus.
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
