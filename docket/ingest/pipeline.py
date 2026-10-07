"""Ingest pipeline: PDF folder → pages → chunks → engine documents → persisted
lineage. One place wires the stages together so the CLI and web share exactly
one ingest path (CLAUDE.md reusability).

Post-cutover, docket owns ONLY the chunker and the citation lineage: chunks are
POSTed to the engine's collection (one engine document per docket chunk, id =
chunk_id, metadata = filing lineage) and the local store keeps lineage rows
only — text, embeddings and the index are the engine's (cutover contract §1–2,
2026-09-10). Engine unreachable → the pipeline fails loudly; there is no local
fallback.

Per-document atomicity (B4, contract D3): each document's lineage is persisted
to disk the moment ALL of its chunks are confirmed posted — never batched up
and saved only at the very end. If a document's POST fails partway through
(a dropped model slot, any other HTTP error), that document is recorded
``"failed"`` with the reason, no lineage is written for it, its already-posted
chunks are best-effort deleted (so a retry doesn't 409 on them), and the batch
moves on to the next document rather than aborting — see ``_ingest_doc``.
"""

from __future__ import annotations

import os
import threading
from typing import Callable, Optional

from ..config import Settings, load_settings
from ..providers.base import Provider
from ..providers.router import get_provider
from .chunk import FILING_FIELDS, Chunk, chunk_pages
from .index import Corpus
from .metadata import extract_filing_metadata
from .ocr import pdf_to_pages

# Progress hook: on_event(doc_id, stage, info). Stages are the pipeline steps a
# document moves through — the web ingest-job runner subscribes to render live
# per-document progress; passing None (the default) keeps the CLI path silent.
StageEvent = Callable[[str, str, dict], None]

# Serialises every corpus-mutating operation across the whole process: an
# ingest job's load→(post…)→save run, a sample load, and a document removal
# must never interleave their ``Corpus.load``/``save`` pairs, or the slower
# writer's save silently clobbers the other's (B4). Every entry point below —
# and the web layer's ``_require_idle`` — acquires this NON-BLOCKING, so a
# caller that finds the corpus busy fails fast with a clear "try again" rather
# than queuing behind what might be a long-running ingest.
CORPUS_LOCK = threading.Lock()


def _noop() -> None:
    return None


def doc_id_of(path: str) -> str:
    """A document's identity: the filename stem (known limitation — two
    different folders' same-named file collide; callers that admit duplicate
    basenames, e.g. uploads, disambiguate the on-disk name before this runs)."""
    return os.path.splitext(os.path.basename(path))[0]


def discover_pdfs(folder: str) -> list[str]:
    """Every ``*.pdf`` under ``folder``, sorted for deterministic ingest order.
    The one PDF-discovery walk shared by ``ingest_folder`` and the web job
    runner (``jobs.IngestJobs.start``) — previously duplicated between them."""
    return sorted(
        os.path.join(root, f)
        for root, _, files in os.walk(folder)
        for f in files
        if f.lower().endswith(".pdf")
    )


def _ready_provider(settings: Settings) -> Provider:
    """Build the one provider for this ingest run and ensure its collection
    exists — ONCE per run, not once per document (the prior version paid for a
    fresh httpx client plus an extra POST on every single document). Looked up
    via this module's own ``get_provider`` name — not re-exported/re-imported
    by callers — so tests can monkeypatch ``pipeline.get_provider`` and have it
    take effect no matter which module ends up calling this."""
    provider = get_provider(settings)
    provider.ensure_collection(settings.engine_collection)
    return provider


def _emitter(on_event: Optional[StageEvent]) -> Callable[..., None]:
    """The one ``emit(doc_id, stage, **info)`` shape over an optional
    ``on_event`` sink (``None`` on the silent CLI path) — previously hand-rolled
    as the same three-line closure in both ``ingest_paths`` and
    ``load_samples`` (CLAUDE.md 'Reusability')."""

    def emit(doc_id: str, stage: str, **info: object) -> None:
        if on_event is not None:
            on_event(doc_id, stage, dict(info))

    return emit


def _ingest_doc(
    doc_id: str,
    pages: list[dict],
    *,
    source: str,
    meta: dict,
    corpus: Corpus,
    provider: Provider,
    collection: str,
    settings: Settings,
    emit: Callable[..., None],
    empty_reason: str = "no extractable text",
) -> str:
    """Chunk, POST and persist ONE document — the per-document tail shared by
    ``ingest_paths`` (PDF front, pages come from ``pdf_to_pages``) and
    ``load_samples`` (plain-text front, pages ship bundled): chunk →
    skip-if-empty → POST every chunk → persist lineage → done. One
    implementation instead of two near-identical copies (CLAUDE.md).

    Lineage is saved to ``settings.index_dir`` immediately after this
    document's chunks are ALL confirmed posted (contract D3): a mid-batch
    failure on a LATER document can never leave an EARLIER document's
    already-posted chunks without a matching lineage row.

    If any chunk's POST raises (``SlotUnavailable``/503, or any other HTTP
    error — mettle writes nothing engine-side for a failed POST), this
    document is marked ``"failed"`` with the reason, no lineage is written for
    it, and every chunk of THIS document already posted is best-effort
    DELETEd (so a retry doesn't 409 on them). The caller is expected to
    continue with the next document rather than aborting the whole batch.

    Returns the outcome: ``"done"``, ``"skipped"``, or ``"failed"``.
    """
    emit(doc_id, "chunk", pages=len(pages))
    chunks: list[Chunk] = chunk_pages(
        pages,
        doc_id=doc_id,
        source=source,
        words=settings.chunk_words,
        overlap=settings.chunk_overlap,
        **meta,
    )
    if not chunks:
        emit(doc_id, "skipped", reason=empty_reason)
        return "skipped"

    emit(doc_id, "post", chunks=len(chunks))
    posted: list[str] = []
    try:
        for ch in chunks:
            provider.post_document(
                collection,
                ch.chunk_id,
                ch.text,
                {"page": str(ch.page), **{f: getattr(ch, f) for f in FILING_FIELDS}},
            )
            posted.append(ch.chunk_id)
    except Exception as e:  # noqa: BLE001 - one bad document must not sink the batch (D3)
        for chunk_id in posted:
            try:
                provider.delete_document(collection, chunk_id)
            except Exception:  # noqa: BLE001 - best-effort cleanup; a retry re-POSTs anyway
                pass
        emit(doc_id, "failed", reason=str(e))
        return "failed"

    corpus.add(chunks)
    corpus.save(settings.index_dir)
    emit(doc_id, "done", chunks=len(chunks))
    return "done"


def ingest_paths(
    paths: list[str],
    *,
    settings: Settings | None = None,
    corpus: Corpus | None = None,
    on_event: Optional[StageEvent] = None,
    source_of: Optional[Callable[[str], str]] = None,
    skip_existing: bool = True,
    between_documents: Optional[Callable[[], None]] = None,
) -> Corpus:
    """Ingest specific PDF files into (a new or given) corpus and return it.

    ``on_event`` (optional) is called as the single document walks the stages
    ``ocr → chunk → post → done/skipped/failed`` so callers can surface live
    progress without re-implementing the pipeline (CLAUDE.md: one orchestration
    path). A document whose POST fails is recorded ``failed`` (with no lineage
    written) and the batch continues with the next one — see ``_ingest_doc``.

    ``source_of`` (optional) maps a read-from path to the citation ``source``
    stored on each chunk. It lets the upload path read from a throwaway temp file
    while recording the user's original filename — so temp paths never leak into
    the persisted index. Defaults to the path itself.

    ``skip_existing`` (default True): a document whose id is already in the
    corpus is skipped WITHOUT re-running OCR (the expensive stage) or re-posting
    its chunks to the engine. Identity is the filename stem (``doc_id_of``); to
    intentionally refresh a document, remove it first (``remove_document``) then
    ingest, or pass ``skip_existing=False`` to force a re-run.

    ``between_documents`` (optional) is called once after EVERY document
    finishes — done, skipped, or failed — before the next one starts. It is
    the ONE seam a future ticket can hook to pause a long-running batch between
    documents (e.g. to let a live question preempt ingest and switch the
    engine's profile); this ticket adds only the seam, not any pause behaviour.
    Defaults to a no-op.
    """
    s = settings or load_settings()
    # NB: `corpus or Corpus()` would discard a passed-in *empty* corpus (len 0 is
    # falsy), silently dropping incremental ingests into a fresh index. Use an
    # explicit None check.
    c = corpus if corpus is not None else Corpus()
    # Doc-level dedup: never re-ingest an already-indexed document (nor the same
    # doc_id twice within one batch). Seeded from the corpus, grows as we add.
    seen: set[str] = set(c.doc_ids())
    emit = _emitter(on_event)
    pause = between_documents or _noop
    provider = _ready_provider(s)  # one provider + one ensure_collection per run

    for path in paths:
        doc_id = doc_id_of(path)
        if skip_existing and doc_id in seen:
            emit(doc_id, "skipped", reason="already indexed")
        else:
            emit(doc_id, "ocr", source=path)
            pages = pdf_to_pages(path)
            src = source_of(path) if source_of else path
            meta = extract_filing_metadata(doc_id, src)
            outcome = _ingest_doc(
                doc_id, pages,
                source=src, meta=meta, corpus=c,
                provider=provider, collection=s.engine_collection, settings=s,
                emit=emit,
            )
            if outcome == "done":
                seen.add(doc_id)
        pause()  # the between-documents seam — no-op until a future ticket wires one
    return c


def ingest_into_index(
    paths: list[str],
    *,
    settings: Settings | None = None,
    on_event: Optional[StageEvent] = None,
    source_of: Optional[Callable[[str], str]] = None,
    between_documents: Optional[Callable[[], None]] = None,
) -> Corpus:
    """Load the persisted corpus, ingest ``paths`` into it, and return it — the
    ONE load→ingest path shared by ``ingest_folder`` (CLI) and the web job
    runner (``jobs._run``), which each used to re-implement this triplet
    (CLAUDE.md). No final save is needed here: each document already persisted
    its own lineage as it completed (see ``_ingest_doc``)."""
    s = settings or load_settings()
    corpus = Corpus.load(s.index_dir)
    ingest_paths(
        paths, settings=s, corpus=corpus, on_event=on_event, source_of=source_of,
        between_documents=between_documents,
    )
    return corpus


def ingest_folder(
    folder: str,
    *,
    settings: Settings | None = None,
    on_event: Optional[StageEvent] = None,
) -> Corpus:
    """Ingest every ``*.pdf`` under a folder into the persisted corpus.

    Each document's lineage is saved as it completes (see ``_ingest_doc``); a
    document whose POST fails is skipped (recorded via ``on_event`` if given)
    and the rest of the folder still ingests.
    """
    s = settings or load_settings()
    return ingest_into_index(discover_pdfs(folder), settings=s, on_event=on_event)
