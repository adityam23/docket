"""Bundled, opt-in sample corpus.

Nothing auto-loads (the index starts empty until the user acts).
``load_samples`` is invoked only on an explicit user action — the CLI ``dk
samples`` command or the dashboard "Load sample documents" button — and adds a
few *synthetic* documents so a new user can try retrieval/QA without exposing any
real data. Sample doc ids are prefixed ``sample-`` so they are easy to spot and
remove via the normal document controls.

Reuses the shared chunk → POST → persist tail (``pipeline._ingest_doc``,
CLAUDE.md: one ingest path); only the PDF/OCR front is skipped because the
content ships as plain text. Doc-level dedup mirrors ``ingest_paths``: a
sample doc_id already in the corpus is skipped without re-POSTing its chunks —
previously every "Load samples" click re-POSTed every chunk, 409ing against
the engine on the second click."""

from __future__ import annotations

import json
from importlib import resources
from typing import Optional

from ..config import Settings, load_settings
from .index import Corpus
from .pipeline import StageEvent, _emitter, _ingest_doc, _ready_provider


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
    """Add the bundled synthetic sample documents to the corpus.

    Idempotent: a sample doc_id already in the corpus is skipped outright (no
    re-chunk, no re-POST), so re-loading the samples is a no-op — like
    ``ingest_paths``'s doc-level dedup. Chunks are POSTed to the engine like
    every other document (engine unreachable → loud failure, like every
    engine feature)."""
    s = settings or load_settings()
    c = corpus if corpus is not None else Corpus.load(s.index_dir)
    seen: set[str] = set(c.doc_ids())
    emit = _emitter(on_event)
    provider = _ready_provider(s)  # one provider + one ensure_collection per run

    for doc in _sample_docs():
        doc_id = doc["doc_id"]
        if doc_id in seen:
            emit(doc_id, "skipped", reason="already indexed")
            continue
        source = f"sample://{doc.get('title', doc_id)}"
        outcome = _ingest_doc(
            doc_id, doc["pages"],
            source=source, meta={}, corpus=c,
            provider=provider, collection=s.engine_collection, settings=s,
            emit=emit, empty_reason="empty sample",
        )
        if outcome == "done":
            seen.add(doc_id)

    return c
