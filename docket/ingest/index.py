"""Corpus store — docket's OWN slice post-cutover: chunk lineage for citations,
doc-level dedup, and ingest state. Chunk TEXT is not stored; display text comes
back from mettle search hits, joined on the engine document id (= docket
chunk_id). Dense vectors are the engine's (turbovec); local brute-force cosine
is gone with the local retrieval legs (cutover contract §1, 2026-09-10).

One concept, one implementation: this is the *only* corpus store; the retriever
joins engine hits through it."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

from .chunk import FILING_FIELDS


@dataclass(frozen=True)
class Lineage:
    """Where a chunk came from — the citation surface. ``chars`` preserves the
    chunk size for telemetry once the text itself is dropped (engine hits carry
    it back). ``chunk_id`` doubles as the engine document id."""

    chunk_id: str           # stable, unique: f"{doc_id}#{n}"; = engine doc id
    doc_id: str
    page: int               # 1-based source page
    source: str = ""        # original file path
    form_type: str = ""     # e.g. "10-K", "10-Q", "8-K"
    company: str = ""       # registrant name
    ticker: str = ""        # e.g. "MMM"
    fiscal_period: str = "" # e.g. "2018", "FY2022", "Q3 2021"
    chars: int = 0


def _lineage_of(chunk) -> Lineage:
    return Lineage(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        page=chunk.page,
        source=chunk.source,
        chars=len(chunk.text),
        **{f: getattr(chunk, f) for f in FILING_FIELDS},
    )


class Corpus:
    """Chunk-lineage rows addressable by ``chunk_id``, dedupable by ``doc_id``."""

    def __init__(self) -> None:
        self.chunks: list[Lineage] = []
        self._by_id: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks) -> None:
        """Append lineage rows for chunks. Idempotent per chunk_id (a re-added
        chunk replaces nothing — callers dedup at doc level before adding)."""
        for ch in chunks:
            if ch.chunk_id in self._by_id:
                continue
            row = ch if isinstance(ch, Lineage) else _lineage_of(ch)
            self._by_id[row.chunk_id] = len(self.chunks)
            self.chunks.append(row)

    def find(self, chunk_id: str) -> Lineage | None:
        """Look up a chunk's lineage, or ``None`` when the id has no row (an
        orphaned engine hit — old delete bug, mid-batch ingest failure, a
        benchmark run that wrote into the prod collection). Callers must treat
        ``None`` as "can't cite this", never invent a row for it."""
        i = self._by_id.get(chunk_id)
        return self.chunks[i] if i is not None else None

    def doc_ids(self) -> list[str]:
        """Distinct document ids in stable first-seen order."""
        seen: dict[str, None] = {}
        for ch in self.chunks:
            seen.setdefault(ch.doc_id, None)
        return list(seen)

    def remove(self, doc_id: str) -> int:
        """Drop every lineage row belonging to ``doc_id``. Returns rows removed
        (0 if the id was not present)."""
        kept = [ch for ch in self.chunks if ch.doc_id != doc_id]
        removed = len(self.chunks) - len(kept)
        if removed:
            self.chunks = kept
            self._by_id = {ch.chunk_id: i for i, ch in enumerate(self.chunks)}
        return removed

    def save(self, path: str) -> None:
        """Persist the lineage as JSONL under ``path``."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "corpus.jsonl"), "w", encoding="utf-8") as f:
            for row in self.chunks:
                f.write(json.dumps(asdict(row)) + "\n")

    @classmethod
    def load(cls, path: str) -> "Corpus":
        """Load the lineage persisted under ``path`` (see :meth:`save`)."""
        c = cls()
        fp = os.path.join(path, "corpus.jsonl")
        if os.path.exists(fp):
            with open(fp, encoding="utf-8") as f:
                # Tolerate rows carrying fields Lineage doesn't know (older or
                # newer stores): drop the extras, keep the lineage. Text is
                # never stored.
                fields = set(Lineage.__dataclass_fields__)
                rows = []
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rows.append(Lineage(**{k: v for k, v in row.items() if k in fields}))
                c.add(rows)
        return c
