"""Retrieval over the engine. Docket implements NO retrieval itself: dense,
keyword and fused search are mettle's ``/v1/collections/{c}/search`` modes
(RRF-union hybrid lives engine-side, cutover contract §1–2, 2026-09-10). What
stays client-side is orchestration only:

- **Lineage join** — engine hits carry text + score but not citation metadata;
  docket joins them to its lineage store on ``document_id`` (= docket chunk_id).
- **Rerank** — optional cross-encoder reorder of the engine's pool via the ONE
  provider's ``/v1/rerank`` (batching is engine-side).
- **Iterative** — ``iterative_retrieve`` merges de-duplicated hits across
  several queries (the agent's re-queries; multi-hop needle recall).

Every engine failure is loud — there is no local fallback to degrade to."""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

from ..ingest.chunk import FILING_FIELDS
from ..ingest.index import Corpus

log = logging.getLogger(__name__)

# (query, docs) -> one score per document in the input order (higher = better).
Reranker = Callable[[str, Sequence[str]], list[float]]

# Default candidate pool pulled before rerank, when the caller doesn't wire
# Settings.retrieval_k through (e.g. a bare Retriever in a test or script).
DEFAULT_CANDIDATES = 40


class Retriever:
    def __init__(
        self,
        provider,
        lineage: Corpus,
        *,
        collection: str,
        mode: str = "hybrid",
        reranker: Optional[Reranker] = None,
        scope_doc: Optional[str] = None,
        candidates: int = DEFAULT_CANDIDATES,
    ) -> None:
        if mode not in ("dense", "keyword", "hybrid"):
            raise ValueError(f"unknown search mode: {mode!r}")
        self.provider = provider
        self.lineage = lineage
        self.collection = collection
        self.mode = mode
        self.reranker = reranker
        # Optional per-document scope: restrict engine search to one filing's
        # chunks via the search allowlist (document ids = docket chunk ids).
        self.scope_doc = scope_doc
        # Candidate pool pulled before rerank — Settings.retrieval_k when wired
        # through load_retriever, this default otherwise (bare Retriever in
        # tests/scripts).
        self.candidates = candidates

    def retrieve(self, query: str, *, k: int = 10, candidates: int | None = None) -> list[dict]:
        """Return up to ``k`` ranked hits: dicts with lineage + text + score +
        rank. Text comes from the engine; citation fields from docket's lineage
        (joined on ``document_id``), empty (never invented) when unknown.

        Hits whose ``chunk_id`` has no lineage row are dropped — an orphan
        (old delete bug, a mid-batch ingest failure, a benchmark run into the
        prod collection) has no doc_id/page to cite, so ``Citation.from_hit``
        would either invent one or raise. One warning is logged with the
        dropped count rather than one log line per hit.
        """
        if not self.lineage.chunks:
            return []
        if self.scope_doc:
            allowlist = [ch.chunk_id for ch in self.lineage.chunks if ch.doc_id == self.scope_doc]
            if not allowlist:
                # The scope resolved to zero chunks. An empty allowlist reads
                # as "no scope" to the real provider (and FakeEngine, which
                # mirrors it) — falling through would silently search the
                # WHOLE collection instead of the (empty) requested scope.
                return []
        else:
            allowlist = None
        rows = self.provider.collection_search(
            self.collection, query, mode=self.mode, k=candidates if candidates is not None else self.candidates,
            allowlist=allowlist,
        )
        # Drop orphans BEFORE rerank: cross-encoder time is the dominant cost
        # (~0.5 ms/token on the reference card), never spend it on a hit we
        # can't cite.
        lineage = {row["document_id"]: self.lineage.find(row["document_id"]) for row in rows}
        kept = [row for row in rows if lineage[row["document_id"]] is not None]
        dropped = len(rows) - len(kept)
        if dropped:
            log.warning(
                "retrieve(%r): dropped %d engine hit(s) with no lineage row (orphaned chunk_id)",
                query, dropped,
            )
        rows = kept
        if not rows:
            return []

        # ``score`` must reflect the ranking that produced the final order: the
        # engine's score when no reranker runs, or the cross-encoder's score
        # once it reorders — otherwise a downstream read of ``score``
        # misrepresents rerank's effect (score would contradict the order).
        if self.reranker:
            rr = self.reranker(query, [row["text"] for row in rows])
            rows = [
                dict(row, score=float(s))
                for row, s in sorted(zip(rows, rr), key=lambda t: t[1], reverse=True)
            ]

        hits = []
        for row in rows[:k]:
            lin = lineage[row["document_id"]]
            hits.append(
                {
                    "chunk_id": row["document_id"],
                    "doc_id": lin.doc_id,
                    "engine_chunk_id": row["chunk_id"],
                    "page": lin.page,
                    "text": row["text"],
                    "source": lin.source,
                    **{f: getattr(lin, f) for f in FILING_FIELDS},
                    "score": round(float(row["score"]), 6),
                }
            )
        for rank, hit in enumerate(hits):
            hit["rank"] = rank
        return hits

    def iterative_retrieve(self, queries: Sequence[str], *, k: int = 10) -> list[dict]:
        """Merge de-duplicated hits across several queries (multi-hop recall).

        Best rank across queries wins; results are re-sorted by that rank. This
        is the retrieval half of the agent's iterative re-query loop (Q8)."""
        best: dict[str, dict] = {}
        for q in queries:
            for hit in self.retrieve(q, k=k):
                cur = best.get(hit["chunk_id"])
                if cur is None or hit["rank"] < cur["rank"]:
                    best[hit["chunk_id"]] = hit
        merged = sorted(best.values(), key=lambda h: h["rank"])
        return merged[:k]
