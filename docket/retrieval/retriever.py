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

from typing import Callable, Optional, Sequence

from ..ingest.index import Corpus

# (query, docs) -> one score per document in the input order (higher = better).
Reranker = Callable[[str, Sequence[str]], list[float]]


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

    def retrieve(self, query: str, *, k: int = 10, candidates: int = 40) -> list[dict]:
        """Return up to ``k`` ranked hits: dicts with lineage + text + score +
        rank. Text comes from the engine; citation fields from docket's lineage
        (joined on ``document_id``), empty (never invented) when unknown."""
        if not self.lineage.chunks:
            return []
        allowlist = (
            [ch.chunk_id for ch in self.lineage.chunks if ch.doc_id == self.scope_doc]
            if self.scope_doc
            else None
        )
        rows = self.provider.collection_search(
            self.collection, query, mode=self.mode, k=candidates, allowlist=allowlist
        )
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
        for rank, row in enumerate(rows[:k]):
            lin = self.lineage._by_id.get(row["document_id"])
            lin = self.lineage.chunks[lin] if lin is not None else None
            hits.append(
                {
                    "chunk_id": row["document_id"],
                    "doc_id": lin.doc_id if lin else row["document_id"].split("#")[0],
                    "engine_chunk_id": row["chunk_id"],
                    "page": lin.page if lin else None,
                    "text": row["text"],
                    "source": lin.source if lin else "",
                    "form_type": lin.form_type if lin else "",
                    "company": lin.company if lin else "",
                    "ticker": lin.ticker if lin else "",
                    "fiscal_period": lin.fiscal_period if lin else "",
                    "score": round(float(row["score"]), 6),
                    "rank": rank,
                }
            )
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
