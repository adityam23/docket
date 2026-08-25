"""Retrieval that lifts a small model to needle-in-haystack parity (the thesis,
docs/decisions Q8): hybrid dense+sparse → rerank → ITERATIVE agentic re-query.

Layers, each degrading gracefully so the base install works offline:
- **Sparse (BM25)** — always on. Uses ``rank_bm25`` when present (the ``ingest``
  extra); otherwise a small, correct pure-Python BM25 so retrieval never
  hard-depends on a package.
- **Dense** — cosine over corpus vectors, active only when an embedding endpoint
  is configured (a ``embed_query`` callable is passed). Absent → sparse-only.
- **Fusion** — Reciprocal Rank Fusion (rank-based, score-scale-free) merges the
  two rankings.
- **Rerank** — optional cross-encoder seam (BGE-reranker-v2 / Qwen3-Reranker);
  when unset the fused order stands.
- **Iterative** — ``iterative_retrieve`` runs multiple queries (the agent's
  re-queries) and merges de-duplicated hits, the mechanism behind multi-hop
  needle recall.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Optional, Sequence

from ..ingest.index import Corpus

_TOKEN = re.compile(r"[a-z0-9]+")

EmbedQuery = Callable[[str], list[float]]
Reranker = Callable[[str, list[str]], list[float]]  # query, docs → scores


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class _PurePythonBM25:
    """Minimal BM25 Okapi — the fallback when ``rank_bm25`` isn't installed."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = corpus_tokens
        self.N = len(corpus_tokens)
        self.avgdl = (sum(len(d) for d in corpus_tokens) / self.N) if self.N else 0.0
        self.tf = [Counter(d) for d in corpus_tokens]
        df: Counter = Counter()
        for d in corpus_tokens:
            for term in set(d):
                df[term] += 1
        # idf with the +1 smoothing that keeps scores non-negative.
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }

    def get_scores(self, query_tokens: Sequence[str]) -> list[float]:
        scores = [0.0] * self.N
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for term in query_tokens:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = self.idf.get(term, 0.0)
                s += idf * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                )
            scores[i] = s
        return scores


def _build_bm25(corpus_tokens: list[list[str]]):
    try:
        from rank_bm25 import BM25Okapi  # type: ignore

        return BM25Okapi(corpus_tokens)
    except ImportError:
        return _PurePythonBM25(corpus_tokens)


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: fuse ranked id lists into one id→score map."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return fused


class Retriever:
    def __init__(
        self,
        corpus: Corpus,
        *,
        embed_query: Optional[EmbedQuery] = None,
        reranker: Optional[Reranker] = None,
    ) -> None:
        self.corpus = corpus
        self.embed_query = embed_query
        self.reranker = reranker
        self._chunk_ids = [c.chunk_id for c in corpus.chunks]
        corpus_tokens = [tokenize(c.text) for c in corpus.chunks]
        # Retain per-chunk token SETS: membership (token overlap), not the sign of
        # the BM25 score, decides what counts as a match in _sparse_ranking.
        self._chunk_token_sets = [set(t) for t in corpus_tokens]
        self._bm25 = _build_bm25(corpus_tokens) if corpus.chunks else None

    def _sparse_ranking(self, query: str) -> list[str]:
        """Rank chunks that share at least one query term, best BM25 score first.

        Membership is decided by token OVERLAP, never by ``score > 0``: BM25 Okapi
        idf turns negative for a term present in a majority of documents (and
        rank_bm25's epsilon guard can't recover it when the mean idf itself is
        negative), so a positivity filter silently drops genuinely-matching
        chunks — on a homogeneous corpus, all of them, leaving retrieval empty.
        Overlap also makes the pure-Python fallback and rank_bm25 agree, so recall
        doesn't depend on which is installed. Negative scores still order the
        survivors correctly; only rank feeds RRF downstream, so the sign is moot.
        """
        if not self._bm25:
            return []
        q_tokens = tokenize(query)
        q_set = set(q_tokens)
        if not q_set:
            return []
        scores = self._bm25.get_scores(q_tokens)
        matched = [i for i in range(len(scores)) if q_set & self._chunk_token_sets[i]]
        matched.sort(key=lambda i: scores[i], reverse=True)
        return [self._chunk_ids[i] for i in matched]

    def _dense_ranking(self, query: str, k: int) -> list[str]:
        if not (self.embed_query and self.corpus.has_vectors):
            return []
        qv = self.embed_query(query)
        return [cid for cid, _ in self.corpus.dense_search(qv, k=k)]

    def retrieve(self, query: str, *, k: int = 10, candidates: int = 40) -> list[dict]:
        """Return up to ``k`` ranked hits: dicts with lineage + score + rank."""
        if not self.corpus.chunks:
            return []
        sparse = self._sparse_ranking(query)[:candidates]
        dense = self._dense_ranking(query, candidates)
        rankings = [r for r in (sparse, dense) if r]
        if not rankings:
            return []
        fused = rrf(rankings)
        ranked_ids = sorted(fused, key=lambda c: fused[c], reverse=True)[:candidates]

        # ``score`` must reflect the ranking that produced the final order: the
        # fused RRF score when no reranker runs, or the cross-encoder's score once
        # it reorders — otherwise a downstream read of ``score`` misrepresents
        # rerank's effect (the fused score no longer matches the emitted order).
        scores: dict[str, float] = dict(fused)
        if self.reranker and ranked_ids:
            docs = [self.corpus.get(cid).text for cid in ranked_ids]
            rr = self.reranker(query, docs)
            order = sorted(zip(ranked_ids, rr), key=lambda t: t[1], reverse=True)
            ranked_ids = [cid for cid, _ in order]
            scores = {cid: float(s) for cid, s in order}

        hits: list[dict] = []
        for rank, cid in enumerate(ranked_ids[:k]):
            ch = self.corpus.get(cid)
            hits.append(
                {
                    "chunk_id": cid,
                    "doc_id": ch.doc_id,
                    "page": ch.page,
                    "text": ch.text,
                    "source": ch.source,
                    "score": round(scores[cid], 6),
                    "rank": rank,
                }
            )
        return hits

    def iterative_retrieve(self, queries: Sequence[str], *, k: int = 10) -> list[dict]:
        """Merge de-duplicated hits across several queries (multi-hop recall).

        Best rank across queries wins; results are re-sorted by that rank. This
        is the retrieval half of the agent's iterative re-query loop (Q8).
        """
        best: dict[str, dict] = {}
        for q in queries:
            for hit in self.retrieve(q, k=k):
                cur = best.get(hit["chunk_id"])
                if cur is None or hit["rank"] < cur["rank"]:
                    best[hit["chunk_id"]] = hit
        merged = sorted(best.values(), key=lambda h: h["rank"])
        return merged[:k]
