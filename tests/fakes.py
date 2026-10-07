"""Shared test doubles for the engine's OpenAI-/v1 + collection surface — ONE
implementation per fake (CLAUDE.md "Reusability"), instead of a bespoke class
hand-rolled per test file. ``ScriptedProvider`` is chat-only (FinanceBench
generator/judge style); ``FakeEngine`` is the full chat+collection surface
(``rank="overlap"`` ranks by query-token overlap, ``rank="all"`` returns every
chunk unranked for tests where a reranker does the ordering).

``collection_search`` treats an EMPTY allowlist as "no allowlist" (mirrors the
real provider's ``if allowlist:`` in ``openai_compat.py`` — a separate ticket
stops an empty scope reaching the provider at all). ``delete_document`` is
False for an id it never held (mirrors the real 404 -> False). ``fail_post_at``
(B4) makes the Nth ``post_document`` call across the fake's whole lifetime
raise, to exercise per-document ingest failure/rollback without a bespoke
local fake per test file."""

from __future__ import annotations

import re
from typing import Callable, Sequence

from docket.ingest.chunk import Chunk, chunk_pages
from docket.ingest.index import Corpus
from docket.providers.base import ChatResult
from docket.retrieval.retriever import Retriever


class ScriptedProvider:
    """Chat-only fake: a scripted reply (string, or ``callable(messages) ->
    str``) and recorded calls."""

    def __init__(self, reply="?", logprobs: list[float] | None = None):
        self._reply = reply
        self.logprobs = logprobs
        self.calls: list[list[dict]] = []

    def health(self) -> dict:
        return {}

    def chat(self, messages: list[dict], **kw) -> ChatResult:
        self.calls.append(messages)
        text = self._reply(messages) if callable(self._reply) else self._reply
        return ChatResult(text=text, logprobs=self.logprobs, raw={})

    @property
    def last_user(self) -> str | None:
        return self.calls[-1][-1]["content"] if self.calls else None


class FakeEngine:
    """Full engine double: chat + the collection surface (ensure/post/delete
    document, search, optional rerank), covering both the "ingest + chat" role
    (``posted``/``deleted`` tracking) and the "search" role."""

    def __init__(
        self,
        chunks: Sequence[Chunk] = (),
        *,
        rank: str = "overlap",
        text: str = "Revenue was $10M [1].",
        logprobs: list[float] | None = None,
        rerank_fn: Callable[[str, list[str]], list[float]] | None = None,
        fail_post_at: int | None = None,
        fail_post_exc: Exception | None = None,
    ) -> None:
        if rank not in ("overlap", "all"):
            raise ValueError(f"unknown rank mode: {rank!r}")
        self._rank = rank
        self._text = text
        self._logprobs = logprobs if logprobs is not None else [-0.02, -0.05, -0.03]
        self._rerank_fn = rerank_fn
        self._docs: dict[str, str] = {ch.chunk_id: ch.text for ch in chunks}
        self.posted: list[tuple[str, str, dict]] = []  # (collection, doc_id, metadata)
        self.deleted: list[tuple[str, str]] = []        # (collection, doc_id)
        self.last_allowlist: list[str] | None = None
        # 1-based count of post_document calls across this fake's lifetime; the
        # call AT this count raises instead of succeeding (B4 ingest-failure
        # tests). None (default) never fails.
        self._fail_post_at = fail_post_at
        self._fail_post_exc = fail_post_exc
        self._post_calls = 0
        self.ensure_collection_calls = 0

    def health(self) -> dict:
        return {"models": ["fake"]}

    def chat(self, messages: list[dict], **kw) -> ChatResult:
        return ChatResult(text=self._text, logprobs=self._logprobs, raw={})

    def ensure_collection(self, collection: str) -> None:
        self.ensure_collection_calls += 1

    def post_document(self, collection, doc_id, text, metadata=None):
        self._post_calls += 1
        if self._fail_post_at is not None and self._post_calls == self._fail_post_at:
            raise self._fail_post_exc or RuntimeError("simulated post_document failure")
        self.posted.append((collection, doc_id, metadata or {}))
        self._docs[doc_id] = text
        return [f"{doc_id}/chunk/0"]

    def delete_document(self, collection, doc_id) -> bool:
        self.deleted.append((collection, doc_id))
        return self._docs.pop(doc_id, None) is not None

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if self._rerank_fn is None:
            raise RuntimeError("rerank unconfigured on this fake")
        return self._rerank_fn(query, documents)

    def collection_search(self, collection, query, *, mode="hybrid", k=10, allowlist=None):
        if mode not in ("dense", "keyword", "hybrid"):
            raise ValueError(f"unknown search mode: {mode!r}")
        self.last_allowlist = list(allowlist) if allowlist is not None else None
        # Mirror the real provider's `if allowlist:` truthiness: an EMPTY
        # allowlist is "no scope", not "match nothing".
        ids = set(allowlist) if allowlist else set(self._docs)

        if self._rank == "all":
            rows = [
                {"chunk_id": cid, "document_id": cid, "text": t, "score": 1.0}
                for cid, t in self._docs.items() if cid in ids
            ]
            return rows[:k]

        q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        rows = []
        for cid in sorted(self._docs):
            if cid not in ids:
                continue
            toks = set(re.findall(r"[a-z0-9]+", self._docs[cid].lower()))
            overlap = len(q_tokens & toks)
            if overlap:
                rows.append({"chunk_id": cid, "document_id": cid,
                             "text": self._docs[cid], "score": float(overlap)})
        rows.sort(key=lambda r: (-r["score"], r["chunk_id"]))
        return rows[:k]


def acme_chunks() -> list[Chunk]:
    """The one shared "Acme 10-K" lineage fixture: revenue p1, opex p2."""
    pages = [
        {"page": 1, "text": "Total revenue for the fiscal year was ten million dollars."},
        {"page": 2, "text": "Operating expenses rose while headcount stayed flat."},
    ]
    return chunk_pages(pages, doc_id="acme-10k", source="acme.pdf", words=8, overlap=2)


def retriever_over(chunks: Sequence[Chunk], *, rank: str = "overlap", **kw) -> Retriever:
    """A Retriever over ``chunks``, backed by a fresh FakeEngine."""
    c = Corpus()
    c.add(chunks)
    engine = FakeEngine(chunks, rank=rank)
    return Retriever(engine, c, collection="docket", **kw)
