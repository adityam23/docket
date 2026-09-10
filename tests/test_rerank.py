"""Offline tests for the cross-encoder rerank seam (T26). No network: a fake
httpx client stands in for the engine's `/v1/rerank` endpoint, and a stub
provider stands in for the `load_retriever` wiring. Deterministic."""

from __future__ import annotations

import pytest

from docket.config import Settings
from docket.ingest.chunk import chunk_pages
from docket.ingest.index import Corpus
from docket.providers.base import Capability
from docket.providers.openai_compat import OpenAICompatProvider
from docket.retrieval.retriever import Retriever


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeClient:
    """Records the last POST and returns a canned Cohere-shaped rerank body."""

    def __init__(self, payload):
        self._payload = payload
        self.last_url = None
        self.last_json = None

    def post(self, url, json=None):
        self.last_url = url
        self.last_json = json
        return _FakeResp(self._payload)


def _provider(payload, **kw):
    p = OpenAICompatProvider(
        name="mettle",
        base_url="http://x/v1",
        chat_model="m",
        rerank_model="bge-reranker-v2-m3:latest",
        **kw,
    )
    p._client = _FakeClient(payload)
    return p


def test_rerank_scatters_scores_back_to_input_order():
    # Endpoint returns results SORTED by score, carrying the original index.
    payload = {
        "model": "bge-reranker-v2-m3:latest",
        "results": [
            {"index": 2, "relevance_score": 0.97},
            {"index": 0, "relevance_score": 0.30},
            {"index": 1, "relevance_score": -5.0},
        ],
    }
    p = _provider(payload)
    scores = p.rerank("capex?", ["a", "b", "c"])
    # One score per input doc, in INPUT order (not the sorted response order).
    assert scores == [0.30, -5.0, 0.97]
    # Request shape matches the contract.
    assert p._client.last_url == "http://x/v1/rerank"
    assert p._client.last_json == {
        "model": "bge-reranker-v2-m3:latest",
        "query": "capex?",
        "documents": ["a", "b", "c"],
    }
    assert Capability.RERANK in p.capabilities


def test_rerank_empty_docs_short_circuits():
    p = _provider({"results": []})
    assert p.rerank("q", []) == []
    assert p._client.last_url is None  # never hit the network


def test_rerank_ignores_out_of_range_index():
    # A malformed/oversized index must not corrupt the score vector.
    payload = {"results": [{"index": 9, "relevance_score": 1.0}, {"index": 0, "relevance_score": 0.5}]}
    p = _provider(payload)
    assert p.rerank("q", ["only"]) == [0.5]


def test_rerank_unconfigured_raises():
    # An engine declared without RERANK must fail loudly, not silently degrade
    # to unranked hits (cutover contract: loud failure).
    p = OpenAICompatProvider(
        name="mettle", base_url="http://x/v1", chat_model="m",
        capabilities=Capability.CHAT,
    )
    assert Capability.RERANK not in p.capabilities
    with pytest.raises(RuntimeError, match="rerank unconfigured"):
        p.rerank("q", ["a"])


def test_engine_serves_rerank_on_the_one_base():
    # No separate rerank_url: the provider was built with just base_url and the
    # RERANK capability is on by default (mettle serves /v1/rerank unconditionally).
    p = OpenAICompatProvider(name="mettle", base_url="http://x/v1", chat_model="m")
    assert Capability.RERANK in p.capabilities


def _acme_chunks():
    pages = [
        {"page": 1, "text": "Total revenue for the fiscal year was ten million dollars."},
        {"page": 2, "text": "Payments for property plant and equipment were thirteen billion."},
    ]
    return chunk_pages(pages, doc_id="acme-10k", source="acme.pdf", words=8, overlap=2)


def test_load_retriever_wires_reranker_and_reorders(monkeypatch):
    from docket import service

    chunks = _acme_chunks()
    texts = {ch.chunk_id: ch.text for ch in chunks}

    class _StubProvider:
        name = "stub"
        capabilities = Capability.CHAT

        # Return every chunk unranked; the reranker does the ordering.
        def collection_search(self, collection, query, *, mode="hybrid", k=10, allowlist=None):
            rows = [{"chunk_id": cid, "document_id": cid, "text": t, "score": 0.0}
                    for cid, t in texts.items()]
            return rows[:k]

        # Rank by document length descending — a deterministic, non-identity order.
        def rerank(self, query, docs):
            return [float(len(d)) for d in docs]

    monkeypatch.setattr(service, "get_provider", lambda s: _StubProvider())
    s = Settings(rerank_model="bge-reranker-v2-m3:latest")
    c = Corpus()
    c.add(chunks)

    r = service.load_retriever(s, corpus=c)
    assert r.reranker is not None
    hits = r.retrieve("equipment", k=5)
    # With the length-ranking reranker the longest chunk must sort first.
    assert hits
    longest = max((texts[h["chunk_id"]] for h in hits), key=len)
    assert hits[0]["text"] == longest


def test_load_retriever_wires_reranker_by_default():
    from docket import service

    r = service.load_retriever(Settings(), corpus=Corpus())
    assert r.reranker is not None  # mettle ships the qwen3-rr reranker default
