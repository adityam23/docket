"""Offline tests for the lite pipeline: chunk → index → retrieve → answer →
report, plus the /ask endpoint. No network, no backend, no real LLM — a fake
provider stands in for inference (per docs/decisions: tests stay deterministic)."""

from __future__ import annotations

import json
import re

import pytest

from docket.agent.graph import answer, plan_operation
from docket.agent.tools import (
    aggregate,
    compute_metric,
    generate_report,
    parse_number,
)
from docket.config import Settings
from docket.ingest.chunk import chunk_pages, chunk_text
from docket.ingest.index import Corpus
from docket.providers.base import Capability, ChatResult
from docket.report.render import to_markdown
from docket.retrieval.retriever import Retriever, rrf, tokenize


# --- fakes ------------------------------------------------------------------

class FakeProvider:
    name = "fake"
    capabilities = Capability.CHAT | Capability.LOGPROBS

    def __init__(self, text="Revenue was $10M [1].", logprobs=None):
        self._text = text
        self._logprobs = logprobs if logprobs is not None else [-0.02, -0.05, -0.03]

    def health(self):
        return {"models": ["fake"]}

    def chat(self, messages, **kw) -> ChatResult:
        return ChatResult(text=self._text, logprobs=self._logprobs, raw={})

    def embed(self, texts):
        # crude deterministic bag-of-words vector so dense search is meaningful
        return [[float(t.count(c)) for c in "abcdefghijklmnopqrstuvwxyz"] for t in texts]


def _corpus():
    pages = [
        {"page": 1, "text": "Total revenue for the fiscal year was ten million dollars."},
        {"page": 2, "text": "Operating expenses rose while headcount stayed flat."},
    ]
    c = Corpus()
    c.add(chunk_pages(pages, doc_id="acme-10k", source="acme.pdf", words=8, overlap=2))
    return c


# --- chunking ---------------------------------------------------------------

def test_corpus_remove_and_doc_ids():
    c = Corpus()
    c.add(chunk_pages([{"page": 1, "text": "alpha beta gamma delta"}], doc_id="a", words=2, overlap=0))
    c.add(chunk_pages([{"page": 1, "text": "one two three four"}], doc_id="b", words=2, overlap=0))
    assert set(c.doc_ids()) == {"a", "b"}

    before = len(c)
    removed = c.remove("a")
    assert removed > 0
    assert c.doc_ids() == ["b"] and len(c) == before - removed
    # id index was rebuilt: every surviving chunk is still addressable
    for ch in c.chunks:
        assert c.get(ch.chunk_id) is ch
    assert c.remove("a") == 0  # removing a gone doc is a no-op


def test_remove_document_service_persists(tmp_path):
    from docket.service import remove_document

    s = Settings(index_dir=str(tmp_path), embed_url=None)
    c = Corpus()
    c.add(chunk_pages([{"page": 1, "text": "foo bar baz"}], doc_id="d1", words=2, overlap=0))
    c.add(chunk_pages([{"page": 1, "text": "qux quux corge"}], doc_id="d2", words=2, overlap=0))
    c.save(s.index_dir)

    assert remove_document("d1", settings=s) > 0
    assert Corpus.load(s.index_dir).doc_ids() == ["d2"]
    assert remove_document("nope", settings=s) == 0  # unknown id


def test_load_samples_idempotent(tmp_path):
    from docket.ingest.samples import load_samples, sample_doc_ids

    s = Settings(index_dir=str(tmp_path), embed_url=None)  # embeddings off → sparse
    c = load_samples(settings=s)
    ids = set(c.doc_ids())
    assert set(sample_doc_ids()) <= ids
    assert all(i.startswith("sample-") for i in sample_doc_ids())

    n = len(c)
    again = load_samples(settings=s)          # idempotent: re-load adds nothing
    assert len(again) == n
    assert set(Corpus.load(s.index_dir).doc_ids()) >= set(sample_doc_ids())  # persisted


def test_capacity_stats_empty_and_measured(tmp_path):
    from docket.web.observability import capacity_stats

    s = Settings(index_dir=str(tmp_path), embed_url=None)
    empty = capacity_stats(s)
    assert empty["documents_ingested"] == 0
    assert empty["bytes_per_document_measured"] is False
    assert empty["remaining_documents_est"] >= 0
    assert empty["device"]["disk_total_bytes"] > 0

    c = Corpus()
    c.add(chunk_pages([{"page": 1, "text": "alpha " * 60}], doc_id="d1", words=20, overlap=5))
    c.save(s.index_dir)
    measured = capacity_stats(s)
    assert measured["documents_ingested"] == 1
    assert measured["bytes_per_document_measured"] is True
    assert measured["corpus_bytes"] > 0


def test_chunk_respects_page_and_lineage():
    chunks = chunk_text("one two three four five", doc_id="d", page=7, source="f.pdf", words=3, overlap=1)
    assert all(ch.page == 7 and ch.doc_id == "d" and ch.source == "f.pdf" for ch in chunks)
    assert chunks[0].chunk_id == "d#0"
    assert len(chunks) >= 2  # windowed


def test_chunk_never_crosses_pages():
    pages = [{"page": 1, "text": "a b c"}, {"page": 2, "text": "d e f"}]
    chunks = chunk_pages(pages, doc_id="d", words=10, overlap=0)
    pages_seen = {ch.page for ch in chunks}
    assert pages_seen == {1, 2}
    assert all(" " in ch.text for ch in chunks)


# --- index ------------------------------------------------------------------

def test_dense_search_and_persistence(tmp_path):
    c = Corpus()
    from docket.ingest.chunk import Chunk

    c.add(
        [Chunk("d", "d#0", 1, "alpha"), Chunk("d", "d#1", 1, "beta")],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    hits = c.dense_search([0.9, 0.1], k=1)
    assert hits[0][0] == "d#0"

    c.save(str(tmp_path))
    loaded = Corpus.load(str(tmp_path))
    assert len(loaded) == 2
    assert loaded.has_vectors
    assert loaded.dense_search([0.0, 1.0], k=1)[0][0] == "d#1"


def test_add_is_idempotent():
    c = _corpus()
    n = len(c)
    c.add(chunk_pages([{"page": 1, "text": "Total revenue for the fiscal year was ten million dollars."}],
                      doc_id="acme-10k", words=8, overlap=2))
    assert len(c) == n  # duplicate chunk_ids skipped


# --- retriever --------------------------------------------------------------

def test_rrf_fuses_rankings():
    fused = rrf([["a", "b", "c"], ["c", "a"]])
    assert fused["a"] > fused["b"]  # a ranks high in both
    assert set(fused) == {"a", "b", "c"}


def test_sparse_retrieval_finds_needle():
    r = Retriever(_corpus())
    hits = r.retrieve("revenue", k=3)
    assert hits and "revenue" in hits[0]["text"].lower()
    assert hits[0]["doc_id"] == "acme-10k" and hits[0]["page"] == 1


def test_hybrid_uses_dense_when_embedder_present():
    c = _corpus()
    # attach vectors so dense path is live
    fp = FakeProvider()
    vecs = fp.embed([ch.text for ch in c.chunks])
    c.vectors = vecs
    r = Retriever(c, embed_query=lambda q: fp.embed([q])[0])
    hits = r.retrieve("expenses headcount", k=2)
    assert any("expenses" in h["text"].lower() for h in hits)


def test_retrieve_empty_corpus():
    assert Retriever(Corpus()).retrieve("anything") == []


def _homogeneous_corpus():
    """Every doc shares the query terms → BM25 idf goes negative. Plus one doc
    that shares nothing, to check it stays excluded."""
    c = Corpus()
    for i in range(1, 4):
        c.add(chunk_pages([{"page": 1, "text": f"Your monthly charge is {i}0 dollars."}],
                          doc_id=f"bill-{i}", words=12, overlap=0))
    c.add(chunk_pages([{"page": 1, "text": "Unrelated board meeting minutes."}],
                      doc_id="misc", words=12, overlap=0))
    return c


def test_sparse_recall_on_ubiquitous_terms():
    # Regression: 'monthly'/'charge' are in every bill, so Okapi idf is negative
    # and the old `score > 0` filter dropped all matches (retrieval returned []).
    hits = Retriever(_homogeneous_corpus()).retrieve("monthly charge", k=10)
    ids = {h["doc_id"] for h in hits}
    assert ids == {"bill-1", "bill-2", "bill-3"}  # all matches kept, non-match excluded


def test_sparse_recall_consistent_across_bm25_backends(monkeypatch):
    # The pure-Python fallback and rank_bm25 must give the SAME recall, so the
    # experience doesn't change with the `ingest` extra installed or not.
    import docket.retrieval.retriever as rmod
    from docket.retrieval.retriever import _PurePythonBM25

    default_ids = {h["doc_id"] for h in Retriever(_homogeneous_corpus()).retrieve("monthly charge", k=10)}
    monkeypatch.setattr(rmod, "_build_bm25", lambda toks: _PurePythonBM25(toks))
    pure_ids = {h["doc_id"] for h in Retriever(_homogeneous_corpus()).retrieve("monthly charge", k=10)}
    assert default_ids == pure_ids == {"bill-1", "bill-2", "bill-3"}


def test_tokenize():
    assert tokenize("Revenue: $10M, up 5%!") == ["revenue", "10m", "up", "5"]


# --- orchestrator -----------------------------------------------------------

def test_answer_is_grounded_cited_and_scored():
    r = Retriever(_corpus())
    res = answer("What was revenue?", retriever=r, provider=FakeProvider(), settings=Settings())
    assert res.reliability == "high"           # low surprisal logprobs
    assert res.citations and res.citations[0].doc_id == "acme-10k"
    assert res.hits


def test_answer_low_confidence_label():
    r = Retriever(_corpus())
    unsure = FakeProvider(logprobs=[-1.8, -2.2, -1.5])
    res = answer("What was revenue?", retriever=r, provider=unsure, settings=Settings(max_hops=1))
    assert res.reliability == "low"


def test_answer_unknown_without_logprobs():
    r = Retriever(_corpus())
    res = answer("q", retriever=r, provider=FakeProvider(logprobs=[]), settings=Settings())
    assert res.reliability == "unknown"


# --- observability read model -----------------------------------------------

def test_corpus_stats_reports_docs_stages_and_embeddings(tmp_path, monkeypatch):
    from docket.ingest.chunk import Chunk
    from docket.web.observability import corpus_stats

    c = Corpus()
    c.add(
        [Chunk("acme", "acme#0", 1, "alpha revenue"), Chunk("acme", "acme#1", 2, "beta expenses")],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    c.add([Chunk("memo", "memo#0", 1, "gamma note")])  # sparse-only doc
    c.save(str(tmp_path))

    s = Settings(index_dir=str(tmp_path), embed_url="http://x/v1")  # dense enabled
    stats = corpus_stats(s)
    assert stats["totals"]["documents"] == 2 and stats["totals"]["chunks"] == 3
    assert stats["embeddings"]["enabled"] and stats["embeddings"]["dim"] == 2
    by_id = {d["doc_id"]: d for d in stats["documents"]}
    assert by_id["acme"]["stage"] == "embedded" and by_id["acme"]["pages"] == 2
    assert by_id["memo"]["stage"] == "indexed"  # no vectors → sparse-only


# --- tools + report ---------------------------------------------------------

def test_compute_metric():
    assert compute_metric("growth", current=110, previous=100) == pytest.approx(0.10)
    assert compute_metric("margin", numerator=20, denominator=100) == pytest.approx(0.20)
    with pytest.raises(ValueError):
        compute_metric("bogus")


# --- deterministic aggregation (cross-document totals) ----------------------

def test_parse_number_units_and_multipliers():
    assert parse_number("$1,200") == (1200.0, "USD")
    assert parse_number("$3.2 billion") == (3_200_000_000.0, "USD")
    assert parse_number("1.5M") == (1_500_000.0, "")   # bare magnitude, no currency mark
    assert parse_number("10%") == (10.0, "%")
    assert parse_number(42) == (42.0, "")
    with pytest.raises(ValueError):
        parse_number("n/a")


def test_aggregate_ops_and_unit_guard():
    usd = [{"value": "$10"}, {"value": "$20"}, {"value": "$30"}]
    s = aggregate("sum", usd)
    assert s["result"] == 60.0 and s["unit"] == "USD" and s["count"] == 3
    assert aggregate("mean", usd)["result"] == pytest.approx(20.0)
    assert aggregate("max", usd)["result"] == 30.0
    assert aggregate("count", usd)["result"] == 3
    with pytest.raises(ValueError):        # refuse to add dollars to percentages
        aggregate("sum", [{"value": "$10"}, {"value": "5%"}])
    with pytest.raises(ValueError):        # empty roll-up is an error, not a silent 0
        aggregate("sum", [])
    assert aggregate("count", [])["result"] == 0  # ...but counting nothing is 0


def test_plan_operation_classifies():
    assert plan_operation("What is the total monthly charge across all documents?").name == "sum"
    assert plan_operation("What is the average monthly bill?").name == "mean"
    assert plan_operation("Which vendor charges the highest fee?").name == "max"
    assert plan_operation("How many invoices are there?").name == "count"
    assert plan_operation("How much do I pay per month across my subscriptions?").name == "sum"
    assert plan_operation("What was revenue in 2023?") is None  # plain QA, untouched


def _multi_doc_corpus():
    # Three near-identical bills: the charge terms appear in EVERY doc, which
    # drives BM25 idf negative. Retrieval must still find all three (see
    # test_sparse_recall_on_ubiquitous_terms) — no filler docs papering over it.
    c = Corpus()
    for i, amt in enumerate(("$10", "$20", "$30"), start=1):
        c.add(chunk_pages(
            [{"page": 1, "text": f"Your monthly subscription charge is {amt} per month."}],
            doc_id=f"bill-{i}", source=f"bill{i}.pdf", words=12, overlap=0))
    return c


class _ExtractingProvider(FakeProvider):
    """A faithful extraction stub: reads the numbered context it's actually given
    and emits the correct ``i``→value JSON (a real model does this from the same
    prompt). Unlike a blind fixed-index fake, this exercises the real hit→citation
    mapping, so a provenance bug can't hide behind an order-independent sum."""

    _LINE = re.compile(r"\[(\d+)\] \([^)]*\) .*?(\$[\d.,]+) per month")

    def chat(self, messages, **kw) -> ChatResult:
        context = messages[-1]["content"]
        found = [{"i": int(i), "value": v} for i, v in self._LINE.findall(context)]
        return ChatResult(text=json.dumps(found), logprobs=self._logprobs, raw={})


def test_answer_aggregates_across_docs():
    r = Retriever(_multi_doc_corpus())
    res = answer(
        "What is the total monthly charge across all documents?",
        retriever=r, provider=_ExtractingProvider(), settings=Settings(),
    )
    assert res.detail is not None
    assert res.detail["op"] == "sum" and res.detail["result"] == 60.0
    assert res.detail["unit"] == "USD" and res.detail["count"] == 3
    # every summed value is cited AND anchored to the right bill (not a misc doc)
    assert {c.doc_id for c in res.citations} == {"bill-1", "bill-2", "bill-3"}
    assert res.reliability == "high"               # default low-surprisal logprobs
    assert "Total = $60.00" in res.answer


def test_answer_aggregate_without_values_is_honest():
    r = Retriever(_multi_doc_corpus())
    res = answer(
        "What is the total charge across all documents?",
        retriever=r, provider=FakeProvider(text="[]"), settings=Settings(),
    )
    assert res.detail is None
    assert "do not contain" in res.answer.lower()   # no invented number
    assert res.reliability == "unknown"


def test_aggregate_dedups_overlapping_duplicate_values():
    # Regression: chunk overlap can surface the SAME figure in two adjacent
    # retrieved chunks; extracting from both would double-count it (the reported
    # "incorrect total" bug). Same doc+page+value → counted once, not twice.
    from docket.ingest.chunk import Chunk

    c = Corpus()
    c.add([
        Chunk("bill", "bill#0", 1, "monthly charge is $10 per month"),
        Chunk("bill", "bill#1", 1, "the $10 per month monthly charge applies"),
    ])
    res = answer(
        "What is the total monthly charge across all documents?",
        retriever=Retriever(c), provider=_ExtractingProvider(), settings=Settings(),
    )
    assert res.detail["result"] == 10.0 and res.detail["count"] == 1  # not $20
    assert {cit.doc_id for cit in res.citations} == {"bill"}


def test_aggregate_flags_partial_coverage():
    # Two documents are in scope but only one states an extractable value; the
    # total must be flagged partial rather than presented as complete.
    from docket.ingest.chunk import Chunk

    c = Corpus()
    c.add([Chunk("bill-a", "bill-a#0", 1, "monthly charge is $10 per month")])
    c.add([Chunk("bill-b", "bill-b#0", 1, "a monthly charge applies but no figure per month here")])
    res = answer(
        "What is the total monthly charge across all documents?",
        retriever=Retriever(c), provider=_ExtractingProvider(), settings=Settings(),
    )
    cov = res.detail["coverage"]
    assert cov["partial"] is True
    assert cov["documents_contributing"] == 1 and cov["documents_in_scope"] == 2
    assert "may be incomplete" in res.answer


def test_ingest_paths_skips_already_indexed_doc(monkeypatch, tmp_path):
    # T09: re-ingesting an already-indexed document must skip WITHOUT re-running
    # OCR/embed; skip_existing=False forces a re-run.
    import docket.ingest.pipeline as pipe

    monkeypatch.setattr(
        pipe, "pdf_to_pages",
        lambda path: [{"page": 1, "text": "monthly charge is ten dollars per year"}],
    )
    s = Settings(index_dir=str(tmp_path), embed_url=None)  # embeddings off → no network
    c = Corpus()

    first = []
    pipe.ingest_paths(["/some/where/acme.pdf"], settings=s, corpus=c,
                      on_event=lambda d, st, i: first.append((d, st)))
    n = len(c)
    assert n > 0 and ("acme", "ocr") in first

    again = []
    pipe.ingest_paths(["/other/path/acme.pdf"], settings=s, corpus=c,
                      on_event=lambda d, st, i: again.append((d, st)))
    assert len(c) == n                              # no growth
    assert ("acme", "skipped") in again             # surfaced as skipped
    assert all(st != "ocr" for _, st in again)      # OCR never re-ran

    forced = []
    pipe.ingest_paths(["/other/path/acme.pdf"], settings=s, corpus=c, skip_existing=False,
                      on_event=lambda d, st, i: forced.append((d, st)))
    assert ("acme", "ocr") in forced                # force bypasses the skip


def test_capacity_reports_ram_bound(tmp_path):
    # T14: capacity is the tighter of disk and RAM, and reports which one binds.
    from docket.web.observability import capacity_stats

    s = Settings(index_dir=str(tmp_path), embed_url=None)
    c = Corpus()
    c.add(chunk_pages([{"page": 1, "text": "alpha " * 60}], doc_id="d1", words=20, overlap=5))
    c.save(s.index_dir)

    cap = capacity_stats(s)
    assert cap["binding_constraint"] in ("disk", "ram")
    assert cap["disk_bound_documents_est"] >= 0
    assert cap["ram_bytes_per_document"] > 0
    if cap["device"]["ram_total_bytes"]:  # platform exposes total RAM
        assert cap["ram_bound_documents_est"] is not None
        assert cap["remaining_documents_est"] == min(
            cap["disk_bound_documents_est"], cap["ram_bound_documents_est"]
        )


def test_generate_report_and_render():
    r = Retriever(_corpus())
    res = answer("What was revenue?", retriever=r, provider=FakeProvider(), settings=Settings())
    report = generate_report(res, title="Acme 10-K")
    md = to_markdown(report)
    assert "# Acme 10-K" in md
    assert "🟢" in md  # reliability banner
    assert "acme-10k" in md
