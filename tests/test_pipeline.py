"""Offline tests for the pipeline: chunk → engine-post → lineage → retrieve →
answer → report. No network, no engine, no real LLM — a fake provider stands in
for the engine's /v1 surface (per docs/decisions: tests stay deterministic)."""

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
from docket.providers.base import ChatResult
from docket.report.render import to_markdown

from .fakes import FakeEngine, acme_chunks, retriever_over


def _corpus():
    c = Corpus()
    c.add(acme_chunks())
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
        assert c.find(ch.chunk_id) is ch
    assert c.find("no-such-chunk") is None  # unknown id -> None, never a KeyError
    assert c.remove("a") == 0  # removing a gone doc is a no-op


def test_lineage_persists_without_text(tmp_path):
    # The corpus stores lineage (incl. chars), never chunk text: display text
    # comes back from engine hits (cutover contract §1.4).
    c = _corpus()
    c.save(str(tmp_path))
    loaded = Corpus.load(str(tmp_path))
    assert [ch.chunk_id for ch in loaded.chunks] == [ch.chunk_id for ch in c.chunks]
    first = c.chunks[0]
    back = loaded.find(first.chunk_id)
    assert back.doc_id == first.doc_id and back.page == first.page
    assert back.source == first.source


def test_remove_document_service_persists(tmp_path, monkeypatch):
    import docket.service as svc

    d1_chunks = chunk_pages([{"page": 1, "text": "foo bar baz"}], doc_id="d1", words=2, overlap=0)
    d2_chunks = chunk_pages([{"page": 1, "text": "qux quux corge"}], doc_id="d2", words=2, overlap=0)
    # The engine actually holds these chunks, so every delete succeeds (2xx) --
    # the normal case.
    engine = FakeEngine(list(d1_chunks) + list(d2_chunks))
    monkeypatch.setattr(svc, "get_provider", lambda s: engine)

    s = Settings(index_dir=str(tmp_path))
    c = Corpus()
    c.add(d1_chunks)
    c.add(d2_chunks)
    c.save(s.index_dir)

    assert svc.remove_document("d1", settings=s) > 0
    assert Corpus.load(s.index_dir).doc_ids() == ["d2"]
    assert svc.remove_document("nope", settings=s) == 0  # unknown id


def test_remove_document_raises_and_keeps_lineage_when_engine_missing_chunk(tmp_path, monkeypatch):
    """The engine reports a chunk 404 (out of sync with our lineage) -- removal
    must fail loudly and leave the WHOLE document's lineage untouched, not
    silently report success while chunks are left dangling or half-removed."""
    import docket.service as svc

    d1_chunks = chunk_pages([{"page": 1, "text": "foo bar baz"}], doc_id="d1", words=2, overlap=0)
    # Empty engine: it never held any of d1's chunks, so every delete 404s.
    engine = FakeEngine()
    monkeypatch.setattr(svc, "get_provider", lambda s: engine)

    s = Settings(index_dir=str(tmp_path))
    c = Corpus()
    c.add(d1_chunks)
    c.save(s.index_dir)

    with pytest.raises(RuntimeError, match="does not hold"):
        svc.remove_document("d1", settings=s)
    assert Corpus.load(s.index_dir).doc_ids() == ["d1"]  # lineage kept for retry


def test_load_samples_idempotent(tmp_path, monkeypatch):
    from docket.ingest.samples import load_samples, sample_doc_ids

    monkeypatch.setattr("docket.ingest.pipeline.get_provider", lambda s: FakeEngine())
    s = Settings(index_dir=str(tmp_path))
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

    s = Settings(index_dir=str(tmp_path))
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

def test_add_is_idempotent():
    c = _corpus()
    n = len(c)
    c.add(chunk_pages(
        [{"page": 1, "text": "Total revenue for the fiscal year was ten million dollars."}],
        doc_id="acme-10k", source="acme.pdf", words=8, overlap=2))
    assert len(c) == n  # duplicate chunk_ids skipped


# --- retriever (engine client) ----------------------------------------------

def test_retrieval_joins_engine_hits_to_lineage():
    r = retriever_over(acme_chunks())
    hits = r.retrieve("revenue", k=3)
    assert hits and "revenue" in hits[0]["text"].lower()
    assert hits[0]["doc_id"] == "acme-10k" and hits[0]["page"] == 1
    assert hits[0]["engine_chunk_id"]  # engine's own chunk id carried along


def test_retrieve_empty_corpus_skips_engine():
    # No lineage → no candidates exist; the doomed engine call is skipped, not
    # a silent substitute (there is nothing to retrieve over).
    r = retriever_over([])
    assert r.retrieve("anything") == []
    assert not r.provider.posted


def test_unknown_mode_rejected_loudly():
    with pytest.raises(ValueError, match="unknown search mode"):
        retriever_over(acme_chunks(), mode="rerank_experimental")


def test_scope_doc_restricts_engine_allowlist():
    chunks = _multi_doc_chunks()
    r = retriever_over(chunks, scope_doc="bill-2")
    hits = r.retrieve("monthly charge", k=10)
    # The allowlist the engine received covers ONLY bill-2's chunks.
    assert r.provider.last_allowlist == [ch.chunk_id for ch in chunks if ch.doc_id == "bill-2"]
    assert {h["doc_id"] for h in hits} == {"bill-2"}


def test_scope_doc_with_no_lineage_returns_empty_not_whole_collection():
    # scope_doc names a doc with NO lineage rows (stale/retired/typo'd id) ->
    # the allowlist resolves to []. An EMPTY allowlist reads as "no scope" to
    # the real provider (FakeEngine mirrors that semantics) -- falling
    # through to collection_search would silently search the WHOLE
    # collection instead of the (empty) requested scope.
    chunks = _multi_doc_chunks()
    r = retriever_over(chunks, scope_doc="no-such-doc")
    hits = r.retrieve("monthly charge", k=10)
    assert hits == []
    assert r.provider.last_allowlist is None  # engine search was never reached


def test_retrieve_drops_orphaned_hits_and_warns(caplog):
    # An engine hit whose chunk_id has no lineage row (old delete bug,
    # mid-batch ingest failure, a benchmark run into the prod collection) has
    # no doc_id/page to cite -- it must be dropped, not surfaced with page=None
    # (which previously blew up Citation.from_hit's `page: int`).
    from docket.retrieval.retriever import Retriever

    chunks = acme_chunks()
    orphan = chunk_pages(
        [{"page": 1, "text": "orphan chunk with revenue mentioned but no lineage row"}],
        doc_id="ghost", words=8, overlap=0,
    )
    engine = FakeEngine(chunks + orphan, rank="all")
    corpus = Corpus()
    corpus.add(chunks)  # lineage deliberately omits the orphan chunk

    r = Retriever(engine, corpus, collection="docket")
    with caplog.at_level("WARNING"):
        hits = r.retrieve("revenue", k=10)

    assert {h["chunk_id"] for h in hits} == {ch.chunk_id for ch in chunks}
    assert all(h["page"] is not None for h in hits)
    assert any("lineage" in rec.message.lower() for rec in caplog.records)


def test_iterative_retrieve_merges_and_dedups():
    chunks = chunk_pages(
        [{"page": 1, "text": "revenue grew ten million"}], doc_id="d", words=6, overlap=0)
    hits = retriever_over(chunks).iterative_retrieve(["revenue", "revenue grew"], k=5)
    assert hits
    assert [h["rank"] for h in hits] == sorted(h["rank"] for h in hits)
    assert len({h["chunk_id"] for h in hits}) == len(hits)  # deduped


# --- orchestrator -----------------------------------------------------------

def test_answer_is_grounded_cited_and_scored():
    r = retriever_over(acme_chunks())
    res = answer("What was revenue?", retriever=r, provider=FakeEngine(), settings=Settings())
    assert res.reliability == "high"           # low surprisal logprobs
    assert res.citations and res.citations[0].doc_id == "acme-10k"
    assert res.hits


def test_answer_without_markers_has_no_citations():
    # No [n] marker in the model's text -> no citations. Citing hit 1 anyway
    # (the old fallback) fakes grounding for text the model never anchored.
    r = retriever_over(acme_chunks())
    res = answer(
        "What was revenue?", retriever=r,
        provider=FakeEngine(text="Revenue was ten million dollars."),
        settings=Settings(),
    )
    assert res.citations == []


def test_refusal_has_no_citations():
    from docket.agent.graph import REFUSAL

    r = retriever_over(acme_chunks())
    res = answer(
        "What was revenue?", retriever=r,
        provider=FakeEngine(text=REFUSAL),
        settings=Settings(),
    )
    assert res.answer == REFUSAL
    assert res.citations == []


def test_answer_low_confidence_label():
    r = retriever_over(acme_chunks())
    unsure = FakeEngine(logprobs=[-1.8, -2.2, -1.5])
    res = answer("What was revenue?", retriever=r, provider=unsure, settings=Settings(max_hops=1))
    assert res.reliability == "low"


def test_answer_unknown_without_logprobs():
    r = retriever_over(acme_chunks())
    res = answer("q", retriever=r, provider=FakeEngine(logprobs=[]), settings=Settings())
    assert res.reliability == "unknown"


# --- observability read model -----------------------------------------------

def test_corpus_stats_reports_docs_and_stages(tmp_path):
    from docket.ingest.chunk import Chunk
    from docket.web.observability import corpus_stats

    c = Corpus()
    c.add([Chunk("acme", "acme#0", 1, "alpha revenue"), Chunk("acme", "acme#1", 2, "beta expenses")])
    c.add([Chunk("memo", "memo#0", 1, "gamma note")])
    c.save(str(tmp_path))

    s = Settings(index_dir=str(tmp_path))
    stats = corpus_stats(s)
    assert stats["totals"]["documents"] == 2 and stats["totals"]["chunks"] == 3
    assert stats["totals"]["chars"] > 0
    by_id = {d["doc_id"]: d for d in stats["documents"]}
    assert by_id["acme"]["stage"] == "indexed" and by_id["acme"]["pages"] == 2
    assert by_id["memo"]["stage"] == "indexed"
    assert stats["retrieval"]["mode"] == "hybrid"


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


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Suffix must be a whole word, not any string starting with it.
        ("$1,200 monthly", (1200.0, "USD")),     # "monthly" is not the "m" (million) suffix
        ("5 bps", (5.0, "")),                     # "bps" is not the "b" (billion) suffix
        ("12 months", (12.0, "")),                # "months" is not the "m" (million) suffix
        ("$500 mid-year", (500.0, "USD")),        # "mid-year" is not the "m" (million) suffix
        # Suffixes that ARE the whole word still apply.
        ("1.5M", (1_500_000.0, "")),
        ("3.2 billion", (3_200_000_000.0, "")),
        # Accounting parentheses denote negative.
        ("($1,234)", (-1234.0, "USD")),
        ("(1,234)", (-1234.0, "")),
    ],
)
def test_parse_number_suffix_word_boundary_and_parens(raw, expected):
    assert parse_number(raw) == expected


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
    assert plan_operation("What is the average monthly bill across all accounts?").name == "mean"
    assert plan_operation("Which vendor charges the highest fee, combined across all vendors?").name == "max"
    assert plan_operation("How many invoices are there, added up across all vendors?").name == "count"
    assert plan_operation("How much do I pay per month across my subscriptions?").name == "sum"
    assert plan_operation("What was revenue in 2023?") is None  # plain QA, untouched


@pytest.mark.parametrize(
    "question,expected",
    [
        # Positive: an explicit multi-item/across-documents signal is present
        # alongside the op keyword, so the roll-up path is genuinely warranted.
        ("What is the average revenue across all segments?", "mean"),
        ("What is the combined revenue of all subsidiaries?", "sum"),
        ("Sum of all R&D expenses across each of the filings.", "sum"),
        ("What was the highest revenue among all reporting segments?", "max"),
        ("What is the total of combined assets for both Company A and Company B?", "sum"),
        ("How many of the filings, combined, mention a material weakness?", "count"),
        # Negative: single-entity, single-period metric questions — plain QA
        # even though they contain an _OP_TRIGGERS keyword. These are the
        # FinanceBench single-fact questions this ticket stops mis-routing.
        ("What is the FY2017 operating cash flow ratio for Adobe?", None),
        ("What was the year end FY2019 total amount of inventories for Adobe?", None),
        ("What is total revenue for Adobe in FY2017?", None),
        ("What is the total number of shares outstanding?", None),
        ("How many employees does Adobe have?", None),
        ("What was the highest revenue reported this year?", None),
        ("What was revenue in 2023?", None),
    ],
)
def test_plan_operation_requires_cross_scope_signal(question, expected):
    op = plan_operation(question)
    assert (op.name if op else None) == expected


def _multi_doc_chunks():
    # Three near-identical bills: the charge terms appear in EVERY doc. The
    # engine's search (fake here) must still return all three — no filler docs
    # papering over it.
    chunks = []
    for i, amt in enumerate(("$10", "$20", "$30"), start=1):
        chunks += chunk_pages(
            [{"page": 1, "text": f"Your monthly subscription charge is {amt} per month."}],
            doc_id=f"bill-{i}", source=f"bill{i}.pdf", words=12, overlap=0)
    return chunks


class _ExtractingProvider(FakeEngine):
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
    r = retriever_over(_multi_doc_chunks())
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
    r = retriever_over(_multi_doc_chunks())
    res = answer(
        "What is the total charge across all documents?",
        retriever=r, provider=FakeEngine(text="[]"), settings=Settings(),
    )
    assert res.detail is None
    assert "do not contain" in res.answer.lower()   # no invented number
    assert res.reliability == "unknown"


def test_aggregate_dedups_overlapping_duplicate_values():
    # Regression: chunk overlap can surface the SAME figure in two adjacent
    # retrieved chunks; extracting from both would double-count it (the reported
    # "incorrect total" bug). Same doc+page+value → counted once, not twice.
    from docket.ingest.chunk import Chunk

    chunks = [
        Chunk("bill", "bill#0", 1, "monthly charge is $10 per month"),
        Chunk("bill", "bill#1", 1, "the $10 per month monthly charge applies"),
    ]
    res = answer(
        "What is the total monthly charge across all documents?",
        retriever=retriever_over(chunks), provider=_ExtractingProvider(), settings=Settings(),
    )
    assert res.detail["result"] == 10.0 and res.detail["count"] == 1  # not $20
    assert {cit.doc_id for cit in res.citations} == {"bill"}


def test_aggregate_flags_partial_coverage():
    # Two documents are in scope but only one states an extractable value; the
    # total must be flagged partial rather than presented as complete.
    from docket.ingest.chunk import Chunk

    chunks = [
        Chunk("bill-a", "bill-a#0", 1, "monthly charge is $10 per month"),
        Chunk("bill-b", "bill-b#0", 1, "a monthly charge applies but no figure per month here"),
    ]
    res = answer(
        "What is the total monthly charge across all documents?",
        retriever=retriever_over(chunks), provider=_ExtractingProvider(), settings=Settings(),
    )
    cov = res.detail["coverage"]
    assert cov["partial"] is True
    assert cov["documents_contributing"] == 1 and cov["documents_in_scope"] == 2
    assert "may be incomplete" in res.answer


def test_ingest_paths_skips_already_indexed_doc(monkeypatch, tmp_path):
    # Re-ingesting an already-indexed document must skip WITHOUT re-running OCR
    # or re-posting chunks to the engine; skip_existing=False forces a re-run.
    import docket.ingest.pipeline as pipe

    stub = FakeEngine()
    monkeypatch.setattr(pipe, "get_provider", lambda s: stub)
    monkeypatch.setattr(
        pipe, "pdf_to_pages",
        lambda path: [{"page": 1, "text": "monthly charge is ten dollars per year"}],
    )
    s = Settings(index_dir=str(tmp_path))
    c = Corpus()

    first = []
    pipe.ingest_paths(["/some/where/acme.pdf"], settings=s, corpus=c,
                      on_event=lambda d, st, i: first.append((d, st)))
    n = len(c)
    assert n > 0 and ("acme", "ocr") in first
    assert stub.posted  # chunks actually went to the engine

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


def test_ingest_posts_filing_metadata_to_engine(monkeypatch, tmp_path):
    # Citation metadata must ride along to the engine document (round-trips into
    # mettle's documents.metadata_json).
    import docket.ingest.pipeline as pipe

    stub = FakeEngine()
    monkeypatch.setattr(pipe, "get_provider", lambda s: stub)
    monkeypatch.setattr(
        pipe, "pdf_to_pages",
        lambda path: [{"page": 1, "text": "total revenue was ten million dollars"}],
    )
    s = Settings(index_dir=str(tmp_path))
    pipe.ingest_paths(["/some/where/3M_2018_10K.pdf"], settings=s, corpus=Corpus())
    assert stub.posted
    _, doc_id, meta = stub.posted[0]
    assert doc_id == "3M_2018_10K#0"
    assert meta["form_type"] == "10-K"
    assert meta["page"] == "1" and meta["fiscal_period"] == "2018"


def test_ingest_paths_builds_provider_once_per_run(monkeypatch, tmp_path):
    # B4 item 4: a prior version called get_provider() + ensure_collection()
    # once PER DOCUMENT (a fresh httpx client + an extra POST each) -- must
    # happen exactly once for the whole run, regardless of document count.
    import docket.ingest.pipeline as pipe

    stub = FakeEngine()
    calls = {"get_provider": 0}

    def fake_get_provider(s):
        calls["get_provider"] += 1
        return stub

    monkeypatch.setattr(pipe, "get_provider", fake_get_provider)
    monkeypatch.setattr(
        pipe, "pdf_to_pages", lambda path: [{"page": 1, "text": "alpha beta gamma"}],
    )
    s = Settings(index_dir=str(tmp_path))
    pipe.ingest_paths(["/x/a.pdf", "/x/b.pdf", "/x/c.pdf"], settings=s, corpus=Corpus())
    assert calls["get_provider"] == 1
    assert stub.ensure_collection_calls == 1


def test_ingest_paths_between_documents_seam_fires_once_per_document(monkeypatch, tmp_path):
    # B4 follow-up seam: exactly one call point between documents (done,
    # skipped, OR failed), so a future ticket can hook a pause there.
    import docket.ingest.pipeline as pipe

    monkeypatch.setattr(pipe, "get_provider", lambda s: FakeEngine())
    monkeypatch.setattr(
        pipe, "pdf_to_pages", lambda path: [{"page": 1, "text": "alpha beta gamma"}],
    )
    s = Settings(index_dir=str(tmp_path))
    ticks = []
    pipe.ingest_paths(
        ["/x/a.pdf", "/x/b.pdf"], settings=s, corpus=Corpus(),
        between_documents=lambda: ticks.append(True),
    )
    assert len(ticks) == 2  # once per document, including the default no-op path

    # Default (no hook given) must not raise -- the no-op is the default.
    pipe.ingest_paths(["/x/a.pdf"], settings=s, corpus=Corpus())


def test_ingest_doc_failure_persists_earlier_lineage_rolls_back_and_continues(monkeypatch, tmp_path):
    # Contract D3: a document whose POST fails mid-way is recorded "failed"
    # with NO lineage written, its already-posted chunks are best-effort
    # DELETEd (so a retry doesn't 409), and the batch continues -- an EARLIER
    # successful document's lineage must already be on disk, not just held in
    # memory until some final batch-wide save.
    import docket.ingest.pipeline as pipe

    pages_by_doc = {
        "a": [{"page": 1, "text": "alpha"}],                  # 1 chunk -> posts fine
        "b": [{"page": 1, "text": "bravo charlie delta"}],     # 3 chunks @ words=1
        "c": [{"page": 1, "text": "echo"}],                    # 1 chunk -> posts fine
    }
    monkeypatch.setattr(
        pipe, "pdf_to_pages",
        lambda path: pages_by_doc[pipe.doc_id_of(path)],
    )
    # Calls in order: a#0 (1,ok) b#0 (2,ok) b#1 (3,FAILS) -- b#2 never attempted.
    engine = FakeEngine(fail_post_at=3)
    monkeypatch.setattr(pipe, "get_provider", lambda s: engine)

    s = Settings(index_dir=str(tmp_path), chunk_words=1, chunk_overlap=0)
    events = []
    corpus = pipe.ingest_into_index(
        ["/x/a.pdf", "/x/b.pdf", "/x/c.pdf"], settings=s,
        on_event=lambda d, st, i: events.append((d, st)),
    )

    assert ("b", "failed") in events
    assert ("a", "done") in events and ("c", "done") in events
    assert set(corpus.doc_ids()) == {"a", "c"}            # b never got lineage

    # Persisted to disk DURING the run, not just reconstructable in memory.
    persisted = Corpus.load(s.index_dir)
    assert set(persisted.doc_ids()) == {"a", "c"}

    # b's one already-posted chunk was best-effort cleaned up engine-side.
    assert ("docket", "b#0") in engine.deleted
    assert "b#1" not in [cid for _, cid in engine.deleted]  # the failing chunk itself was never written


def test_load_samples_skips_already_indexed_and_does_not_repost(tmp_path, monkeypatch):
    # B4 item 3: loading samples twice used to re-POST every chunk every time,
    # 409ing against the engine on the second "Load samples" click.
    import docket.ingest.pipeline as pipe
    from docket.ingest.samples import load_samples

    engine = FakeEngine()
    monkeypatch.setattr(pipe, "get_provider", lambda s: engine)
    s = Settings(index_dir=str(tmp_path))

    load_samples(settings=s)
    first_count = len(engine.posted)
    assert first_count > 0

    load_samples(settings=s)  # second click
    assert len(engine.posted) == first_count  # nothing re-POSTed


def test_capacity_reports_disk_bound(tmp_path):
    # The local store is lineage-only; capacity is disk-bound (engine-side
    # storage owns text and vectors).
    from docket.web.observability import capacity_stats

    s = Settings(index_dir=str(tmp_path))
    c = Corpus()
    c.add(chunk_pages([{"page": 1, "text": "alpha " * 60}], doc_id="d1", words=20, overlap=5))
    c.save(s.index_dir)

    cap = capacity_stats(s)
    assert cap["binding_constraint"] == "disk"
    assert cap["disk_bound_documents_est"] >= 0
    assert cap["remaining_documents_est"] == cap["disk_bound_documents_est"]


def test_generate_report_and_render():
    r = retriever_over(acme_chunks())
    res = answer("What was revenue?", retriever=r, provider=FakeEngine(), settings=Settings())
    report = generate_report(res, title="Acme 10-K")
    md = to_markdown(report)
    assert "# Acme 10-K" in md
    assert "🟢" in md  # reliability banner
    assert "acme-10k" in md
